#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:38:44 2020

@author: Carlos Gómez-Huélamo

SmartMOT: Code to track the detections given by a sensor fusion algorithm (converted into Bird's Eye View (Image frame, z-axis inwards 
with the origin located at the top-left corner) using the SORT (Simple Online and Real-Time Tracking) algorithm as backbone,
and filtered by a Monitored Lanes-based Attention module

Communications are based on ROS (Robot Operating Sytem)

Inputs: 3D Object Detections topic
Outputs: Tracked obstacles topic and monitors information (collision prediction, front obstacle, etc.)

Note that each obstacle shows an unique ID in addition to its semantic information (person, car, ...), 
in order to make easier the decision making processes.
"""

# General purpose imports

import os
import sys
import time
import cv2
import numpy as np
import math
import matplotlib
import git
matplotlib.use('Agg') # In order to avoid: RuntimeError: main thread is not in main loop exception

from argparse import ArgumentParser

# Sklearn imports

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ROS imports

import rospy
import visualization_msgs.msg
import sensor_msgs.msg
import nav_msgs.msg
import std_msgs.msg
import tf 

from t4ac_msgs.msg import BEV_detection, BEV_detections_list, MonitorizedLanes, Node, Obstacle
from message_filters import TimeSynchronizer, ApproximateTimeSynchronizer, Subscriber

# Custom functions imports

repo = git.Repo(__file__, search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from src.aux_functions import geometric_functions
from src.aux_functions import monitors_functions
from src.aux_functions import sort_functions
from src.aux_functions import tracking_functions

# Global variables

detection_threshold = 0.3
max_age = 3
min_hits = 1 # 1
kitti = 0
n = 4 # Predict x bounding boxes ahead
seconds_ahead = 3 # Predict all trajectories at least x seconds ahead
    
filename = ''
path = os.path.curdir + '/results' + filename
header_synchro = 100
slop = 1.0

#######################################

# Main class

class SmartMOT:
    def __init__(self):
        # Auxiliar variables
       
        self.use_mot = True
        self.init_scene = False
        self.use_gaussian_noise = True
        self.filter_hdmap = True
        self.detection_threshold = detection_threshold
        self.colours = np.random.rand(32,3)

        # Display config

        self.image_width = 0 # This value will be updated
        self.image_height = 1000 # Pixels 
        self.shapes = []
        self.view_ahead = 0 

        # Ego-vehicle and Prediction variables

        self.n = n 
        self.ego_vehicle_x, self.ego_vehicle_y = 0.0,0.0
        self.ego_orientation_cumulative_diff = 0 # Cumulative orientation w.r.t the original 
                                                 # orientation of the ego-vehicle (radians)
        self.initial_angle = False
        self.previous_angle = float(0)
        self.previous_yaw = float(0)
        self.current_yaw = float(0) 
        self.ego_braking_distance = 0
        self.ego_dimensions = np.array([4.4,  # Length
                                        1.8]) # Width
        self.ego_forecasted_bboxes = []
        self.seconds_ahead = seconds_ahead

        # SmartMOT Callback 

        self.start = float(0)
        self.end = float(0)
        self.frame_no = 0
        self.avg_fps = float(0)
        self.write_video = False
        self.video_flag = False 

        # Emergency break

        self.cont = 0
        self.collision_flag = std_msgs.msg.Bool()
        self.collision_flag.data = False
        self.nearest_object_in_route = 50000
        self.geometric_monitorized_area = []

        # Pedestrian Crossing

        self.closest_crosswalk = []

        # Merge (Give Way / STOP)

        self.monitorized_intersections_rosmsg = []

        # Overtaking 

        self.front_obstacle = Obstacle() 

        # Arguments from ROS params

        root = rospy.get_param('/t4ac/perception/tracking_and_prediction/classic/t4ac_smartmot_ros/t4ac_smartmot_ros_node/root')

        self.display = rospy.get_param(os.path.join(root,"display"))
        self.trajectory_prediction = rospy.get_param(os.path.join(root,"trajectory_prediction"))
        self.ros = rospy.get_param(os.path.join(root,"use_ros"))
        self.grid = rospy.get_param(os.path.join(root,"use_grid"))
        self.map_frame = rospy.get_param('t4ac/frames/map')
        self.lidar_frame = rospy.get_param('t4ac/frames/lidar')
        
        # ROS publishers

        monitorized_area_topic = rospy.get_param(os.path.join(root,"pub_rectangular_monitorized_area_marker"))
        self.pub_monitorized_area = rospy.Publisher(monitorized_area_topic, visualization_msgs.msg.Marker, queue_size = 20)
        particular_monitorized_area_markers_list_topic = rospy.get_param(os.path.join(root,"pub_particular_monitorized_areas_marker"))
        self.pub_particular_monitorized_area_markers_list = rospy.Publisher(particular_monitorized_area_markers_list_topic, visualization_msgs.msg.MarkerArray, queue_size = 20)
        bev_sort_tracking_markers_list_topic = rospy.get_param(os.path.join(root,"pub_BEV_tracked_obstacles_marker"))
        self.pub_bev_sort_tracking_markers_list = rospy.Publisher(bev_sort_tracking_markers_list_topic, visualization_msgs.msg.MarkerArray, queue_size = 20)
        ego_vehicle_forecasted_trajectory_markers_list = rospy.get_param(os.path.join(root,"pub_ego_vehicle_forecasted_trajectory_marker"))
        self.pub_ego_vehicle_forecasted_trajectory_markers_list = rospy.Publisher(ego_vehicle_forecasted_trajectory_markers_list, visualization_msgs.msg.MarkerArray, queue_size = 20)
        # self.pub_ego_vehicle_forecasted_trajectory_markers_list = rospy.Publisher(ego_vehicle_forecasted_trajectory_markers_list, nav_msgs.msg.Path, queue_size = 20)
        topic_bb_forecasted = os.path.join(root,"object_forecasted")
        self.pub_object_forecasted_trajectory_markers_list = rospy.Publisher(topic_bb_forecasted, visualization_msgs.msg.MarkerArray, queue_size = 20)

        ## Monitors

        predicted_collision_topic = rospy.get_param(os.path.join(root,"pub_predicted_collision")) # Unexpected Pedestrian or Collision
        self.pub_predicted_collision = rospy.Publisher(predicted_collision_topic, std_msgs.msg.Bool, queue_size = 1)
        nearest_object_distance_topic = rospy.get_param(os.path.join(root,"pub_nearest_object_distance")) # ACC or Obstacle in the route
        self.pub_nearest_object_distance = rospy.Publisher(nearest_object_distance_topic, std_msgs.msg.Float64, queue_size = 1)
        pedestrian_crossing_occupied_topic = rospy.get_param(os.path.join(root,"pub_pedestrian_crossing_occupied")) # Pedestrian Crossing
        self.pub_pedestrian_crossing_occupied = rospy.Publisher(pedestrian_crossing_occupied_topic, std_msgs.msg.Bool, queue_size = 1)
        merge_occupied_topic = rospy.get_param(os.path.join(root,"pub_merge_occupied")) # Give Way / Stop
        self.pub_merge_occupied = rospy.Publisher(merge_occupied_topic, std_msgs.msg.Bool, queue_size = 1)
        front_obstacle_topic = rospy.get_param(os.path.join(root,"pub_front_obstacle")) # Overtaking
        self.pub_front_obstacle = rospy.Publisher(front_obstacle_topic, Obstacle, queue_size = 1)
        left_lane_occupied_topic = rospy.get_param(os.path.join(root,"pub_left_lane_occupied")) # Overtaking
        self.pub_left_lane_occupied = rospy.Publisher(left_lane_occupied_topic, std_msgs.msg.Bool, queue_size = 1)
        right_lane_occupied_topic = rospy.get_param(os.path.join(root,"pub_right_lane_occupied")) # Overtaking
        self.pub_right_lane_occupied = rospy.Publisher(right_lane_occupied_topic, std_msgs.msg.Bool, queue_size = 1)

        # ROS subscribers

        if not self.filter_hdmap:
            self.rc_max = rospy.get_param('/t4ac/control/classic/t4ac_lqr_ros/t4ac_lqr_ros_node/rc_max')
            self.sub_road_curvature = rospy.Subscriber("/t4ac/control/rc", std_msgs.msg.Float64, self.road_curvature_callback)
            
        detections_topic = rospy.get_param(os.path.join(root,"sub_BEV_merged_obstacles"))
        location_topic = rospy.get_param(os.path.join(root,"sub_localization_pose"))
        monitorized_lanes_topic = rospy.get_param(os.path.join(root,"sub_monitorized_lanes"))

        self.sub_detections = Subscriber(detections_topic, BEV_detections_list)
        self.sub_location = Subscriber(location_topic, nav_msgs.msg.Odometry)
        self.sub_monitorized_lanes = Subscriber(monitorized_lanes_topic, MonitorizedLanes)

        self.sub_closest_crosswalk = rospy.Subscriber("/t4ac/mapping/monitor/crosswalk_marker", visualization_msgs.msg.Marker, self.closest_crosswalk_callback)
        self.sub_monitorized_intersections = rospy.Subscriber(monitorized_intersections_topic, MonitorizedLanes, self.monitorized_intersections_callback)

        self.ts = ApproximateTimeSynchronizer([self.sub_detections, 
                                               self.sub_location, 
                                               self.sub_monitorized_lanes], 
                                               header_synchro, slop)
        self.ts.registerCallback(self.smartmot_callback)
        
        self.listener = tf.TransformListener()

    def road_curvature_callback(self, msg):
        """
        """

        rc = msg.data
        rc_ratio = rc/self.rc_max

        self.geometric_monitorized_area
        
        xmax = float(rc_ratio*self.ego_braking_distance*1.4) # We consider a 40 % safety factor 

        if xmax > 30:
            xmax = 30
        elif xmax < 12 and rc_ratio > 0.8: # rc_ratio < 0.8, we are in curve, so x_max must be reduced to 0
            xmax = 12
        
        lateral = 2.6
        xmin = 0
        ymin = rc_ratio * (-lateral)
        ymax = rc_ratio * lateral

        self.geometric_monitorized_area = [xmax,xmin,ymax,ymin]

        geometric_monitorized_area_marker = visualization_msgs.msg.Marker()

        geometric_monitorized_area_marker.header.frame_id = self.lidar_frame
        geometric_monitorized_area_marker.ns = "geometric_monitorized_area"
        geometric_monitorized_area_marker.action = geometric_monitorized_area_marker.ADD
        geometric_monitorized_area_marker.lifetime = rospy.Duration.from_sec(1)
        geometric_monitorized_area_marker.type = geometric_monitorized_area_marker.CUBE

        geometric_monitorized_area_marker.color.r = 1.0
        geometric_monitorized_area_marker.color.g = 1.0
        geometric_monitorized_area_marker.color.b = 1.0
        geometric_monitorized_area_marker.color.a = 0.4

        geometric_monitorized_area_marker.pose.position.x = xmax/2   
        geometric_monitorized_area_marker.pose.position.y = (ymin+ymax)/2
        geometric_monitorized_area_marker.pose.position.z = -2.0
        
        geometric_monitorized_area_marker.scale.x = xmax
        geometric_monitorized_area_marker.scale.y = abs(ymin) + ymax
        geometric_monitorized_area_marker.scale.z = 0.2

        self.pub_monitorized_area.publish(geometric_monitorized_area_marker)
    
    # TODO: Implement in mapping_layer
    def closest_crosswalk_callback(self, msg):
        """
        """
        del self.closest_crosswalk[:]
        aux = msg.points[:-1]
        for p_aux in aux:
            p = Node()
            p.x = p_aux.x
            p.y = -p_aux.y
            self.closest_crosswalk.append(p)

    def smartmot_callback(self, detections_rosmsg, odom_rosmsg, monitorized_lanes_rosmsg):
        """
        """
        
        print(">>>>>>>>>>>>")
        
        try:                                                         # Target        # Pose
            (translation,quaternion) = self.listener.lookupTransform(self.map_frame, self.lidar_frame, rospy.Time(0)) 
            # rospy.Time(0) get us the latest available transform
            rot_matrix = tf.transformations.quaternion_matrix(quaternion)
            
            self.tf_map2lidar = rot_matrix
            self.tf_map2lidar[:3,3] = self.tf_map2lidar[:3,3] + translation # This matrix transforms local to global coordinates

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("\033[1;33m"+"TF exception"+'\033[0;m')
            self.ts.registerCallback(self.smartmot_callback)

        self.start = time.time()

        # Initialize the scene

        if not self.init_scene:
            self.real_height = detections_rosmsg.front - detections_rosmsg.back # Grid height (m)
            self.real_front = detections_rosmsg.front + self.view_ahead # To study obstacles (view_head) meters ahead
            self.real_width = -detections_rosmsg.left + detections_rosmsg.right # Grid width (m)
            self.real_left = -detections_rosmsg.left

            r = float(self.image_height)/self.real_height

            self.image_width = int(round(self.real_width*r)) # Grid width (pixels)
            self.image_front = int(round(self.real_front*r))
            self.image_left = int(round(self.real_left*r))

            self.shapes = (self.real_front,self.real_left,self.image_front,self.image_left)
            self.scale_factor = (self.image_height/self.real_height,self.image_width/self.real_width)

            # Our centroid is defined in BEV camera coordinates but centered in the ego-vehicle. 
            # We need to traslate it to the top-left corner (required by the SORT algorithm)

            self.tf_bevtl2bevcenter_m = np.array([[1.0, 0.0, 0.0, self.shapes[1]],
                                                  [0.0, 1.0, 0.0, self.shapes[0]],
                                                  [0.0, 0.0, 1.0, 0.0], 
                                                  [0.0, 0.0, 0.0, 1.0]])

            # From BEV centered to BEV top-left (in pixels)

            self.tf_bevtl2bevcenter_px = np.array([[1.0, 0.0, 0.0, -self.shapes[3]], 
                                                   [0.0, 1.0, 0.0, -self.shapes[2]],
                                                   [0.0, 0.0, 1.0, 0.0], 
                                                   [0.0, 0.0, 0.0, 1.0]])

            # Rotation matrix from LiDAR frame to BEV frame (to convert to global coordinates)
    
            self.tf_lidar2bev = np.array([[0.0,-1.0,0.0,0.0],
                                          [-1.0,0.0,0.0,0.0],
                                          [0.0,0.0,-1.0,0.0],
                                          [0.0,0.0,0.0, 1.0]])

            # Initialize the regression model to calculate the braking distance

            self.velocity_braking_distance_model = monitors_functions.fit_velocity_braking_distance_model()

            # Tracker
            
            self.mot_tracker = sort_functions.Sort(max_age,min_hits,n,self.shapes,self.trajectory_prediction,self.filter_hdmap)

            self.init_scene = True

        output_image = np.ones((self.image_height,self.image_width,3),dtype = "uint8") # Black image to represent the scene

        # print("----------------")

        # Initialize ROS and monitors variables
        
        self.trackers_marker_list = visualization_msgs.msg.MarkerArray() # Initialize trackers list
        monitorized_area_colours = []

        trackers_in_route = 0
        front_obstacle = Obstacle()
        front_obstacle.dist2ego = 50000
        front_obstacle.twist.linear.x = 50000
        front_obstacle.twist.angular.z = 50000
        left_lane_occupied = std_msgs.msg.Bool()
        right_lane_occupied = std_msgs.msg.Bool()
        nearest_distance = std_msgs.msg.Float64()
        nearest_distance.data = float(self.nearest_object_in_route)
        pedestrian_crossing_occupied = std_msgs.msg.Bool()
        pedestrian_crossing_occupied.data = False
        merge_occupied = std_msgs.msg.Bool()
        merge_occupied.data = False

        world_features = []
        trackers = []
        dynamic_trackers = []
        static_trackers = []
        ndt = 0 # Number of dynamic trackers
        
        timer_rosmsg = detections_rosmsg.header.stamp.to_sec()

        # Predict the ego-vehicle trajectory

        for lane in monitorized_lanes_rosmsg.lanes:
            if (lane.role == "current" and len(lane.left.way) >= 2):
                monitors_functions.new_ego_vehicle_prediction(self,odom_rosmsg, lane)
                # monitors_functions.ego_vehicle_prediction(self,odom_rosmsg)
        # print("Ego vehicle braking distance: ", float(self.ego_braking_distance))

        # print("Forecasted ego: ", self.ego_forecasted_bboxes)

        # Convert input data to bboxes to perform Multi-Object Tracking 

        # print("\nNumer of total detections: ", len(detections_rosmsg.bev_detections_list))
        bboxes_features,types = sort_functions.bbox_to_xywh_cls_conf(self,detections_rosmsg,output_image)
        # print("Number of relevant detections: ", len(bboxes_features)) # score > detection_threshold

        start_1 = end_1 = 0.0

        if not self.use_mot:
        # Only detection
            start_1 = time.time()

            nearest_object_distance = 50000
            object_in_route = False
            dist = 99999

            # Monitorized Lanes (Current, Back, Left, Right)

            # print("Flag collision: ", self.collision_flag.data)
            if not self.collision_flag.data:
                k = 0
                # print("Len objects: ", len(bboxes_features))
                for bbox,type_object in zip(bboxes_features,types):
                    for lane in monitorized_lanes_rosmsg.lanes: 
                        if (lane.role == "current" and len(lane.left.way) >= 2):
                            # print("CURRENT")
                            object_in_route = True
                            detection = Node()
                            bbox = bbox.reshape(1,-1)
                            detection.x = bbox[0,0]
                            detection.y = -bbox[0,1] # N.B. In OpenDrive this coordinate is the opposite

                            in_polygon, in_road, particular_monitorized_area, dist2centroid = monitors_functions.inside_lane(lane,detection,type_object)

                            if in_polygon or in_road:
                                #distance_to_object = monitors_functions.calculate_distance_to_nearest_object_inside_route(monitorized_lanes_rosmsg,global_pos)

                                ego_x_global = odom_rosmsg.pose.pose.position.x
                                ego_y_global = -odom_rosmsg.pose.pose.position.y
                                dist2object = math.sqrt(pow(ego_x_global-detection.x,2)+pow(ego_y_global-detection.y,2))

                                w = detections_rosmsg.bev_detections_list[k].w
                                l = detections_rosmsg.bev_detections_list[k].l
                                yaw = detections_rosmsg.bev_detections_list[k].o
                                vel_lin = detections_rosmsg.bev_detections_list[k].vel_lin
                                vel_ang = -detections_rosmsg.bev_detections_list[k].vel_ang

                                object_forecasted_bboxes = monitors_functions.aux_prediction(self,w,l,yaw,vel_lin,vel_ang,detection)
                                flag_collision = monitors_functions.aux_predict_collision(self.ego_forecasted_bboxes,object_forecasted_bboxes)
        
                                # if distance_to_object < self.nearest_object_in_route:
                                if dist2object < nearest_object_distance:
                                    front_obstacle.type = str(type_object)
                                    front_obstacle.dist2ego = dist2object
                                    front_obstacle.twist.linear.x = detections_rosmsg.bev_detections_list[k].vel_lin
                                    front_obstacle.twist.angular.z = detections_rosmsg.bev_detections_list[k].vel_ang
                                    # self.nearest_object_in_route = distance_to_object
                                    # nearest_distance.data = float(self.nearest_object_in_route)
                                    nearest_object_distance = dist2object
                                    nearest_distance.data = nearest_object_distance
                                
                                if flag_collision:
                                    self.collision_flag.data = True
                                    self.pub_predicted_collision.publish(self.collision_flag)
                                    self.cont = 0
                                    break
                        elif (lane.role == "left" and len(lane.left.way) >= 2):
                            detection = Node()
                            bbox = bbox.reshape(1,-1)
                            detection.x = bbox[0,0]
                            detection.y = -bbox[0,1] # N.B. In OpenDrive this coordinate is the opposite
                            _, left_lane_occupied.data, _, _ = monitors_functions.inside_lane(lane,detection,type_object)
                        elif (lane.role == "right" and len(lane.left.way) >= 2):
                            detection = Node()
                            bbox = bbox.reshape(1,-1)
                            detection.x = bbox[0,0]
                            detection.y = -bbox[0,1] # N.B. In OpenDrive this coordinate is the opposite
                            _, right_lane_occupied.data, _, _ = monitors_functions.inside_lane(lane,detection,type_object)
                    else: 
                        continue 
                    break
                    k = k+1
                if not object_in_route:
                    self.collision_flag.data = False
                    nearest_distance.data = float(50000)
            else:
                k = 0
                for bbox,type_object in zip(bboxes_features,types):
                    for lane in monitorized_lanes_rosmsg.lanes: 
                        if (lane.role == "current" and len(lane.left.way) >= 2):
                            detection = Node()
                            bbox = bbox.reshape(1,-1)
                            detection.x = bbox[0,0]
                            detection.y = -bbox[0,1] # N.B. In OpenDrive this coordinate is the opposite

                            in_polygon, in_road, particular_monitorized_area, dist2centroid = monitors_functions.inside_lane(lane,detection,type_object)
                            
                            if in_polygon or in_road:
                                ego_x_global = odom_rosmsg.pose.pose.position.x
                                ego_y_global = -odom_rosmsg.pose.pose.position.y
                                dist2object = math.sqrt(pow(ego_x_global-detection.x,2)+pow(ego_y_global-detection.y,2))
                                # if aux < dist:
                                if dist2object < nearest_object_distance:
                                    front_obstacle.type = str(type_object)
                                    front_obstacle.dist2ego = dist2object
                                    front_obstacle.twist.linear.x = detections_rosmsg.bev_detections_list[k].vel_lin
                                    front_obstacle.twist.angular.z = detections_rosmsg.bev_detections_list[k].vel_ang
                                    # dist = aux
                                    nearest_object_distance = dist2object
                                    nearest_distance.data = nearest_object_distance
                            break
                    k = k+1

                if nearest_distance.data > 5:
                    self.cont += 1
                else:
                    self.cont = 0
            if self.cont >= 3:
                self.collision_flag.data = False

            # Pedestrian Crossing

            if self.closest_crosswalk:
                for bbox,type_object in zip(bboxes_features,types):
                    detection = Node()
                    bbox = bbox.reshape(1,-1)
                    detection.x = bbox[0,0]
                    detection.y = -bbox[0,1] # N.B. In OpenDrive this coordinate is the opposite

                    pedestrian_crossing_flag = monitors_functions.inside_polygon(detection,self.closest_crosswalk)
                    pedestrian_crossing_occupied.data = pedestrian_crossing_flag
                    if pedestrian_crossing_occupied.data:
                        break

            # Monitorized Intersections (Split, Merge, Intersection)

            if self.monitorized_intersections_rosmsg:
                for bbox,type_object in zip(bboxes_features,types):
                    for lane in self.monitorized_intersections_rosmsg.lanes: 
                        if (lane.role == "merge" and len(lane.left.way) >= 2):
                            detection = Node()
                            bbox = bbox.reshape(1,-1)
                            detection.x = bbox[0,0]
                            detection.y = -bbox[0,1] # N.B. In OpenDrive this coordinate is the opposite

                            merge_flag,_,_,_ = monitors_functions.inside_lane(lane,detection,type_object)
                            merge_occupied.data = merge_flag
                            if merge_occupied.data:
                                # print("Merge 2: ", merge_occupied.data, id(merge_occupied))
                                break
                    else: 
                        continue
                    break

            # Monitors

            # print("Collision: ", self.collision_flag.data)
            # print("Front obstacle dist: ", front_obstacle.dist2ego)
            # print("Front obstacle vel lin: ", front_obstacle.twist.linear.x)
            # print("Left lane occupied: ", left_lane_occupied.data)
            # print("Right lane occupied: ", left_lane_occupied.data)
            # print("Merging Occupied: ", self.merge_occupied.data)
            # self.pub_nearest_object_distance.publish(nearest_distance)

            self.pub_predicted_collision.publish(self.collision_flag)
            self.pub_front_obstacle.publish(front_obstacle)
            self.pub_pedestrian_crossing_occupied.publish(pedestrian_crossing_occupied)
            self.pub_merge_occupied.publish(merge_occupied)
            self.pub_left_lane_occupied.publish(left_lane_occupied)
            self.pub_right_lane_occupied.publish(right_lane_occupied)

            end_1 = time.time()
            # hz_1 = 1/(end_1-start_1)
            # print("Hz: ", hz_1)
        else:
            # Multi-Object Tracking

            # TODO: Publish on tracked_obstacle message instead of visualization marker
            # TODO: Evaluate the predicted position to predict its influence in a certain use case

            ego_vel_px = 0 # TODO: Delete this
            angle_bb = 0 # TODO: Delete this
            
            if (len(bboxes_features) > 0): # At least one object was detected
                trackers,object_types,object_scores,object_observation_angles,dynamic_trackers,static_trackers = self.mot_tracker.update(bboxes_features,types,
                                                                                                                                        ego_vel_px,
                                                                                                                                        self.tf_map2lidar,
                                                                                                                                        self.shapes,
                                                                                                                                        self.scale_factor,
                                                                                                                                        monitorized_lanes_rosmsg,
                                                                                                                                        timer_rosmsg,
                                                                                                                                        angle_bb,
                                                                                                                                        self.geometric_monitorized_area)
                                                                                                                                        
                print("Number of trackers: ", len(trackers))
                if len(dynamic_trackers.shape) == 3:
                    print("Dynamic trackers", dynamic_trackers.shape[1])
                    ndt = dynamic_trackers.shape[1]
                else:
                    print("Dynamic trackers: ", 0)
                    ndt = 0
                print("Static trackers: ", static_trackers.shape[0])

                id_nearest = -1
                
                if (len(trackers) > 0): # At least one object was tracked
                    for i,tracker in enumerate(trackers): 
                        object_type  = object_types[i]
                        object_score = object_scores[i]
                        object_rotation = np.float64(object_observation_angles[i,0])
                        object_observation_angle = np.float64(object_observation_angles[i,1])

                        color = self.colours[tracker[5].astype(int)%32]
                        #print("hago append")
                        monitorized_area_colours.append(color)

                        if self.ros:
                            # world_features = monitors_functions.tracker_to_topic_real(self,tracker,object_type,color) # world_features (w,l,h,x,y,z,id)
                            world_features = monitors_functions.tracker_to_topic(self,tracker,object_type,color)
 
                            if kitti:
                                num_image = detections_rosmsg.header.seq-1 # Number of image in the dataset, e.g. 0000.txt -> 0
                                object_properties = object_observation_angle,object_rotation,object_score
                                monitors_functions.store_kitti(num_image,path,object_type,world_features,object_properties)

                        if (self.display):
                            print("Tracker: ", tracker)
                            my_thickness = -1
                            geometric_functions.compute_and_draw(tracker,color,my_thickness,output_image)

                        label = 'ID %06d'%tracker[5].astype(int)
                        cv2.putText(output_image,label,(tracker[0].astype(int),tracker[1].astype(int)-20), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2)
                        cv2.putText(output_image,object_type,(tracker[0].astype(int),tracker[1].astype(int)-40), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2)
                        
                        # Evaluate if there is some obstacle in lane and calculate nearest distance
                        
                        if self.filter_hdmap:
                            if tracker[-1]: # In route, last element of the array 
                                trackers_in_route += 1
                                obstacle_local_position = np.zeros((1,9))

                                obstacle_local_position[0,7] = world_features[3]
                                obstacle_local_position[0,8] = world_features[4]

                                obstacle_global_position = sort_functions.store_global_coordinates(self.tf_map2lidar,obstacle_local_position)
        
                                #distance_to_object = monitors_functions.calculate_distance_to_nearest_object_inside_route(monitorized_lanes_rosmsg,obstacle_global_position)
                                
                                
                                detection = Node()

                                detection.x = obstacle_global_position[0,0]
                                detection.y = -obstacle_global_position[1,0]
                                
                                ego_x_global = odom_rosmsg.pose.pose.position.x
                                ego_y_global = -odom_rosmsg.pose.pose.position.y

                                distance_to_object = math.sqrt(pow(ego_x_global-detection.x,2)+pow(ego_y_global-detection.y,2))

                                if distance_to_object < self.nearest_object_in_route:
                                    id_nearest = tracker[5]
                                    self.nearest_object_in_route = distance_to_object
                        else:
                            # Evaluate in the geometric monitorized area

                            x = world_features[3]
                            y = world_features[4]


                            print("main")
                            print("goemetric area: ", self.geometric_monitorized_area)
                            print("x y: ", x,y)

                            if (x < self.geometric_monitorized_area[0] and x > self.geometric_monitorized_area[1]
                                and y < self.geometric_monitorized_area[2] and y > self.geometric_monitorized_area[3]):
                                trackers_in_route += 1
                                self.cont = 0
                                print("\n\n\nDentro")
                                distance_to_object = math.sqrt(pow(x,2)+pow(y,2))
                                print("Nearest: ", self.nearest_object_in_route)
                                print("distance: ", distance_to_object)
                                if distance_to_object < self.nearest_object_in_route:
                                    self.nearest_object_in_route = distance_to_object
                        
                    print("Collision: ", self.collision_flag.data)
                    print("trackers in route: ", trackers_in_route)
                    print("Distance nearest: ", self.nearest_object_in_route)
                    if self.collision_flag.data and (trackers_in_route == 0 or (self.nearest_object_in_route > 12 and self.abs_vel < 1)):
                        print("suma A")
                        self.cont += 1
                    else:
                        self.cont == 0

                    nearest_distance.data = self.nearest_object_in_route
    
                    if(self.trajectory_prediction):

                        collision_id_list = [[],[]]
                        
                        # Evaluate collision with dynamic trackers
                        
                        for a in range(dynamic_trackers.shape[1]):
                            for j in range(self.n.shape[0]):
                                e = dynamic_trackers[j][a]
                                color = self.colours[e[5].astype(int)%32]
                                my_thickness = 2
                                geometric_functions.compute_and_draw(e,color,my_thickness,output_image)
                                
                                if (self.ros):
                                    object_type = "trajectory_prediction"
                                    monitors_functions.tracker_to_topic_real(self,e,object_type,color,j) 

                            # Predict possible collision (including the predicted bounding boxes)
                            
                            # collision_id,index_bb = monitors_functions.predict_collision(self.ego_predicted,dynamic_trackers[:,a]) 
                            collision_id,index_bb = monitors_functions.predict_collision(self.ego_forecasted_bboxes,dynamic_trackers[:,a])
                            
                            if collision_id != -1:
                                collision_id_list[0].append(collision_id)
                                collision_id_list[1].append(index_bb)     

                        # Evaluate collision with static trackers
                        
                        for b in static_trackers:
                            if b[-1]: # In route, last element of the array                     
                                # collision_id,index_bb = monitors_functions.predict_collision(self.ego_predicted,b,static=True) # Predict possible collision
                                collision_id,index_bb = monitors_functions.predict_collision(self.ego_forecasted_bboxes,b,static=True)

                                if (collision_id != -1): 
                                    collision_id_list[0].append(collision_id)
                                    collision_id_list[1].append(index_bb)
                        

                        #if self.nearest_object_in_route < self.ego_braking_distance:
                        #    self.collision_flag.data = True

                        





                        # Collision

                        if not self.collision_flag.data:
                            if not collision_id_list[0]: # Empty
                            #if id_nearest == -1:
                                collision_id_list[0].append(-1) # The ego-vehicle will not collide with any object
                                collision_id_list[1].append(-1) 
                                self.collision_flag.data = False 
                            elif collision_id_list[0] and (self.nearest_object_in_route < self.ego_braking_distance or self.nearest_object_in_route < 12):
                            #elif id_nearest != -1 and (self.nearest_object_in_route < self.ego_braking_distance or self.nearest_object_in_route < 12):
                                self.collision_flag.data = True  
                                self.cont = 0 

                        #print("Collision id list: ", collision_id_list)
                        if (len(collision_id_list)>1):
                            message = 'Predicted collision with objects: ' + str(collision_id_list[0])
                        else:
                            message = 'Predicted collision with object: ' + str(collision_id_list[0])
        
                        cv2.putText(output_image,message,(30,140), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2) # Predicted collision message   
                else:
                    print("\033[1;33m"+"No object to track"+'\033[0;m')
                    monitors_functions.empty_trackers_list(self)  

                    if self.collision_flag.data:
                        print("suma B")
                        self.cont += 1
                
            else: 
                print("\033[1;33m"+"No objects detected"+'\033[0;m')
                monitors_functions.empty_trackers_list(self)

                if self.collision_flag.data:
                    # print("suma C")
                    self.cont += 1
            
            # print("cont: ", self.cont)
            if self.cont >= 3:
                self.collision_flag.data = False
                self.nearest_object_in_route = 50000
                nearest_distance.data = float(self.nearest_object_in_route)

            
            self.end = time.time()
                    
            fps = 1/(self.end-self.start)

            self.avg_fps += fps 
            self.frame_no += 1
            
            # print("SORT time: {}s, fps: {}, avg fps: {}".format(round(self.end-self.start,3), round(fps,3), round(self.avg_fps/self.frame_no,3)))

            message = 'Trackers: ' + str(len(trackers))
            cv2.putText(output_image,message,(30,20), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2)
            message = 'Dynamic trackers: ' + str(ndt)
            cv2.putText(output_image,message,(30,50), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2)
            try:
                message = 'Static trackers: ' + str(static_trackers.shape[0])
            except:
                message = 'Static trackers: ' + str(0)
            cv2.putText(output_image,message,(30,80), cv2.FONT_HERSHEY_PLAIN, 1.5, [255,255,255], 2)
            
            # Publish the list of tracked obstacles and predicted collision
            
            # print("Data: ", nearest_distance.data)
            # print("Collision: ", self.collision_flag.data)
            # print("Trackers list: ", len(self.trackers_marker_list.markers))
            self.pub_bev_sort_tracking_markers_list.publish(self.trackers_marker_list) 
            self.pub_predicted_collision.publish(self.collision_flag)         
            self.pub_nearest_object_distance.publish(nearest_distance)

            self.particular_monitorized_area_list = self.mot_tracker.get_particular_monitorized_area_list(detections_rosmsg.header.stamp,
                                                                                                        monitorized_area_colours)
            self.pub_particular_monitorized_area_markers_list.publish(self.particular_monitorized_area_list)

            if self.write_video:
                if not self.video_flag:
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    self.output = cv2.VideoWriter("map-filtered-mot.avi", fourcc, 20, (self.image_width, self.image_height))
                self.output.write(output_image)
        
            # Add a grid to appreciate the obstacles coordinates
            
            if (self.grid):
                gap = int(round(self.view_ahead*(self.image_width/self.real_width)))
                geometric_functions.draw_basic_grid(gap,output_image,pxstep=50)
            
            if(self.display):
                cv2.imshow("SORT tracking", output_image)
                cv2.waitKey(1)

def main():
    print("Init the node")

    node_name = rospy.get_param("/t4ac/perception/tracking_and_prediction/classic/t4ac_smartmot_ros/t4ac_smartmot_ros_node/node_name")
    print("Node_name: ", node_name)
    rospy.init_node(node_name, anonymous=True)
    
    SmartMOT()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down ROS Tracking node")

if __name__ == '__main__':
    main()
