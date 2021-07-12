#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:07:29 2020

AB4COGT to SmartMOT

@author: Javier del Egido & Carlos Gómez-Huélamo

"""

# General purpose imports

import math
import numpy as np
from scipy.spatial.distance import euclidean

# ROS imports

import rospy
from carla_msgs.msg import CarlaEgoVehicleInfo
from derived_object_msgs.msg import ObjectArray
from tf.transformations import euler_from_quaternion
from t4ac_msgs.msg import BEV_detection, BEV_detections_list
from visualization_msgs.msg import Marker, MarkerArray

from ros_functions import marker_bb

classification_list = ["unknown",
			           "Unknown_Small",
			           "Unknown_Medium",
			           "Unknown_Big",
			           "Pedestrian",
			           "Cyclist",
			           "Car",
			           "Truck",
			           "Motorcycle",
			           "Other_Vehicle",
			           "Barrier",
			           "Sign"]

class AB4COGT2SORT():
    def rotz(self,t):
        """
        Rotation about the z-axis
        """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  -s,  0],
                         [s,   c,  0],
                         [0,   0,  1]])
    
    def listener(self):

        self.first_time = 0 # Flag
        self.first_seq_value = 0 # Stores first seq value to start on 0

        rospy.init_node('t4ac_ab4cogt_ros_node', anonymous=True)
        rospy.Subscriber("/carla/ego_vehicle/vehicle_info", CarlaEgoVehicleInfo, self.ego_vehicle_callback)
        self.pub_groundtruth_obstacles = rospy.Publisher("/t4ac/perception/detection/BEV_groundtruth_obstacles",\
                                                          BEV_detections_list, queue_size=10)
        self.pub_groundtruth_obstacles_marker = rospy.Publisher("/t4ac/perception/detection/3D_groundtruth_obstacles_marker",\
                                                          MarkerArray, queue_size=10)
        rospy.spin()
        
    def ego_vehicle_callback(self, ego_vehicle_info_msg):
        print("Receiving ego_vehicle")
        self.ego_vehicle_id = ego_vehicle_info_msg.id
        rospy.Subscriber("/carla/objects", ObjectArray, self.callback)
        
    def callback(self, carla_objects_msg):
        """
        """

        groundtruth_obstacles_marker_array = MarkerArray()

        groundtruth_obstacles_list = BEV_detections_list()	
        groundtruth_obstacles_list.header.stamp = carla_objects_msg.header.stamp
        groundtruth_obstacles_list.front = 40
        groundtruth_obstacles_list.back = -15
        groundtruth_obstacles_list.left = -15
        groundtruth_obstacles_list.right = 15

        number_objects = len(carla_objects_msg.objects) # Including the ego-vehicle

        if number_objects > 1:
            if self.first_time ==0:
                self.first_time = 1
                self.first_seq_value = carla_objects_msg.header.seq
                
            # First, find ego_vehicle position

            for obj1 in range(len(carla_objects_msg.objects)):
                identity = carla_objects_msg.objects[obj1].id
                if identity == self.ego_vehicle_id:
                    xyz_ego = carla_objects_msg.objects[obj1].pose.position
                    quat_ego = carla_objects_msg.objects[obj1].pose.orientation
                    break

            location_ego = [xyz_ego.x, xyz_ego.y, xyz_ego.z]

            # frame = carla_objects_msg.header.seq - self.first_seq_value
             
            published_obj = 0
            
            for i in range(len(carla_objects_msg.objects)):
                identity = carla_objects_msg.objects[i].id
                if identity != self.ego_vehicle_id:			
                    xyz = carla_objects_msg.objects[i].pose.position
                    location = [xyz.x, xyz.y, xyz.z]

                    # Only store groundtruth of objects in Lidar range

                    # TODO: Improve with ray tracing!!!
                    # if euclidean(location, location_ego) < 25: # Compare to LiDAR range
                        # Get data from object topic
                    
                    quat_xyzw 	= carla_objects_msg.objects[i].pose.orientation
                    dim 		= carla_objects_msg.objects[i].shape.dimensions
                    l,w,h       = dim
                    label 		= classification_list[carla_objects_msg.objects[i].classification]
                    
                    # Calculate heading and alpha (obs_angle)

                    quaternion 	= np.array((quat_xyzw.x, quat_xyzw.y, quat_xyzw.z, quat_xyzw.w))
                    heading 	= euler_from_quaternion(quaternion)[2] #- np.pi/2
                    beta = np.arctan2(xyz.x - xyz_ego.x, xyz_ego.y - xyz.y)
                    obs_angle = ( (heading) + (beta) ) # Alpha (according to KITTI doc)

                    location_local = (np.asarray(location) - np.asarray(location_ego)).tolist()

                    quaternion_ego = np.array((quat_ego.x, quat_ego.y, quat_ego.z, quat_ego.w))
                    heading_ego    = euler_from_quaternion(quaternion_ego)[2]
                    R = self.rotz(-heading_ego)
                    location_local = np.dot(R, location_local) # [0:2]
                
                    if (location_local[0] > groundtruth_obstacles_list.back) and (location_local[0] < groundtruth_obstacles_list.front) \
                        and (location_local[1] > groundtruth_obstacles_list.left) and (location_local[1] < groundtruth_obstacles_list.right):
                        published_obj += 1

                        # Local position, heading and velocities (w.r.t map_frame)

                        xyz     = location_local
                        heading = -heading # In CARLA it is the opposite
                        vel_x = carla_objects_msg.objects[i].twist.linear.x
                        vel_y = carla_objects_msg.objects[i].twist.linear.y
                        vel_lin = math.sqrt(pow(vel_x,2)+pow(vel_y,2))
                        vel_ang = carla_objects_msg.objects[i].twist.angular.z

                        # Get 3D bounding box corners

                        x_lidar = xyz[0]
                        y_lidar = xyz[1]
                        z_lidar = xyz[2]-3

                        x_corners = [-l/2,-l/2, l/2, l/2,-l/2,-l/2,l/2, l/2] 
                        y_corners = [ w/2,-w/2, w/2,-w/2, w/2,-w/2,w/2,-w/2]
                        z_corners = [-h/2,-h/2,-h/2,-h/2, h/2, h/2,h/2, h/2]

                        if heading > np.pi:
                            heading = heading - np.pi
                        R = self.rotz(heading-np.pi/2)

                        corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
                        corners_3d = corners_3d + np.vstack([x_lidar, y_lidar, z_lidar])
        
                        # Publish in ROS topic

                        groundtruth_obstacle = BEV_detection()

                        groundtruth_obstacle.type = label
                        groundtruth_obstacle.score = 1.0 # Groundtruth
                        groundtruth_obstacle.object_id = int(identity)

                        groundtruth_obstacle.x = xyz[0] # Lidar_frame coordinates # -xyz[1]
                        groundtruth_obstacle.y = xyz[1]                           # -xyz[0]
                        groundtruth_obstacle.vel_lin = vel_lin
                        groundtruth_obstacle.vel_ang = vel_ang
                        groundtruth_obstacle.tl_br = [0,0,0,0] # 2D bbox (Image plane) top-left, bottom-right  xy coordinates  
                        groundtruth_obstacle.l = l # Lidar_frame coordinates
                        groundtruth_obstacle.w = w  
                        groundtruth_obstacle.o = heading   

                        groundtruth_obstacle.x_corners = [corners_3d[0,0], corners_3d[0,1], corners_3d[0,2], corners_3d[0,3]] #Array of x coordinates (upper left, upper right, lower left, lower right)
                        groundtruth_obstacle.y_corners = [corners_3d[1,0], corners_3d[1,1], corners_3d[1,2], corners_3d[1,3]]

                        groundtruth_obstacles_list.bev_detections_list.append(groundtruth_obstacle)

                        # Marker

                        groundtruth_obstacle_marker = marker_bb(location,quat_xyzw,dim,published_obj)
                        groundtruth_obstacles_marker_array.markers.append(groundtruth_obstacle_marker)
            
            if published_obj == 0:
                groundtruth_obstacle = BEV_detection()
                groundtruth_obstacles_list.bev_detections_list.append(groundtruth_obstacle)

                groundtruth_obstacle_marker = Marker()
                groundtruth_obstacle_marker.header.frame_id = "/map"
                groundtruth_obstacles_marker_array.markers.append(groundtruth_obstacle_marker)
        else:
            groundtruth_obstacle = BEV_detection()
            groundtruth_obstacles_list.bev_detections_list.append(groundtruth_obstacle)

            groundtruth_obstacle_marker = Marker()
            groundtruth_obstacle_marker.header.frame_id = "/map"
            groundtruth_obstacles_marker_array.markers.append(groundtruth_obstacle_marker)
        
        self.pub_groundtruth_obstacles.publish(groundtruth_obstacles_list) 
        self.pub_groundtruth_obstacles_marker.publish(groundtruth_obstacles_marker_array) 
    
if __name__ == '__main__':
    program = AB4COGT2SORT()
    program.listener()
