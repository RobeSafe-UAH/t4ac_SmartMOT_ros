# ROS imports

import rospy
from visualization_msgs.msg import Marker

def marker_bb(location,quat_xyzw,dim,published_obj):
    """
    If corners = True, visualize the 3D corners instead of a solid cube
    """

    box_marker = Marker()
    box_marker.header.stamp = rospy.Time.now()
    box_marker.header.frame_id = "/map"
    box_marker.action = Marker.ADD
    box_marker.id = published_obj
    box_marker.lifetime = rospy.Duration.from_sec(0.2)

    box_marker.type = Marker.CUBE
    box_marker.pose.position.x = location[0]
    box_marker.pose.position.y = location[1]
    box_marker.pose.position.z = location[2]+0.8
    box_marker.pose.orientation.x = quat_xyzw.x
    box_marker.pose.orientation.y = quat_xyzw.y 
    box_marker.pose.orientation.z = quat_xyzw.z
    box_marker.pose.orientation.w = quat_xyzw.w 
    box_marker.scale.x = dim[0]
    box_marker.scale.y = dim[1]
    box_marker.scale.z = dim[2]
    box_marker.color.a = 1.0

    return box_marker