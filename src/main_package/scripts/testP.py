#!/usr/bin/python3

from exercise6.msg import Cylinder 
from geometry_msgs.msg import PointStamped 
import tf2_geometry_msgs
import tf2_ros
import rospy

def handle_add_two_ints(req): 

    print(req.point.x) 
    return req.point

def add_two_ints_server():
    rospy.init_node('add_two_ints_server')
    s = rospy.Subscriber('/cylinderDetection', PointStamped, handle_add_two_ints)
    print("Ready to sum numbers from array.")
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()