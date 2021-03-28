#!/usr/bin/python3 

import sys
import rospy
import dlib
import cv2
import numpy as np
import tf2_geometry_msgs
import tf2_ros
import math
#import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA 
from face_detector.msg import FaceDetected, Detected 
from enum import Enum 

# states 
class State(Enum): 
    STATIONARY = 1
    EXPLORE = 2
    GREET = 3 

class Face:
    def __init__(self, x, y, z):
        # coordinates of a detected face
        self.x = x
        self.y = y
        self.z = z 

        # Detection publisher
        self.detection_publisher = rospy.Publisher('detection', Detected, queue_size=10); 

    def publish(self, x, y, z, exists, index): 
        # function for publishing
        self.detection_publisher.publish(x, y, z, exists, index) 


def faceDetection(data, faces): 
    # checks if face was detected before and creates a marker if not
    detectedFace = Face(data.world_x, data.world_y, data.world_z) 

    exists = False
    index = 0

    if (len(faces) == 0):
        # first detected face (publish)
        print("First one")
        faces.append(detectedFace)
        detectedFace.publish(detectedFace.x, detectedFace.y, detectedFace.z, exists, index)
    else:
        print("More than one") 
        count = 0

        # checks if face already exists
        for face in faces:
            distanceM = math.sqrt((face.x - detectedFace.x)**2 + (face.y - detectedFace.y)**2 + (face.z - detectedFace.z)**2) 

            if (distanceM < 1):
                exists = True
                index = count
            count+=1  

        if (exists): 
            print("Already exist") 
            # if already exists update the coordinates
            avgX = (faces[index].x + detectedFace.x)/2 
            avgY = (faces[index].y + detectedFace.y)/2 
            avgZ = (faces[index].z + detectedFace.z)/2 

            faces[index].x = avgX
            faces[index].y = avgY
            faces[index].z = avgZ 

            # publish the changed coordinates
            detectedFace.publish(avgX, avgY, avgZ, exists, index)
           
        else:
            # publish the new coordinates 
            faces.append(detectedFace)
            detectedFace.publish(detectedFace.x, detectedFace.y, detectedFace.z, exists, index)

    #rospy.loginfo('worldX: %3.5f, worldY: %3.5f, worldZ: %3.5f', data.face_x, data.face_y, data.face_z) 

def greetPosition(pose):
    # position
    greetX = 0 
    greetY = 0
    greetZ = 0 

    # rotation of a robot 
    rotation = {'r1':0.000, 'r2':0.000, 'r3':0.000, 'r4':0.000} 


    



def workspace(data, args): 
    # list of faces
    faces = args 
    faceDetection(data, faces) 


def main(): 
    # list for storing faces
    faces = []

    rospy.init_node('tester', anonymous=True)
    rospy.Subscriber('face_detection', FaceDetected, workspace, (faces)) 
    rospy.spin()

if __name__ == '__main__':
    main()
