#!/usr/bin/python3 

import sys
import math
import rospy
from enum import Enum 
from time import sleep

import tf2_geometry_msgs
import tf2_ros

from geometry_msgs.msg import PointStamped, Vector3, Pose
from face_detector.msg import FaceDetected, Detected 
from move_manager.mover import Mover

class State(Enum): 
    STATIONARY = 1
    EXPLORE = 2
    FACE_DETECTED = 3
    GREET = 4

class Face:
    def __init__(self, x, y, z):
        # coordinates of a detected face
        self.x = x
        self.y = y
        self.z = z 

        self.num_of_detections = 1
        self.greeted = False


class MainNode:
    def __init__(self):
        self.state = State.STATIONARY
        rospy.init_node('main_node', anonymous=True)

        # All data that needs to be stored
        self.face_detection_treshold = 1
        self.new_face_detection_index = -1
        self.faces = []

        # All message passing nodes
        self.face_detection_subsriber = rospy.Subscriber('face_detection', FaceDetected, self.faceDetection)
        self.face_detection_marker_publisher = rospy.Publisher('detection', Detected, queue_size=10);  

        self.mover = Mover()


    # Processes that need to be updated every iteration 
    def update(self):
        return

    # Act based on current state
    def execute(self):
        rospy.loginfo(State(self.state))

        if self.state == State.STATIONARY:
            self.mover.follow_path()
            self.state = State.EXPLORE
            return

        elif self.state == State.EXPLORE:
            # Check if robot had some error

            return
            
        elif self.state == State.FACE_DETECTED:            
            self.mover.stop_robot()
            self.greetFace(self.faces[self.new_face_detection_index])
            return

        elif self.state == State.GREET:
            print("Greeting")
            sleep(10)

            if(not self.mover.traveling):
                self.state = State.STATIONARY

            return

    #----------------------actions--------------------------
    def greetFace(self, face):
        self.state = State.GREET
        face.greeted = True


        # self.mover.move_to(face.x,  face.y)
        self.mover.is_following_path = False

        # TODO compute how to turn and move to greet a face
        #     # position
        #     greetX = 0 
        #     greetY = 0
        #     greetZ = 0 

        #     # rotation of a robot 
        #     rotation = {'r1':0.000, 'r2':0.000, 'r3':0.000, 'r4':0.000} 



    #----------------call-back-functions--------------------
    
    # Checks if face was detected before and creates a marker if not
    def faceDetection(self, data): 

        # Determine when to ignore this callback
        if (self.state == State.FACE_DETECTED) or (self.state == State.GREET):
            return


        detectedFace = Face(data.world_x, data.world_y, data.world_z) 

        exists = False
        index = 0

        # first detected face (publish)
        if (len(self.faces) == 0):
            print("New face instance detected")
            self.faces.append(detectedFace)

            if(self.face_detection_treshold == 1):
                self.state = State.FACE_DETECTED
                self.face_detection_marker_publisher.publish(detectedFace.x, detectedFace.y, detectedFace.z, exists, index)
       
        else:
            count = 0

            # Checks if face already exists
            for face in self.faces:
                distanceM = math.sqrt((face.x - detectedFace.x)**2 + (face.y - detectedFace.y)**2 + (face.z - detectedFace.z)**2) 

                if (distanceM < 1):
                    face.num_of_detections += 1
                    exists = True
                    index = count
                count+=1  

            
            if (exists): 
                print("Detected face already exists") 

                # if already exists update the coordinates
                avgX = (self.faces[index].x + detectedFace.x)/2 
                avgY = (self.faces[index].y + detectedFace.y)/2 
                avgZ = (self.faces[index].z + detectedFace.z)/2 

                self.faces[index].x = avgX
                self.faces[index].y = avgY
                self.faces[index].z = avgZ 

                self.face_detection_marker_publisher.publish(avgX, avgY, avgZ, exists, index)

                if (self.faces[index].num_of_detections >= self.face_detection_treshold) and (not self.faces[index].greeted):
                    print("Tershold cleared - face detection signal to robot") 
                    self.state = State.FACE_DETECTED
                    self.new_face_detection_index = index

            else:
                print("New face instance detected") 
                self.face_detection_marker_publisher.publish(detectedFace.x, detectedFace.y, detectedFace.z, exists, index)

                if(self.face_detection_treshold == 1):
                    self.state = State.FACE_DETECTED
                    self.faces.append(detectedFace)

        #rospy.loginfo('worldX: %3.5f, worldY: %3.5f, worldZ: %3.5f', data.face_x, data.face_y, data.face_z) 


def main():
    mainNode = MainNode()
    
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        mainNode.update()
        mainNode.execute()
        rate.sleep()


if __name__ == '__main__':
    main()
