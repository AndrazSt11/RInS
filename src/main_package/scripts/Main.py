#!/usr/bin/python3 

import sys
import math
import rospy
from enum import Enum 
from time import sleep
from math import acos, sin, cos

import tf2_geometry_msgs
import tf2_ros

from geometry_msgs.msg import PointStamped, Vector3, Pose, Point, Quaternion
from face_detector.msg import FaceDetected, Detected 
from move_manager.mover import Mover
import numpy as np 

from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient

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
            robotPose = self.mover.get_pose()

            # pose of a detected face
            facePoint = self.faces[self.new_face_detection_index]

            # greet
            #greetPoint, greetOrientation = self.greetFace(robotPose, facePoint)
            #self.mover.move_to(greetPoint, greetOrientation)
            
            point, quat = self.on_face_detected(robotPose, facePoint)
            self.mover.move_to(point, quat)
            self.state = State.GREET
            return

        elif self.state == State.GREET:
            if self.mover.traveling:
                return

            soundhandle = SoundClient()
            rospy.sleep(1)

            voice = 'voice_kal_diphone'
            volume = 2.0
            s = "Hello"

            # greet the face
            soundhandle.say(s, voice, volume)
            print("Greetings")
            rospy.sleep(1)
            sleep(2)
                
            self.state = State.STATIONARY
            return

    def euler_from_quaternion(self, x, y, z, w): 
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1) 

        t2 = +2.0 * (w * y - z * x) 
        t2 = +1.0 if t2 > +1.0 else t2
        tw = -1 if t2 < -1.0 else t2 
        pitch_y = math.asin(t2) 

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4) 

        return roll_x, pitch_y, yaw_z 

    def euler_to_quaternion(self, roll, pitch, yaw):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return qx, qy, qz, qw


    #----------------------actions--------------------------
    def on_face_detected(self, robot_pose, face_pose):
        face_pose.greeted = True
        robot_pos = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z])
        face_pos = np.array([face_pose.x, face_pose.y, face_pose.z])

        forward = face_pos - robot_pos
        forward_norm = forward / np.linalg.norm(forward)
        
        destination = (face_pos - forward_norm * 0.5) if np.linalg.norm(forward) > 0.5 else robot_pos

        quat = quaternion_look_at(forward_norm)
        return Point(destination[0], destination[1], destination[2]), quat

    def greetFace(self, robotPose, facePose):
        facePose.greeted = True

        # TODO compute how to turn and move to greet a face
        # position
        greetX = 0 
        greetY = 0
        greetZ = 0 

        # distance between robot and detected face
        distance =  math.sqrt((facePose.x - robotPose.position.x)**2 + (facePose.y - robotPose.position.y)**2) 

        # travel distance
        if (distance > 0.5):
            travelD = distance - 0.5 
        else:
            travelD = distance

        fi = math.atan2(facePose.y - robotPose.position.y, facePose.x - robotPose.position.x)

        # greet position
        greetX = robotPose.position.x + travelD * math.cos(fi)
        greetY = robotPose.position.y + travelD * math.sin(fi) 

        point = Point(greetX, greetY, greetZ)

        # current orientation of a robot
        Rroll_x, Rpitch_y, Ryaw_z = self.euler_from_quaternion(robotPose.orientation.x, robotPose.orientation.y, robotPose.orientation.z, robotPose.orientation.w) 

        # new yaw 
        yaw_z = Ryaw_z - fi 

        # back to quaternions 
        nX, nY, nZ, nW = self.euler_to_quaternion(Rroll_x, Rpitch_y, yaw_z) 

        quaternion = Quaternion(nX, nY, nZ, nW)

        return point, quaternion 



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


def quaternion_create_from_axis_angle(axis, angle):
        half_angle = angle * 0.5
        s = sin(half_angle)

        return Quaternion(
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
            cos(half_angle)
        )


def quaternion_look_at(forward_norm):
    dot = np.dot(np.array([1,0,0]), forward_norm)

    if abs(dot-(-1.0)) < 0.000001:
        return Quaternion(0,0,1,3.14159265)
    
    if abs(dot-1.0) < 0.000001:
        return Quaternion(0,0,0,1)

    rot_angle = acos(dot)
    rot_axis = np.cross(np.array([1,0,0]), forward_norm)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    return quaternion_create_from_axis_angle(rot_axis, rot_angle)
    

def main():
    mainNode = MainNode()
    
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        mainNode.update()
        mainNode.execute()
        rate.sleep()


if __name__ == '__main__':
    main()
