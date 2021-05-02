#!/usr/bin/python3 

import sys
import math
import rospy
from enum import Enum 
from time import sleep

import tf2_geometry_msgs
import tf2_ros

from geometry_msgs.msg import PointStamped, Vector3, Pose, Point, Quaternion
from face_detector.msg import FaceDetected, Detected, CylinderD 
from object_detection.msg import RingDetected, DetectedR, CylinderDetected
# from object_detection.msg import Cylinder 
from move_manager.mover import Mover
import numpy as np 

from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient

from sklearn import neural_network
import joblib
import pathlib



class State(Enum): 
    STATIONARY = 1
    EXPLORE = 2
    FACE_DETECTED = 3
    CYLINDER_DETECTED = 4
    RING_DETECTED = 5
    GREET_FACE = 6
    GREET_CYLINDER = 7  
    GREET_RING = 8
    FINISH = 9

class Cylinder: 
    def __init__(self, x, y, z, color, norm_x, norm_y): 
        self.x = x 
        self.y = y 
        self.z = z 

        self.color = color

        self.norm_x = norm_x 
        self.norm_y = norm_y 

        self.num_of_detections = 1 
        self.greeted = False

class Ring: 
    def __init__(self, x, y, z, color, norm_x, norm_y): 
        self.x = x 
        self.y = y 
        self.z = z 

        self.color = color

        self.norm_x = norm_x 
        self.norm_y = norm_y 

        self.num_of_detections = 1 
        self.greeted = False

class Face:
    def __init__(self, x, y, z, norm_x, norm_y):
        self.x = x
        self.y = y
        self.z = z 

        self.norm_x = norm_x
        self.norm_y = norm_y

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

        # All data that needs to be stored (cylinders)
        self.cylinder_detection_treshold = 1
        self.new_cylinder_detection_index = -1
        self.cylinders = [] 

        # All data that needs to be stored (cylinders)
        self.ring_detection_treshold = 1
        self.new_ring_detection_index = -1
        self.rings = [] 

        self.mover = Mover()
        self.mlpClf = joblib.load("./src/color_model/Models/MLPRGB.pkl")

        # All message passing nodes
        self.face_detection_subsriber = rospy.Subscriber('face_detection', FaceDetected, self.faceDetection)
        self.face_detection_marker_publisher = rospy.Publisher('detection', Detected, queue_size=10);  

        # All message passing nodes (cylinder)
        self.cylinder_detection_subsriber = rospy.Subscriber('/cylinderDetection', CylinderDetected, self.cylinderDetection)
        self.cylinder_detection_marker_publisher = rospy.Publisher('detectionC', CylinderD, queue_size=10); 

        # All message passing nodes (rings)
        self.ring_detection_subsriber = rospy.Subscriber('ring_detection', RingDetected, self.ringDetection)
        self.ring_detection_marker_publisher = rospy.Publisher('detectionR', DetectedR, queue_size=10);


    # Processes that need to be updated every iteration 
    def update(self):
        return


    # Act based on current state
    def execute(self):
        # print(State(self.state))

        if self.state == State.STATIONARY: 
            if (len(self.faces) == 5): 
                self.mover.stop_robot() 
                self.state = State.FINISH
                return

            # self.mover.follow_path()
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

            # pose and orientation of a robot
            pointR = Point(robotPose.position.x, robotPose.position.y, robotPose.position.z)
            orientationR = Quaternion(robotPose.orientation.x, robotPose.orientation.y, robotPose.orientation.z, robotPose.orientation.w)

            robotPoint = Pose(pointR, orientationR)

            # greet
            greetPoint, greetOrientation = self.greetFace(robotPose, facePoint)
            self.mover.move_to(greetPoint, greetOrientation)

            self.state = State.GREET_FACE
            return

        elif self.state == State.CYLINDER_DETECTED:            
            self.mover.stop_robot()
            robotPose = self.mover.get_pose()

            # pose of a detected face
            cylinderPoint = self.cylinders[self.new_cylinder_detection_index]

            # pose and orientation of a robot
            pointR = Point(robotPose.position.x, robotPose.position.y, robotPose.position.z)
            orientationR = Quaternion(robotPose.orientation.x, robotPose.orientation.y, robotPose.orientation.z, robotPose.orientation.w)

            robotPoint = Pose(pointR, orientationR)

            # greet
            greetPoint, greetOrientation = self.greetCylinder(robotPose, cylinderPoint)
            self.mover.move_to(greetPoint, greetOrientation)

            self.state = State.GREET_CYLINDER
            return 

        elif self.state == State.RING_DETECTED:
            # stop robot
            self.mover.stop_robot() 

            # get the pose of the robot 
            robotPose = self.mover.get_pose() 

            # pose of a detected ring 
            ringPoint = self.rings[self.new_ring_detection_index] 

            # pose and orientation of a robot
            pointR = Point(robotPose.position.x, robotPose.position.y, robotPose.position.z)
            orientationR = Quaternion(robotPose.orientation.x, robotPose.orientation.y, robotPose.orientation.z, robotPose.orientation.w)

            robotPoint = Pose(pointR, orientationR) 

            # greet
            greetPoint, greetOrientation = self.greetRing(robotPose, ringPoint)
            self.mover.move_to(greetPoint, greetOrientation)

            self.state = State.GREET_RING
            return

        elif self.state == State.GREET_FACE: 
            #soundhandle = SoundClient()
            #rospy.sleep(1)

            voice = 'voice_kal_diphone'
            volume = 2.0
            s = "Hello human, how are you today?"

            #soundhandle.say(s, voice, volume)
            #print("Greetings")

            if(not self.mover.traveling):
                # greet the face when the robot stops
                #soundhandle.say(s, voice, volume)
                print("Greetings")
                rospy.sleep(1)
                sleep(2)

                self.state = State.STATIONARY

            return 

        elif self.state == State.GREET_RING: 
            if(not self.mover.traveling):
                print("Greetings ring")
                rospy.sleep(1)
                sleep(2)

                self.state = State.STATIONARY

            return 
        
        elif self.state == State.GREET_CYLINDER: 
            if(not self.mover.traveling):
                print("Greetings cylinder")
                rospy.sleep(1)
                sleep(2)

                self.state = State.STATIONARY

            return 

        elif self.state == State.FINISH: 
            print("Dettected all faces") 
            sleep(10) 
            
            return



    #--------------------------CALLBACKS-----------------------------

    def faceDetection(self, data): 

        # Determine when to ignore this callback
        if (self.state == State.FACE_DETECTED) or (self.state == State.FINISH):
            return

        # Computue face normal
        robotPose = self.mover.get_pose()

        robotPoint_np = np.array([robotPose.position.x, robotPose.position.y])
        facePoint_np = np.array([data.world_x, data.world_y])
        face_normal = robotPoint_np - facePoint_np
        face_normal = face_normal / np.linalg.norm(face_normal)

        # Create face
        detectedFace = Face(data.world_x, data.world_y, data.world_z, face_normal[0], face_normal[1]) 

        # Process detection
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
                
                # Dot product to check if normals on the same side
                normalComparison = face.norm_x * detectedFace.norm_x + face.norm_y * detectedFace.norm_y
                # print("---------------------NORMAL_DOT_PRODUCT-----------")
                # print(normalComparison)

                # Face exists if correct distance away and its normal is aprox max 60 degrees different
                if ((distanceM < 1) and (normalComparison > 0.1)):
                    face.num_of_detections += 1
                    exists = True
                    index = count
                count+=1  


            if (exists): 
                print("Detected face already exists") 

                # moving average
                alpha = 0.15

                # if already exists update the coordinates 
                
                movAvgX = (self.faces[index].x * (1 - alpha)) + (detectedFace.x * alpha)
                movAvgY = (self.faces[index].y * (1 - alpha)) + (detectedFace.y * alpha)
                movAvgZ = (self.faces[index].z * (1 - alpha)) + (detectedFace.z * alpha)

                self.faces[index].x = movAvgX
                self.faces[index].y = movAvgY
                self.faces[index].z = movAvgZ 

                self.face_detection_marker_publisher.publish(movAvgX, movAvgY, movAvgZ, exists, index)

                # update normal
                org_normal = np.array([self.faces[index].norm_x, self.faces[index].norm_y])
                new_normal = np.array([detectedFace.norm_x, detectedFace.norm_y])
                updated_normal = org_normal + alpha * (new_normal - org_normal)
                updated_normal = updated_normal / np.linalg.norm(updated_normal)
                
                self.faces[index].norm_x = updated_normal[0]
                self.faces[index].norm_y = updated_normal[1]

                if ((self.state != State.GREET) and self.faces[index].num_of_detections >= self.face_detection_treshold) and (not self.faces[index].greeted):
                    print("Tershold cleared - face detection signal to robot") 
                    self.state = State.FACE_DETECTED
                    self.new_face_detection_index = index

            else:
                if (self.state == State.GREET_FACE): 
                    print("In greet state")
                else:
                    print("New face instance detected") 
                    self.face_detection_marker_publisher.publish(detectedFace.x, detectedFace.y, detectedFace.z, exists, index)

                    if(self.face_detection_treshold == 1):
                        self.state = State.FACE_DETECTED
                        self.faces.append(detectedFace)


    def cylinderDetection(self, data): 

        # Testing cylinder detection
        print("Cylinder detected")
        print("Color:", self.getStringLabel(self.mlpClf.predict([data.colorHistogram])))
        # print("Colors: Black - Blue - Green - Red - White - Yellow")
        # print(self.mlpClf.predict_proba([data.colorHistogram]))

        # Determine when to ignore this callback
        if (self.state == State.CYLINDER_DETECTED) or (self.state == State.FINISH):
            return

        color = self.getStringLabel(self.mlpClf.predict([data.colorHistogram]))
        if (color == "White") or (color == "Black"):
            print("False positive")
            return

        # TODO: if not sure of the color move around cylinder
        maxScore = max(self.mlpClf.predict_proba([data.colorHistogram]))
        if maxScore < 0.95:
            # TODO: move around the cylinder
            return # temporary solution

        # Compute cylinder normal 
        robotPose = self.mover.get_pose() 

        robotPoint_np = np.array([robotPose.position.x, robotPose.position.y])
        cylinderPoint_np = np.array([data.cylinder_x, data.cylinder_y]) 

        cylinder_normal = robotPoint_np - cylinderPoint_np
        cylinder_normal = cylinder_normal / np.linalg.norm(cylinder_normal) 

        # create a cylinder 
        detectedCylinder = Cylinder(data.cylinder_x, data.cylinder_y, data.cylinder_z, color, cylinder_normal[0], cylinder_normal[1]) 

        # Process detection
        exists = False
        index = 0 

        # first detected cylinder (publish)
        if (len(self.cylinders) == 0):
            print("New cylinder instance detected")
            self.cylinders.append(detectedCylinder)

            if(self.cylinder_detection_treshold == 1):
                self.state = State.CYLINDER_DETECTED 
                self.cylinder_detection_marker_publisher.publish(detectedCylinder.x, detectedCylinder.y, detectedCylinder.z, detectedCylinder.color, exists, index) # dodaj publisher za valje 

        else:
            count = 0

            # Checks if cylinder already exists
            for cylinder in self.cylinders:
                distanceM = math.sqrt((cylinder.x - detectedCylinder.x)**2 + (cylinder.y - detectedCylinder.y)**2 + (cylinder.z - detectedCylinder.z)**2) 
                
                # Dot product to check if normals on the same side
                normalComparison = cylinder.norm_x * detectedCylinder.norm_x + cylinder.norm_y * detectedCylinder.norm_y
                # print("---------------------NORMAL_DOT_PRODUCT-----------")
                # print(normalComparison)

                # Face exists if correct distance away and its normal is aprox max 60 degrees different
                if ((distanceM < 1) and (normalComparison > 0.1)):
                    cylinder.num_of_detections += 1
                    exists = True
                    index = count
                count+=1  

            if (exists): 
                # moving average
                alpha = 0.15

                # if already exists update the coordinates

                movAvgX = (self.cylinders[index].x * (1 - alpha)) + (detectedCylinder.x * alpha)
                movAvgY = (self.cylinders[index].y * (1 - alpha)) + (detectedCylinder.y * alpha)
                movAvgZ = (self.cylinders[index].z * (1 - alpha)) + (detectedCylinder.z * alpha)

                self.cylinders[index].x = movAvgX
                self.cylinders[index].y = movAvgY
                self.cylinders[index].z = movAvgZ 

                self.cylinder_detection_marker_publisher.publish(movAvgX, movAvgY, movAvgZ, detectedCylinder.color, exists, index)

                # update normal 

                org_normal = np.array([self.cylinders[index].norm_x, self.cylinders[index].norm_y])
                new_normal = np.array([detectedCylinder.norm_x, detectedCylinder.norm_y])
                updated_normal = org_normal + alpha * (new_normal - org_normal)
                updated_normal = updated_normal / np.linalg.norm(updated_normal)
                
                self.cylinders[index].norm_x = updated_normal[0]
                self.cylinders[index].norm_y = updated_normal[1]

                if ((self.state != State.GREET_CYLINDER) and self.cylinders[index].num_of_detections >= self.cylinder_detection_treshold) and (not self.cylinders[index].greeted):
                    print("Tershold cleared - cylinder detection signal to robot") 
                    self.state = State.CYLINDER_DETECTED
                    self.new_face_detection_index = index

            else:
                print("New cylinder instance detected") 
                self.cylinder_detection_marker_publisher.publish(detectedCylinder.x, detectedCylinder.y, detectedCylinder.z, detectedCylinder.color, exists, index) # popravi

                if(self.cylinder_detection_treshold == 1):
                    self.state = State.CYLINDER_DETECTED
                    self.cylinders.append(detectedCylinder)


    def ringDetection(self, data): 
        print("Ring detected")

        if (self.state == State.RING_DETECTED) or (self.state == State.FINISH):
            return

        # compute ring normal 
        robotPose = self.mover.get_pose() 

        robotPoint_np = np.array([robotPose.position.x, robotPose.position.y])
        ringPoint_np = np.array([data.ring_x, data.ring_y]) 

        ring_normal = robotPoint_np - ringPoint_np
        ring_normal = ring_normal / np.linalg.norm(ring_normal) 

        # create a ring
        detectedRing = Ring(data.ring_x, data.ring_y, data.ring_z, data.color, ring_normal[0], ring_normal[1]) 

        # Process detection
        exists = False
        index = 0 

        # first detected ring (publish)
        if (len(self.rings) == 0):
            print("New ring instance detected")
            self.rings.append(detectedRing)

            if(self.ring_detection_treshold == 1):
                self.state = State.RING_DETECTED 
                self.ring_detection_marker_publisher.publish(detectedRing.x, detectedRing.y, detectedRing.z, detectedRing.color, exists, index)
        
        else:
            count = 0

            # Checks if ring already exists
            for ring in self.rings:
                distanceM = math.sqrt((ring.x - detectedRing.x)**2 + (ring.y - detectedRing.y)**2 + (ring.z - detectedRing.z)**2) 
                
                # Dot product to check if normals on the same side
                normalComparison = ring.norm_x * detectedRing.norm_x + ring.norm_y * detectedRing.norm_y
                # print("---------------------NORMAL_DOT_PRODUCT-----------")
                # print(normalComparison)

                # Face exists if correct distance away and its normal is aprox max 60 degrees different
                if ((distanceM < 1) and (normalComparison > 0.1) and (ring.color == detectedRing.color)):
                    ring.num_of_detections += 1
                    exists = True
                    index = count
                count+=1  

            if (exists): 
                print("Detected ring already exists")

                # moving average
                alpha = 0.15

                # if already exists update the coordinates
                movAvgX = (self.rings[index].x * (1 - alpha)) + (detectedRing.x * alpha)
                movAvgY = (self.rings[index].y * (1 - alpha)) + (detectedRing.y * alpha)
                movAvgZ = (self.rings[index].z * (1 - alpha)) + (detectedRing.z * alpha)

                self.rings[index].x = movAvgX
                self.rings[index].y = movAvgY
                self.rings[index].z = movAvgZ 

                self.ring_detection_marker_publisher.publish(movAvgX, movAvgY, movAvgZ, detectedRing.color, exists, index)

                # Update normal 
                org_normal = np.array([self.rings[index].norm_x, self.rings[index].norm_y])
                new_normal = np.array([detectedRing.norm_x, detectedRing.norm_y])
                updated_normal = org_normal + alpha * (new_normal - org_normal)
                updated_normal = updated_normal / np.linalg.norm(updated_normal)
                
                self.rings[index].norm_x = updated_normal[0]
                self.rings[index].norm_y = updated_normal[1] 

                if ((self.state != State.GREET_RING) and self.rings[index].num_of_detections >= self.ring_detection_treshold) and (not self.rings[index].greeted):
                    print("Tershold cleared - ring detection signal to robot") 
                    self.state = State.RING_DETECTED
                    self.new_ring_detection_index = index

            else:
                if (self.state == State.GREET_RING): 
                    print("In greet state")
                else:
                    print("New ring instance detected") 
                    self.ring_detection_marker_publisher.publish(detectedRing.x, detectedRing.y, detectedRing.z, detectedRing.color, exists, index) # popravi

                    if(self.ring_detection_treshold == 1):
                        self.state = State.RING_DETECTED
                        self.rings.append(detectedRing)



    #---------------------------------GREET-----------------------------------

    def greetFace(self, robotPose, facePose):
        #self.state = State.GREET
        facePose.greeted = True

        # self.mover.move_to(face.x,  face.y)
        self.mover.is_following_path = False

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
        yaw_z = fi - Ryaw_z

        # back to quaternions 
        nX, nY, nZ, nW = self.euler_to_quaternion(Rroll_x, Rpitch_y, yaw_z) 

        quaternion = Quaternion(nX, nY, nZ, nW)

        return point, quaternion 


    def greetCylinder(self, robotPose, cylinderPose):
        cylinderPose.greeted = True 

        self.mover.is_following_path = False

        # position
        greetX = 0 
        greetY = 0
        greetZ = 0 

        # distance between robot and detected ring
        distance =  math.sqrt((cylinderPose.x - robotPose.position.x)**2 + (cylinderPose.y - robotPose.position.y)**2) 

        # travel distance
        if (distance > 0.5):
            travelD = distance - 0.5
        else:
            travelD = distance 

        fi = math.atan2(cylinderPose.y - robotPose.position.y, cylinderPose.x - robotPose.position.x) 

        # greet position
        greetX = robotPose.position.x + travelD * math.cos(fi)
        greetY = robotPose.position.y + travelD * math.sin(fi) 

        point = Point(greetX, greetY, greetZ)

        # current orientation of a robot
        Rroll_x, Rpitch_y, Ryaw_z = self.euler_from_quaternion(robotPose.orientation.x, robotPose.orientation.y, robotPose.orientation.z, robotPose.orientation.w) 

        # new yaw 
        yaw_z = fi - Ryaw_z

        # back to quaternions 
        nX, nY, nZ, nW = self.euler_to_quaternion(Rroll_x, Rpitch_y, yaw_z) 

        quaternion = Quaternion(nX, nY, nZ, nW)

        return point, quaternion 


    def greetRing(self, robotPose, ringPose):
        ringPose.greeted = True 

        self.mover.is_following_path = False

        # position
        greetX = 0 
        greetY = 0
        greetZ = 0 

        # distance between robot and detected ring
        distance =  math.sqrt((ringPose.x - robotPose.position.x)**2 + (ringPose.y - robotPose.position.y)**2) 

        # travel distance
        if (distance > 0.5):
            travelD = distance - 0.5
        else:
            travelD = distance 

        fi = math.atan2(ringPose.y - robotPose.position.y, ringPose.x - robotPose.position.x) 

        # greet position
        greetX = robotPose.position.x + travelD * math.cos(fi)
        greetY = robotPose.position.y + travelD * math.sin(fi) 

        point = Point(greetX, greetY, greetZ)

        # current orientation of a robot
        Rroll_x, Rpitch_y, Ryaw_z = self.euler_from_quaternion(robotPose.orientation.x, robotPose.orientation.y, robotPose.orientation.z, robotPose.orientation.w) 

        # new yaw 
        yaw_z = fi - Ryaw_z

        # back to quaternions 
        nX, nY, nZ, nW = self.euler_to_quaternion(Rroll_x, Rpitch_y, yaw_z) 

        quaternion = Quaternion(nX, nY, nZ, nW)

        return point, quaternion 



    #----------------------------------HELPERS---------------------------------

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

    def getStringLabel(self, numLabel):
        if numLabel == 0:
            return "Black"
        elif numLabel == 1:
            return "Blue"
        elif numLabel == 2:
            return "Green"
        elif numLabel == 3:
            return "Red"
        elif numLabel == 4:
            return "White"
        elif numLabel == 5:
            return "Yellow"  



def main():
    mainNode = MainNode()

    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        mainNode.update()
        mainNode.execute()
        rate.sleep()


if __name__ == '__main__':
    main()