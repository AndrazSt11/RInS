#!/usr/bin/python3 

from pickle import FALSE
import sys
import math
import rospy
import numpy as np 
from enum import Enum 
from time import sleep

from sklearn import neural_network
import joblib
import pathlib

from tf import transformations
import tf2_geometry_msgs
import tf2_ros

from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient

from move_manager.mover import Mover

from geometry_msgs.msg import PointStamped, Vector3, Pose, Point, Quaternion
from markers.msg import FaceDetectedMarker, CylinderDetectedMarker, RingDetectedMarker
from face_detector.msg import FaceDetected
from object_detection.msg import RingDetected, CylinderDetected
from std_msgs.msg import String


class State(Enum): 
    STATIONARY = 1
    EXPLORE = 2
    APPROACH = 3
    BUSY = 5
    FINISH = 6

class TaskType(Enum):
    FACE = 1
    CYLINDER = 2
    RING = 3


class Task: 
    def __init__(self, type, id, x, y, z, norm_x, norm_y, color=False, wears_mask=False):
        self.id = id
        self.type = type

        self.x = x 
        self.y = y 
        self.z = z 

        self.color = color
        self.wears_mask = wears_mask

        self.norm_x = norm_x 
        self.norm_y = norm_y 

        self.num_of_detections = 1 
        self.update_greet_index = 1

        self.finished = False
        self.aborted = False


class MainNode:
    def __init__(self):
        self.state = State.STATIONARY
        rospy.init_node('main_node', anonymous=True)

        # Task queue (points to visit)
        self.tasks = []
        self.current_task = False

        # How many times we detect an object before we queue task
        self.min_detections = {
            TaskType.RING: 1,
            TaskType.CYLINDER: 1,
            TaskType.FACE: 1,
        }

        # Data about previous tasks (current and pending are also included)
        self.history = {
            TaskType.RING: [],
            TaskType.CYLINDER: [],
            TaskType.FACE: [],
        }

        self.marker_publishers = {
            TaskType.RING: rospy.Publisher('Ring_detected_markes', RingDetectedMarker, queue_size=10),
            TaskType.CYLINDER: rospy.Publisher('Cylinder_detection_markers', CylinderDetectedMarker, queue_size=10),
            TaskType.FACE: rospy.Publisher('Face_detected_markers', FaceDetectedMarker, queue_size=10),
        }

        self.mover = Mover()
        self.mlpClf = joblib.load("./src/color_model/Models/MLPRGB.pkl") 

        # publisher for robot arm 
        self.robot_arm = rospy.Publisher("/arm_command", String, queue_size=10)

        # All message passing nodes
        self.face_detection_subsriber = rospy.Subscriber('face_detection', FaceDetected, self.on_face_detection)
        self.cylinder_detection_subsriber = rospy.Subscriber('/cylinderDetection', CylinderDetected, self.on_cylinder_detection)
        self.ring_detection_subsriber = rospy.Subscriber('ring_detection', RingDetected, self.on_ring_detection) 
        
        # soundhandle
        self.soundhandle = SoundClient()


    # this runs before every execute
    def before_execute(self):
        if self.current_task:
            if not self.current_task.finished and not self.current_task.aborted:
                return
        
        self.current_task = self.get_next_task()
        if self.current_task:
            rospy.loginfo(f"GETTING NEW TASK: type={self.current_task.type} color={self.current_task.color}")


    # Act based on current state
    def execute(self):

        if self.state == State.STATIONARY or self.state == State.EXPLORE:
            if self.current_task:
                self.state = State.APPROACH
            else:
                self.state = State.EXPLORE


        if self.state == State.APPROACH:
            self.mover.stop_robot()
            success, point, quat = self.get_task_point(self.current_task)

            if success:
                self.state = State.BUSY
                self.mover.move_to(point, quat, force_reach=False)
            else:
                self.abort_task(self.current_task)
                self.state = State.STATIONARY


        if self.state == State.BUSY:
            if not self.mover.traveling:
                if self.current_task:
                    if self.current_task.type == TaskType.RING:
                        self.on_ring_reached()

                    elif self.current_task.type == TaskType.CYLINDER:
                        self.on_cylinder_reached()

                    elif self.current_task.type == TaskType.FACE:
                        self.on_face_reached()

                    self.remove_finished_task()
                else:
                    rospy.logwarn("No task in BUSY state?")

                self.state = State.STATIONARY
            
            elif self.current_task: # NOTE: avoiding bool error
                if self.current_task.num_of_detections > self.current_task.update_greet_index * 2:
                    self.current_task.update_greet_index += 1

                    success, point, quat = self.get_task_point(self.current_task)
                    if success:
                        print("Updated greet position")
                        self.mover.stop_robot()
                        self.mover.move_to(point, quat, force_reach=False)


        if self.state == State.EXPLORE:
            self.mover.follow_path()
        

        if self.state == State.FINISH: 
            rospy.loginfo("Robot finished all tasks")

        #rospy.loginfo(f"MAIN STATE: {self.state}")

    #--------------------- TASK HANDLERS ------------------------


    def get_task_point(self, task):
        robot_pose = self.mover.get_pose()

        # travel distance
        travel_distance = math.sqrt((task.x - robot_pose.position.x)**2 + (task.y - robot_pose.position.y)**2)


        fi1 = math.atan2(task.y - robot_pose.position.y, task.x - robot_pose.position.x)
        initial_point = Point(robot_pose.position.x + travel_distance * math.cos(fi1), robot_pose.position.y + travel_distance * math.sin(fi1), 0.0)

        point = self.get_valid_point_near(initial_point)
        if not point:
            rospy.logwarn(f"Couldn't find a valid point. Aborting task: id={task.id}")
            return False, None, None

        # Orient to object
        fi = math.atan2(task.y - point.y, task.x - point.x)
        quat = transformations.quaternion_from_euler(0, 0, fi)

        return True, point, Quaternion(0, 0, quat[2], quat[3])


    def get_next_task(self):
        if len(self.tasks) > 0:
            return self.tasks[0]
        
        return False

    def remove_finished_task(self):
        if len(self.tasks) > 0:
            self.tasks.pop(0)


    def abort_task(self, task):
        rospy.logwarn(f"Aborting task: id={task.id} type={task.type}")
        task.aborted = True

        # Remove from queue
        self.tasks.pop(0)
        # self.remove_from_history(task)

        self.state = State.STATIONARY


    def publish_task_marker(self, task, exists):
        publisher = self.marker_publishers[task.type]
        if publisher:
            if task.type == TaskType.FACE:
                publisher.publish(task.x, task.y, task.z, exists, task.id)
            else:
                publisher.publish(task.x, task.y, task.z, task.color, exists, task.id)

    
    def task_exists_history(self, task):
        # Checks if already exists
        for old_task in self.history[task.type]:    
            # Check normal and distance
            normal_compare = old_task.norm_x * task.norm_x + old_task.norm_y * task.norm_y
            d = math.sqrt((old_task.x - task.x)**2 + (old_task.y - task.y)**2 + (old_task.z - task.z)**2) 

            # Compare normals only for faces
            if task.type == TaskType.FACE and normal_compare < 0.06: 
                continue

            # Task exists if correct distance away and same color
            if d < 1.4 and task.color == old_task.color:
                return old_task

    def update_task(self, old, new):
        # update detecion num
        old.num_of_detections += 1

        # moving average
        alpha = 0.15

        # update coordinates
        movAvgX = (old.x * (1 - alpha)) + (new.x * alpha)
        movAvgY = (old.y * (1 - alpha)) + (new.y * alpha)
        movAvgZ = (old.z * (1 - alpha)) + (new.z * alpha)

        old.x = movAvgX
        old.y = movAvgY
        old.z = movAvgZ 

        # update normal
        old_normal = np.array([old.norm_x, old.norm_y])
        new_normal = np.array([new.norm_x, new.norm_y])
        updated_normal = old_normal + alpha * (new_normal - old_normal)
        updated_normal = updated_normal / np.linalg.norm(updated_normal)
        
        old.norm_x = updated_normal[0]
        old.norm_y = updated_normal[1]

        # update mask detection
        old.wears_mask = old.wears_mask or new.wears_mask

        # upadte marker
        self.publish_task_marker(old, exists=True)


    def add_task(self, type, x, y, z, color, wears_mask):
        # Computue face normal
        robot_pose = self.mover.get_pose()
        rp_np = np.array([robot_pose.position.x, robot_pose.position.y])

        tp_np = np.array([x, y])
        normal = rp_np - tp_np
        normal = normal / np.linalg.norm(normal)

        new_task = Task(type, len(self.history[type]), x, y, z, normal[0], normal[1], color, wears_mask)
        old_task = self.task_exists_history(new_task)

        # Check if we should update the task or create a new one
        if old_task:
            # we don't want to repeat the same task
            if old_task.finished:
                return

            self.update_task(old_task, new_task)

            new_task = old_task

        else:
            # add to history
            self.history[type].append(new_task)

            # upadte marker
            self.publish_task_marker(new_task, exists=False)

        # lets check if we need to queue it
        if (new_task.num_of_detections >= self.min_detections[type]) and (not new_task in self.tasks):
            print("New task added:", new_task.type, new_task.color)
            self.tasks.append(new_task)


    #-------------------------- CALLBACKS -----------------------------
    # FACES
    def on_face_detection(self, data):
        self.add_task(TaskType.FACE, data.world_x, data.world_y, data.world_z, False, data.wears_mask)

    def on_face_reached(self):
        volume = 2.0
        voice = 'voice_kal_diphone'

        # TODO:
        # 1) Greet
        # 2) If it does not wear mask -> warn
        # 3) Check it's social distance
        # 4) Get info (by talking or QR code)
        # 5) Start task of bringing him medicine
        

        # Greet
        rospy.loginfo(f"Greeting face - {self.current_task.id}")
        s = "Hello human!"
        self.soundhandle.say(s, voice, volume)

        # Check if it wears mask
        rospy.loginfo(f"Wears mask? - {self.current_task.wears_mask}")
        if(self.current_task.wears_mask == False):
            s = "Please put on your mask!"
            self.soundhandle.say(s, voice, volume)

        # Check social distancing
        social_distancing_list = self.is_social_distancing(self.current_task)
        print(social_distancing_list)
        is_social_distancing = len(social_distancing_list) == 0
        
        rospy.loginfo(f"Follows social distancing? - {is_social_distancing}")
        if(not is_social_distancing):
            s = "Please keep social distance!"
            self.soundhandle.say(s, voice, volume)

        # TODO: add task to go warn other faces


        self.current_task.finished = True
        rospy.sleep(1)


    # CYLINDERS
    def on_cylinder_detection(self, data): 
        color = self.get_color_label(self.mlpClf.predict([data.colorHistogram]))
        if (color == "White") or (color == "Black"):
            rospy.logwarn(f"Cylinder detection: false-positive color={color}")
            return

        self.add_task(TaskType.CYLINDER, data.cylinder_x, data.cylinder_y, data.cylinder_z, color, False)
    
    def on_cylinder_reached(self): 
    
        soundhandle = SoundClient() 
        voice = 'voice_kal_diphone'
        volume = 5.0 
        s = "Color of the cylinder is: " + self.current_task.color
        
        rospy.sleep(1)
        rospy.loginfo(f"Greeting cylinder - {self.current_task.id}, color: {self.current_task.color}") 
        soundhandle.say(s, voice, volume)
        self.robot_arm.publish("extend")
        rospy.sleep(1) 
        self.robot_arm.publish("retract")
        self.current_task.finished = True
        rospy.sleep(1)


    # RINGS
    def on_ring_detection(self, data):
        if data.color == "White":
            rospy.logwarn(f"Ring detection: false-positive color={data.color}")
            return

        self.add_task(TaskType.RING, data.ring_x, data.ring_y, data.ring_z, data.color, False)

    def on_ring_reached(self):
        soundhandle = SoundClient()
        voice = 'voice_kal_diphone'
        volume = 5.0 
        s = "Color of the ring is: " + self.current_task.color
    
        rospy.loginfo(f"Greeting ring - id: {self.current_task.id}, color: {self.current_task.color}")
        rospy.sleep(1)
        soundhandle.say(s, voice, volume)
        self.current_task.finished = True
        rospy.sleep(1)
    

    #---------------------------- HELPERS ---------------------------
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


    # def get_quaternions(self, fi): 
    #     # q1 = math.cos(fi/2)
    #     # q2 = 0
    #     # q3 = 0 
    #     # q4 = math.sin(fi/2)

    #     q1 = 0
    #     q2 = 0
    #     q3 = 0 
    #     q4 = 1
        
    #     return q1, q2, q3, q4


    def get_color_label(self, num):
        if num == 0:
            return "Black"
        elif num == 1:
            return "Blue"
        elif num == 2:
            return "Green"
        elif num == 3:
            return "Red"
        elif num == 4:
            return "White"
        elif num == 5:
            return "Yellow"
            


    def get_valid_point_near(self, point):
        # Try with different offsets
        for offset in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
            for x in [0, -offset, offset]:
                for y in [0, -offset, offset]:
                    temp = Point( point.x + x, point.y + y, 0)
                    if self.mover.is_valid(temp):
                        return temp
        
        return False

    # TODO: take normals into account
    def is_social_distancing(self, current_task):
        social_dist_id = []
        for face_task in self.history[TaskType.FACE]:
            distance = math.sqrt((current_task.x - face_task.x)**2 + (current_task.y - face_task.y)**2) 
            if((distance < 1.0) and (current_task.id != face_task.id)):
                social_dist_id.append(face_task.id)

        return social_dist_id;


def main():
    mainNode = MainNode()

    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        mainNode.before_execute()
        mainNode.execute()
        rate.sleep()


if __name__ == '__main__':
    main()
