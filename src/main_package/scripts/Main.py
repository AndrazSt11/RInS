#!/usr/bin/python3 

from pickle import FALSE
import sys
import math
import rospy
import numpy as np 
from enum import Enum, IntEnum 
from time import sleep
from urllib.request import urlopen

from sklearn import neural_network
import sklearn.metrics as metrics
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
from data_viewer.msg import Data 
from data_viewer.msg import Detected_number


class State(Enum): 
    STATIONARY = 1
    EXPLORE = 2
    APPROACH = 3
    BUSY = 4
    ADJUSTING = 5
    FINISH = 6

class TaskType(Enum):
    FACE_PROCESS = 1
    SOCIAL_DIST_WARN = 2

class FaceProcessState(IntEnum):
    DETECTED = 1
    FACE_CONVERSATOIN = 2
    CLINIC_CONVERSATOIN = 3
    PICK_UP_VACCINE = 4
    DELIVER_VACCINE = 5
    VACCINE_SEARCH = 6
    CLINIC_SEARCH = 7
    FINISHED = 8

class ObjectType(Enum):
    FACE = 1
    CYLINDER = 2
    RING = 3

class ObjProperty(Enum):
    UNKNOWN = 1
    FALSE = 2
    TRUE = 3

class Color(Enum):
    UNKNOWN = 1
    BLACK = 2
    BLUE = 3
    GREEN = 4
    RED = 5
    WHITE = 6
    YELLOW = 7


class Object:
    def __init__(self, id, type, x, y, z, norm_x, norm_y):
        self.id = id
        self.type = type

        self.x = x 
        self.y = y 
        self.z = z 

        self.norm_x = norm_x 
        self.norm_y = norm_y 

        self.num_of_detections = 1 
        self.update_greet_index = 1

class FaceObj(Object):
    def __init__(self, id, x, y, z, norm_x, norm_y, wears_mask):
        Object.__init__(self, id, ObjectType.FACE, x, y, z, norm_x, norm_y)
        self.wears_mask = wears_mask
        self.is_vaccinated = ObjProperty.UNKNOWN
        self.physical_exercise = -1
        self.doctor = Color.UNKNOWN
        self.suitable_vaccine = Color.UNKNOWN 
        self.age = -1 

class RingObj(Object):
    def __init__(self, id, x, y, z, norm_x, norm_y, color):
        Object.__init__(self, id, ObjectType.RING, x, y, z, norm_x, norm_y)
        self.color = color

class CylinderObj(Object):
    def __init__(self, id, x, y, z, norm_x, norm_y, color):
        Object.__init__(self, id, ObjectType.CYLINDER, x, y, z, norm_x, norm_y)
        self.color = color
        self.classificier = ObjProperty.UNKNOWN


class Task: 
    def __init__(self, id, type):
        self.id = id
        self.type = type

        self.finished = False
        self.aborted = False
        self.ignore_for_loops = 0

class FaceProcess(Task):
    def __init__(self, id, person_id):
        Task.__init__(self, id, TaskType.FACE_PROCESS)
        self.state = FaceProcessState.DETECTED
        self.person_id = person_id
        self.cylinder_id = -1
        self.ring_id = -1

class SocialDistWarn(Task):
    def __init__(self, id, person_id):
        Task.__init__(self, id, TaskType.SOCIAL_DIST_WARN)
        self.person_id = person_id



class MainNode:
    def __init__(self):
        self.state = State.STATIONARY
        rospy.init_node('main_node', anonymous=True)

        # Pending tasks
        self.taskId = 0
        self.tasks = []
        self.current_task = False

        # How many times we detect an object before we add it to objects
        self.min_detections = {
            ObjectType.RING: 1,
            ObjectType.CYLINDER: 1,
            ObjectType.FACE: 1,
        }

        # Data about all detected objects
        self.objects = {
            ObjectType.RING: [],
            ObjectType.CYLINDER: [],
            ObjectType.FACE: [],
        }

        self.marker_publishers = {
            ObjectType.RING: rospy.Publisher('Ring_detected_markes', RingDetectedMarker, queue_size=10),
            ObjectType.CYLINDER: rospy.Publisher('Cylinder_detection_markers', CylinderDetectedMarker, queue_size=10),
            ObjectType.FACE: rospy.Publisher('Face_detected_markers', FaceDetectedMarker, queue_size=10),
        } 
        
        # age of current person 
        self.current_age = -1
        
        # data of current person 
        self.current_data = ""  
        
        # data of current cylinder 
        self.current_cy = ""
        

        self.mover = Mover()
        self.move_less_than_to_get_suck = .1
        self.rotate_less_than_to_get_suck = .085
        self.pose_history_size = 4
        self.pose_history = []

        self.mlpClf = joblib.load("./src/color_model/Models/MLPRGB.pkl") 

        # publisher for robot arm 
        self.robot_arm = rospy.Publisher("/arm_command", String, queue_size=10)

        # All message passing nodes
        self.face_detection_subsriber = rospy.Subscriber('face_detection', FaceDetected, self.on_face_detection)
        self.cylinder_detection_subsriber = rospy.Subscriber('/cylinderDetection', CylinderDetected, self.on_cylinder_detection)
        self.ring_detection_subsriber = rospy.Subscriber('ring_detection', RingDetected, self.on_ring_detection) 
        self.qr_detection_subsriber = rospy.Subscriber('qr_detection', Data, self.on_qr_detected) 
        self.num_detection_subsriber = rospy.Subscriber('num_detection', Detected_number, self.on_num_detected)
        
        # soundhandle
        self.volume = 2.0
        self.voice = 'voice_kal_diphone'
        self.soundhandle = SoundClient()


    def before_execute(self):
        # print(self.state)
        # print(self.tasks)
        # print(self.objects)
        if self.is_stuck():
            rospy.logwarn("We're stuck - ADJUSTING position and retrying task")
            self.state = State.ADJUSTING
            self.mover.stop_robot()
            self.mover.move_to_next_point(True)
            return

        if self.current_task:
            if self.is_task_active(self.current_task):
                return
        
        self.current_task = self.get_available_task()
        if self.current_task:
            rospy.loginfo(f"GETTING NEW TASK: type={self.current_task.type}")


    # Act based on current state
    def execute(self):

        if self.state == State.ADJUSTING:
            if not self.mover.traveling:
                self.state = State.STATIONARY


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
                self.did_approach = False
                self.mover.move_to(point, quat, force_reach=False)
            else:
                # self.abort_task(self.current_task)
                if self.current_task in self.tasks:
                    self.tasks.remove(self.current_task)
                    self.current_task = False

                self.state = State.STATIONARY


        if self.state == State.BUSY:
            if not self.mover.traveling:
                if self.current_task:

                    if not self.did_approach:
                        self.did_approach = True
                        self.mover.move_forward(0.4)
                        rospy.sleep(1.7)

                    if self.current_task.type == TaskType.FACE_PROCESS:
                        self.current_task.state += 1

                        if self.current_task.state == FaceProcessState.FACE_CONVERSATOIN:
                            self.on_face_conversation()

                        if self.current_task.state == FaceProcessState.CLINIC_CONVERSATOIN: 
                            print("Checking if it goes to clinic conversation")
                            self.on_clinic_conversation()

                        if self.current_task.state == FaceProcessState.PICK_UP_VACCINE:
                            self.on_vaccine_pick_up()

                        if self.current_task.state == FaceProcessState.DELIVER_VACCINE:
                            self.on_deliver_vaccine()

                        self.state = State.STATIONARY


                    elif self.current_task.type == TaskType.SOCIAL_DIST_WARN:
                        self.on_social_dist_warn_reached()

                else:
                    rospy.logwarn("No task in BUSY state?")

                self.state = State.STATIONARY
            
            elif self.current_task: # NOTE: avoiding bool error
                self.update_greet_position(self.current_task)


        if self.state == State.EXPLORE:
            self.mover.follow_path()


        if self.state == State.FINISH: 
            rospy.loginfo("Robot finished all tasks")


    def update_greet_position(self, task):
        object = self.get_object(task)
        if not object: 
            return 

        if object.num_of_detections > object.update_greet_index * 2:
            object.update_greet_index += 1

            success, point, quat = self.get_task_point(task)
            if success:
                print("Updated greet position")
                self.mover.stop_robot()
                self.mover.move_to(point, quat, force_reach=False)


    def is_stuck(self):
        # check for traveling
        if not self.mover.traveling:
            self.pose_history.clear()
            return False

        new_pose = self.mover.get_pose()
        
        # fill up the array before checking
        if len(self.pose_history) < self.pose_history_size:
            self.pose_history.append(new_pose)
            return False

        # check position and rotation delta
        avg_delta_pos = 0
        avg_delta_rot = 0
        for old_pose in self.pose_history:
            avg_delta_pos += abs(old_pose.position.x - new_pose.position.x) + abs(old_pose.position.y - new_pose.position.y)
            avg_delta_rot += abs(old_pose.orientation.x - new_pose.orientation.x) + abs(old_pose.orientation.y - new_pose.orientation.y) + abs(old_pose.orientation.z - new_pose.orientation.z) + abs(old_pose.orientation.w - new_pose.orientation.w)

        avg_delta_pos /= self.pose_history_size
        avg_delta_rot /= self.pose_history_size


        if avg_delta_pos < self.move_less_than_to_get_suck and avg_delta_rot < self.rotate_less_than_to_get_suck:
            print(avg_delta_pos, avg_delta_rot)
            self.pose_history.clear()
            return True

        else:
            self.pose_history.pop(0)
            self.pose_history.append(new_pose)

        return False



    #------------------------ TASK HANDLING ---------------------------
    def get_object(self, task):
        # Get correct object
        if task.type == TaskType.SOCIAL_DIST_WARN:
            return self.objects[ObjectType.FACE][task.person_id]

        elif task.type == TaskType.FACE_PROCESS:
            if task.state == FaceProcessState.DETECTED:
                return self.objects[ObjectType.FACE][task.person_id]

            elif task.state == FaceProcessState.FACE_CONVERSATOIN:
                return self.objects[ObjectType.CYLINDER][task.cylinder_id]

            elif task.state == FaceProcessState.CLINIC_CONVERSATOIN: 
                return self.objects[ObjectType.RING][task.ring_id]

            elif task.state == FaceProcessState.PICK_UP_VACCINE:
                return self.objects[ObjectType.FACE][task.person_id]

        return False

    def get_task_point(self, task):
        object = self.get_object(task)
        if not object: 
            return False, None, None      

        robot_pose = self.mover.get_pose()
        initial_point = Point(object.x, object.y, 0.0)
        
        point = self.get_valid_point_near(initial_point, object)
        if not point:
            rospy.logwarn(f"Couldn't find a valid point. Aborting task: id={object.id}")
            return False, None, None

        # Orient to object 
        if (object.type == ObjectType.FACE): 
            # turn a litle to the right of the face to detect QR code
            print("Objekt je obraz")
            fi = math.atan2(object.y - point.y, (object.x + 0.2) - point.x)
            quat = transformations.quaternion_from_euler(0, 0, fi)
        else:
            print("Objekt NI obraz")
            fi = math.atan2(object.y - point.y, object.x - point.x)
            quat = transformations.quaternion_from_euler(0, 0, fi)

        return True, point, Quaternion(0, 0, quat[2], quat[3])

    def is_task_active(self, task):
        if task.finished or task.aborted:
            self.tasks.remove(task)
            return False

        if task.type == TaskType.FACE_PROCESS:
            if task.state == FaceProcessState.CLINIC_SEARCH or task.state == FaceProcessState.VACCINE_SEARCH:
                return False

        return True

    def get_available_task(self):
        for task in self.tasks:
            if task.ignore_for_loops > 0:
                task.ignore_for_loops -= 1
                continue

            if task.type == TaskType.SOCIAL_DIST_WARN:
                if self.is_task_active(task): return task

            if task.type == TaskType.FACE_PROCESS:
                if int(task.state) < 6: return task # Task can be treated (we have necessary information)
                            
        return False

    def update_tasks(self, object):
        for task in self.tasks:
            if task.type == TaskType.FACE_PROCESS:
                person = self.objects[ObjectType.FACE][task.person_id]
                if task.state == FaceProcessState.CLINIC_SEARCH and object.type == ObjectType.CYLINDER and person.doctor == object.color:
                    task.cylinder_id = object.id 
                    task.state = FaceProcessState.FACE_CONVERSATOIN
                
                if task.state == FaceProcessState.VACCINE_SEARCH and object.type == ObjectType.RING and person.suitable_vaccine == object.color:
                    task.ring_id = object.id
                    task.state = FaceProcessState.CLINIC_CONVERSATOIN


    #----------------------- OBJECT HANDLING --------------------------
    def publish_object_marker(self, object, exists):
        publisher = self.marker_publishers[object.type]
        if publisher:
            if object.type == ObjectType.FACE:
                publisher.publish(object.x, object.y, object.z, exists, object.id)
            else:
                publisher.publish(object.x, object.y, object.z, self.get_color_string(object.color), exists, object.id)


    def object_exists(self, object):
        # Checks if already exists
        for old_object in self.objects[object.type]:    
            # Check normal and distance
            normal_compare = old_object.norm_x * object.norm_x + old_object.norm_y * object.norm_y
            dist = math.sqrt((old_object.x - object.x)**2 + (old_object.y - object.y)**2 + (old_object.z - object.z)**2) 

            # Compare normals only for faces
            if object.type == ObjectType.FACE and normal_compare < 0.06: 
                continue

            # Compare colors only for rings and cylinders
            if (object.type == ObjectType.RING or object.type == ObjectType.CYLINDER) and object.color != old_object.color: 
                continue

            # Object exists if correct distance 
            
            # test -------> maby needs fix
            if object.type == ObjectType.FACE:
                if dist < 0.5: # TODO: test different values
                    return old_object
            else:
                if dist < 1.4 :
                    return old_object


    def update_object(self, old, new):
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
        if old.type == ObjectType.FACE:
            old.wears_mask = old.wears_mask or new.wears_mask

        # update marker
        self.publish_object_marker(old, exists=True)

            
    def add_object(self, type, x, y, z, color, wears_mask):
        # Computue normal
        robot_pose = self.mover.get_pose()
        rp_np = np.array([robot_pose.position.x, robot_pose.position.y])

        tp_np = np.array([x, y])
        normal = rp_np - tp_np
        normal = normal / np.linalg.norm(normal)

        # Create task
        if type == ObjectType.FACE:
            new_object = FaceObj(len(self.objects[type]), x, y, z, normal[0], normal[1], wears_mask)
        elif type == ObjectType.CYLINDER:
            new_object = CylinderObj(len(self.objects[type]), x, y, z, normal[0], normal[1], color)
        elif type == ObjectType.RING:
            new_object = RingObj(len(self.objects[type]), x, y, z, normal[0], normal[1], color)

        old_object = self.object_exists(new_object)

        # Check if we should update the task or create a new one
        if old_object:
            self.update_object(old_object, new_object)
            new_object = old_object

        else:
            self.objects[type].append(new_object)
            
            if new_object.type == ObjectType.FACE:
                self.tasks.append(FaceProcess(self.taskId, new_object.id))
                self.taskId += 1
            else:
                self.update_tasks(new_object)

            # update marker
            self.publish_object_marker(new_object, exists=False)


        # NOTE: DEPRECATED - probably detection limit will be set to 1 for all object types
        # TODO: add pending object list, which have not reached detection limit
        # # lets check if we need to queue it
        # if (new_task.num_of_detections >= self.min_detections[type]) and (not new_task in self.tasks):
        #     print("New task added:", new_task.type, new_task.color)
        #     self.tasks.append(new_task)
    

    #-------------------------- CALLBACKS -----------------------------
    def on_face_detection(self, data):
        self.add_object(ObjectType.FACE, data.world_x, data.world_y, data.world_z, False, self.get_obj_property_enum(data.wears_mask))


    def on_cylinder_detection(self, data): 
        color = self.get_color_label(self.mlpClf.predict([data.colorHistogram]))
        if (color == "White") or (color == "Black"):
            rospy.logwarn(f"Cylinder detection: false-positive color={color}")
            return

        self.add_object(ObjectType.CYLINDER, data.cylinder_x, data.cylinder_y, data.cylinder_z, self.get_color_enum(color), False)


    def on_ring_detection(self, data):
        if data.color == "White":
            rospy.logwarn(f"Ring detection: false-positive color={data.color}")
            return

        self.add_object(ObjectType.RING, data.ring_x, data.ring_y, data.ring_z, self.get_color_enum(data.color), False)


    def on_face_conversation(self):
        volume = 2.0
        voice = 'voice_kal_diphone'
        person = self.objects[ObjectType.FACE][self.current_task.person_id]
        
        # Greet
        rospy.loginfo(f"Greeting face - {self.current_task.person_id}")
        s = "Hello human!"
        rospy.sleep(2)
        self.soundhandle.say(s, voice, volume)

        # Check if it wears mask
        rospy.loginfo(f"Wears mask? - {person.wears_mask}")
        if(person.wears_mask == ObjProperty.FALSE):
            s = "Please put on your mask!"
            self.soundhandle.say(s, voice, volume)
            print(s)
            rospy.sleep(2)

        # Check social distancing
        social_distancing_list = self.is_social_distancing(person)
        is_social_distancing = len(social_distancing_list) == 0
        rospy.loginfo(f"Follows social distancing? - {is_social_distancing}")
        
        if(not is_social_distancing):
            # Add task to warn other person as well
            for person_id in social_distancing_list:
                self.tasks.append(SocialDistWarn(self.taskId, person_id))
                self.taskId += 1

            s = "Please keep social distance!" 
            self.soundhandle.say(s, voice, volume)
            rospy.sleep(2)

        #----------------------TESTING---------------------------------
        # person.is_vaccinated = ObjProperty,FALSE
        # person.physical_exercise = 36.734693877551024
        # person.doctor = self.get_color_enum("Red")
        # # person.suitable_vaccine = self.get_color_enum(usr_data[5])
        # person.age = 48.484848484848484
        #-------------------------------------------------------------- 
        
        # trying to detect qr code sequence
        if (self.current_data == ""):
            rospy.logwarn("Couldn't read QR code, readjusting")
            for forward in [0.5, -0.65]:
                self.mover.move_forward(forward)
                rospy.sleep(1.7)

                for deg in [50, -50, -50, 50]:
                    self.mover.rotate_deg(deg)
                    
                    rospy.sleep(1.25)
                    if (self.current_data != ""):
                        break

                if (self.current_data != ""):
                    break

            if (self.current_data == ""):
                print("couldn't detect qr code, ignoring task for now")
                self.mover.rotate_deg(0)
                rospy.sleep(.25)
                self.mover.move_forward(-0.25)
                self.current_task.state -= 1

                self.current_task.ignore_for_loops = 15
                self.current_task = False

                rospy.sleep(1)
                self.mover.move_forward(0)
                self.state = State.STATIONARY

                return False

                

        # Get person info 
        usr_data = self.current_data.split(",") 
        
        # print(usr_data) 
        
        print("R: Have you already been vaccinated?")

        person.is_vaccinated = self.get_obj_property_enum(usr_data[2])
        
        if (person.is_vaccinated != ObjProperty.TRUE):
            print("P: Yes")
        else:
            print("P: No")
            
        print("R: Who is your personal doctor?")
        
        person.physical_exercise = int(usr_data[4][1:])
        
        person.doctor = self.get_color_enum(usr_data[3][1:]) 
        
        print("P: My personal doctor is: ", self.get_color_string(person.doctor))
        # person.suitable_vaccine = self.get_color_enum(usr_data[5]) 
        
        # check if age was detected 
        if self.current_age > -1: 
            person.age = self.current_age 
        else:
            rospy.loginfo(f"Couldn't detect age with digit extractor. Reading it from QR")
            person.age = int(usr_data[1][1:])# self.current_age 
            
        # setting age to default value
        self.current_age = -1
        
        # print("Oseba: ", person.is_vaccinated, person.physical_exercise, person.doctor, person.age) 
        
        # setting current_data on default value 
        self.current_data = ""
        
        if (person.is_vaccinated != ObjProperty.TRUE):
            print("This person was already vaccinated. Finish current task")
            self.current_task.state = FaceProcessState.FINISHED
            self.current_task.finished = True
        else:
            # Check for suitable clinic
            clinic_list = self.objects[ObjectType.CYLINDER]
            for i in range(0, len(clinic_list)): 
                # print("Doktor: ", clinic_list[i].color) 
                # print("Osebni zdravnik trenutne osebe: ", person.doctor)
                if clinic_list[i].color == person.doctor: 
                    print("Going to suitable clinic")
                    self.current_task.cylinder_id = i
                    break 

            # No suitable clinc found
            if self.current_task.cylinder_id == -1: 
                print("No suitable clinic. Look more")
                self.current_task.state = FaceProcessState.CLINIC_SEARCH 


    def on_clinic_conversation(self):
        print("Testing if clinic_conversation is active")
        person = self.objects[ObjectType.FACE][self.current_task.person_id]
        cylinder = self.objects[ObjectType.CYLINDER][self.current_task.cylinder_id]
        
        # Train classificier if necessary
        if cylinder.classificier == ObjProperty.UNKNOWN: 
            
            # trying to detect qr code sequence
            if (self.current_cy == ""):
                rospy.logwarn("Couldn't read QR code, readjusting")
                for forward in [0.5, -0.65]:
                    self.mover.move_forward(forward)
                    rospy.sleep(1.7)

                    for deg in [50, -50, -50, 50]:
                        self.mover.rotate_deg(deg)
                        
                        rospy.sleep(1.25)
                        if (self.current_cy != ""):
                            break

                    if (self.current_cy != ""):
                        break

                if (self.current_cy == ""):
                    print("couldn't detect qr code, ignoring task for some time")
                    self.mover.rotate_deg(0)
                    rospy.sleep(.25)
                    self.mover.move_forward(-0.25)
                    rospy.sleep(1)
                    self.mover.move_forward(0)

                    self.state = State.STATIONARY
                    self.current_task.state = FaceProcessState.CLINIC_SEARCH

                    self.current_task.ignore_for_loops = 15
                    self.current_task = False

                    return False
                

            link = self.current_cy
            # link = "https://box.vicos.si/rins/17.txt" TESTING
            cylinder.classificier = self.train_classificier(link)

        # Predict suitable vaccine
        predicted_vaccine = cylinder.classificier.predict([[person.age, person.physical_exercise]])[0]
        person.suitable_vaccine = self.get_vaccine_color(predicted_vaccine)
        print("R: Person needs vaccine:", person.suitable_vaccine)
        
        # put current_cy back to "" 
        self.current_cy = ""

        # Check for suitable vaccine
        vaccine_list = self.objects[ObjectType.RING]
        for i in range(0, len(vaccine_list)):
            if vaccine_list[i].color == person.suitable_vaccine:
                print("R: Going to suitable vaccine")
                self.current_task.ring_id = i
                break 

        # No suitable vaccine found
        if self.current_task.ring_id == -1: 
            print("R: No suitable vaccine. Look more")
            self.current_task.state = FaceProcessState.VACCINE_SEARCH
        

    def on_vaccine_pick_up(self): 
        rospy.loginfo(f"Picking up the vaccine for the person - {self.current_task.person_id}")
        
        self.robot_arm.publish("ring")
        rospy.sleep(1) 
        self.robot_arm.publish("retract") 
        
        self.current_task.person_id


    def on_deliver_vaccine(self): 
        self.mover.move_forward(0.35)
        rospy.sleep(0.5)
        rospy.loginfo(f"Vaccinating person - {self.current_task.person_id}")
        self.robot_arm.publish("extend")
        rospy.sleep(1) 
        self.robot_arm.publish("retract") 
        
        self.objects[ObjectType.FACE][self.current_task.person_id].is_vaccinated = ObjProperty.TRUE
        self.current_task.state = FaceProcessState.FINISHED
        self.current_task.finished = True


    def on_social_dist_warn_reached(self):
        # Warn
        s = "Please keep social distance!" 
        rospy.sleep(2)
        self.soundhandle.say(s, self.voice, self.volume) 

        self.current_task.finished = True
        

    def on_num_detected(self, data):
        age = str(data.x) + str(data.y)
        self.current_age = int(age) 
        #print("Age of person is: " + self.current_age)
        

    def on_qr_detected(self, data): 
        # print(str(data.data)[2:7]) 
        
        # check if detected QR is on the cylinder or face 
        if (str(data.data)[2:7] == "https"): 
            self.current_cy = str(data.data)[2:-1]
        else:
            self.current_data = str(data.data)


#---------------------------- CLASSIFICIER ---------------------------
    def train_classificier(self, link):
        # Get data
        dataSetX, dataSetY = self.get_classificier_data(link)

        # Train 
        print("Training classificier")
        mlpClf = neural_network.MLPClassifier(hidden_layer_sizes=(32, 32), activation="relu", solver="adam", alpha=0.0001,
                                              max_iter=30, n_iter_no_change=20, verbose=False)
        mlpClf.fit(dataSetX, dataSetY)

        print("Accuracy:", metrics.accuracy_score(dataSetY, mlpClf.predict(dataSetX), normalize=True))
        print("Training finished")
        
        return mlpClf

    def get_classificier_data(self, link):
        dataSetX = []
        dataSetY = []
        
        f = urlopen(link)
        myfile = f.read()
        lines = myfile.splitlines()

        for line in lines:
            line = line.decode("utf-8").split(',')
            dataSetX.append([float(line[0]), float(line[1])])
            dataSetY.append(line[2])

        return dataSetX, dataSetY


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

    def get_color_enum(self, color):
        if color == "Black":
            return Color.BLACK
        elif color == "Blue":
            return Color.BLUE
        elif color == "Green":
            return Color.GREEN
        elif color == "Red":
            return Color.RED
        elif color == "White":
            return Color.WHITE
        elif color == "Yellow":
            return Color.YELLOW

    def get_color_string(self, color_enum):
        if color_enum == Color.BLACK:
            return "Black"
        elif color_enum == Color.BLUE:
            return "Blue"
        elif color_enum == Color.GREEN:
            return "Green"
        elif color_enum == Color.RED:
            return "Red"
        elif color_enum == Color.WHITE:
            return "White"
        elif color_enum == Color.YELLOW:
            return "Yellow"

    def get_vaccine_color(self, vaccine_name):
        if vaccine_name == "BlacknikV":
            return Color.BLACK
        elif vaccine_name == "StellaBluera":
            return Color.BLUE
        elif vaccine_name == "Greenzer":
            return Color.GREEN
        elif vaccine_name == "Rederna":
            return Color.RED

        return Color.UNKNOWN
            
    def get_obj_property_enum(self, prop):
        if prop:
            return ObjProperty.TRUE
        
        return ObjProperty.FALSE

    def get_valid_point_near(self, point, object):
        # Try with different offsets
        for offset in [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8]:
            for x in [0, offset/2, offset, -offset/2, -offset]:
                for y in [0, -offset/2, offset/2, offset, -offset]:
                    temp = Point( point.x + x, point.y + y, 0)
                    if self.mover.is_valid(temp): 
                        if object.type == ObjectType.FACE: 
                            norm = np.array([object.norm_x, object.norm_y]) 
                            
                            current = np.array([temp.x, temp.y]) 
                            plane = np.array([object.x, object.y]) 
                            
                            p1 = np.array([current[0] - plane[0], current[1] - plane[1]])
                            
                            result = np.dot(norm, p1) 
                            
                            if (result > 0):
                                return temp
                        else:
                            return temp
        
        return False

    def is_social_distancing(self, current_person):
        social_dist_id = []
        for person in self.objects[ObjectType.FACE]:
            distance = math.sqrt((current_person.x - person.x)**2 + (current_person.y - person.y)**2) 
            normal_compare = current_person.norm_x * person.norm_x + current_person.norm_y * person.norm_y

            if((distance < 1.0) and (normal_compare > 0.06) and (current_person.id != person.id)):
                social_dist_id.append(person.id)

        return social_dist_id


def main():
    mainNode = MainNode()

    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        mainNode.before_execute()
        mainNode.execute()
        rate.sleep()

if __name__ == '__main__':
    main()
