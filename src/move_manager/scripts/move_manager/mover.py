#!/usr/bin/python3

import rospy
import actionlib

from enum import Enum
from geometry_msgs.msg import Point, Quaternion, Pose
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction

# TODO get current state
# TODO add orientation to move_to

class State(Enum):
    PENDING = 0
    ACTIVE = 1
    PREEMPTED = 2
    SUCCEEDED = 3
    ABORTED = 4
    REJECTED = 5
    PREEMPTING = 6
    RECALLING = 7
    RECALLED = 8
    LOST = 9


class Path():
    def __init__(self, points):
        self.nextPoint = 0
        self.points = points


    def get_next_point(self):
        print(f"next point: {self.nextPoint}")
        return self.points[self.nextPoint][0], self.points[self.nextPoint][1]


    def on_point_reached(self):
        if len(self.points) > self.nextPoint+1:
            self.nextPoint += 1
        else:
            self.nextPoint = 0
    
    def revert_point(self):
        if self.nextPoint == 0:
            self.nextPoint = len(self.points)-1
        else:
            self.nextPoint -= 1



class Mover():
    
    def __init__(self):
        
        self.traveling = False
        self.is_following_path = False
        
        self.force_reach = True

        self.current_pose = Pose(Point(0,0,0), Quaternion(0,0,0,1))
        self.goal_position = Point(0,0,0)
        self.path = Path([
            # Original
            # (-1.471733808517456, 1.7823206186294556),
            # (2.3780295848846436, 1.6542164087295532),
            # (3.7790722846984863, -0.06843602657318115),
            # (-0.052309419959783554, -1.1175440549850464),
            # (-1.3600460290908813, 0.1608671247959137),

            # Test 1 --> cela mapa vendar lahko spusti kaksen obraz - hitrejse
            # (-1.0605767965316772,  0.20456738770008087),
            # (-0.7818923592567444,  1.3890641927719116),
            # (-1.5359022617340088, 2.0269229412078857),
            # (-0.1495995968580246,  2.6942381858825684),
            # (2.065833806991577, 2.8600914478302),
            # (2.297785520553589, 1.47145676612854),
            # (0.9072411060333252, 1.6520534753799438),
            # (1.0559861660003662,  0.42801767587661743),
            # (3.394010305404663,  0.1708206683397293),
            # (2.3318569660186768,  -0.659281313419342),
            # (0.943772554397583,  -0.7427576184272766),
            # (0.12640441954135895,  -1.0241304636001587),
            # (-0.022990774363279343,  0.05159976705908775)

            # Test 2 --> bolj na gosto, da slucajno ne spusti kaksen obraz
            (-0.37434712052345276, 0.3294188380241394),
            (-1.2818517684936523, 0.12442848086357117),
            (-0.37424689531326294, 1.0152627229690552),
            (-1.6752092838287354, 2.1714353561401367),
            (-1.3786342144012451, 1.6557153463363647),
            (-0.3946381211280823, 2.8080992698669434),
            (0.6404500603675842, 2.4048047065734863),
            (1.901002287864685, 3.0965423583984375),
            (1.9940999746322632, 2.0470991134643555),
            (2.3456523418426514, 0.8655444979667664),
            (0.9201184511184692, 1.8283897638320923),
            (1.0682435035705566, 0.4560337960720062),
            (3.0115089416503906, 0.5774980783462524),
            (2.2840499877929688, -0.9804591536521912),
            (1.3340673446655273, -0.6397660970687866),
            (0.30510377883911133, -1.1538519859313965)
        ])
        
        # get move base client
        self.move_base_client = self.get_base_client()
    

    def get_base_client(self):
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server")

        client.wait_for_server()
        rospy.loginfo("move_base action server found!")
        
        return client


    # callback that fires when goal is reached
    def on_goal_reached(self, state, result):
        rospy.loginfo(State(state))

        self.traveling = False

        if self.is_following_path:
            if State(state) is State.SUCCEEDED:
                self.path.on_point_reached()

            self.move_to_next_point()


    # feedback callback
    def on_goal_feedback(self, feedback):
        self.current_pose = feedback.base_position.pose

        if not self.force_reach:
            return

        if abs(self.goal_position.x - self.current_pose.position.x) < 0.2:
            if abs(self.goal_position.y - self.current_pose.position.y) < 0.2:
                self.move_base_client.cancel_goal()
                self.on_goal_reached(3, None)
                print("FORCE REAHCED")


    # for easier access
    def move_to_next_point(self):
        if not self.traveling:
            x, y = self.path.get_next_point()
            self.move_to(Point(x,y,0.0), Quaternion(0,0,0,1))


    # returns current pose of robot
    def get_pose(self):
        return self.current_pose


    # used to move robot to point
    def move_to(self, point, quat):
        if self.traveling:
            rospy.logwarn("robot is already trying to reach path. To cancel path call stop_robot() first.")
            return

        self.traveling = True
        
        # lets move
        goal_msg = MoveBaseGoal()
        goal_msg.target_pose.header.frame_id = "map"
        goal_msg.target_pose.pose.position.x = point.x
        goal_msg.target_pose.pose.position.y = point.y
        goal_msg.target_pose.pose.position.z = 0.0
        goal_msg.target_pose.pose.orientation.x = 0
        goal_msg.target_pose.pose.orientation.y = 0
        goal_msg.target_pose.pose.orientation.z = 0
        goal_msg.target_pose.pose.orientation.w = 1
        goal_msg.target_pose.header.stamp = rospy.get_rostime()

        self.goal_position = goal_msg.target_pose.pose.position

        rospy.loginfo(f"Moving to (x: {point.x}, y: {point.y})")
        self.move_base_client.send_goal(goal_msg, self.on_goal_reached, None, self.on_goal_feedback)


    # used to tell robot to follow his own path
    def follow_path(self):
        if self.is_following_path:
            rospy.logwarn("Already following path")
            return

        self.is_following_path = True
        self.move_to_next_point()


    # stop following next goal (path or any other point)
    def stop_robot(self):

        #if self.is_following_path:
        #    self.path.revert_point()
        
        self.traveling = False
        self.is_following_path = False

        self.move_base_client.cancel_goal()
