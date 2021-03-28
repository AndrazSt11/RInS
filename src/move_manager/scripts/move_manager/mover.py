#!/usr/bin/python3

import rospy
import actionlib

from enum import Enum
from geometry_msgs.msg import Point
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction

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
        return self.points[self.nextPoint][0], self.points[self.nextPoint][1]


    def on_point_reached(self):
        if len(self.points) > self.nextPoint+1:
            self.nextPoint += 1
        else:
            self.nextPoint = 0



class Mover():
    
    def __init__(self):
        
        # init node
        rospy.init_node("mover_client")

        self.traveling = False
        self.is_following_path = False
        self.path = Path([
            (-1.471733808517456, 1.7823206186294556),
            (2.3780295848846436, 1.6542164087295532),
            (3.7790722846984863, -0.06843602657318115),
            (-0.052309419959783554, -1.1175440549850464),
            (-1.3600460290908813, 0.1608671247959137),
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
        # print(result)

        self.path.on_point_reached()
        self.traveling = False

        if self.is_following_path:
            self.move_to_next_point()

    # feedback callback
    def on_goal_feedback(self, feedback):
        #rospy.loginfo(feedback)
        return


    # for easier access
    def move_to_next_point(self):
        x, y = self.path.get_next_point()
        self.move_to(x,y)


    # used to move robot to point
    def move_to(self, x, y):
        if self.traveling:
            rospy.logwarn("robot is already trying to reach path. To cancel path call stop_robot() first.")
            return

        self.traveling = True
        
        # lets move
        goal_msg = MoveBaseGoal()
        goal_msg.target_pose.header.frame_id = "map"
        goal_msg.target_pose.pose.position.x = x
        goal_msg.target_pose.pose.position.y = y
        goal_msg.target_pose.pose.position.z = 0.0
        goal_msg.target_pose.pose.orientation.w = 1.0
        goal_msg.target_pose.header.stamp = rospy.get_rostime()

        rospy.loginfo(f"Moving to (x: {x}, y: {y})")
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

        self.traveling = False
        self.is_following_path = False

        self.move_base_client.cancel_goal()
