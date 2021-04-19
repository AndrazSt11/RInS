#!/usr/bin/python3

import rospy
import actionlib

from enum import Enum
from geometry_msgs.msg import Point, Quaternion, Pose
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from move_manager.map_processer import get_map_points
from os import getcwd

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
        self.path = Path(get_map_points(f'{getcwd()}/src/simulation/maps/map.pgm'))
        
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
