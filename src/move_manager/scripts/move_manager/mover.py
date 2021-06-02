#!/usr/bin/python3

import rospy
import actionlib

from enum import Enum
from geometry_msgs.msg import Point, Quaternion, Pose, Twist, Vector3

from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from move_manager.map_processer import get_map_points, is_point_valid
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


    def get_next_point(self, force=False):
        if force:
            self.nextPoint += 1
        print(f"PATH POINT ID: {self.nextPoint}")
        return self.points[self.nextPoint][0], self.points[self.nextPoint][1]


    def on_point_reached(self):
        if len(self.points) > self.nextPoint+1:
            self.nextPoint += 1
        else:
            self.points.reverse()
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

        points, image_data = get_map_points(f'{getcwd()}/src/simulation/maps/map.pgm')
        self.path = Path(points)
        self.image_data = image_data
        
        # twist move client
        self.move_twist_publisher = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=10)
       
        # get move base client
        self.move_base_client = self.get_base_client()


    def get_base_client(self):
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server")

        client.wait_for_server()
        rospy.loginfo("move_base action server found!")
        
        return client

    def is_valid(self, point):
        return is_point_valid(point.x, point.y, self.image_data)

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
        # print(feedback)

        self.current_pose = feedback.base_position.pose

        if not self.force_reach:
            return

        if abs(self.goal_position.x - self.current_pose.position.x) < 0.2:
            if abs(self.goal_position.y - self.current_pose.position.y) < 0.2:
                self.move_base_client.cancel_goal()
                self.on_goal_reached(3, None)


    # for easier access
    def move_to_next_point(self, force=False):
        if not self.traveling:
            x, y = self.path.get_next_point(force)
            moveToPoint = Point(x, y, 0.0)

            if not self.is_valid(moveToPoint):
                moveToPoint = self.get_valid_point_near(moveToPoint)

            self.move_to(moveToPoint)


    # returns current pose of robot
    def get_pose(self):
        return self.current_pose


    # used to move robot to point
    def move_to(self, point, quat=Quaternion(0,0,0,1), force_reach=True):
        if self.traveling:
            rospy.logwarn("robot is already trying to reach path. To cancel path call stop_robot() first.")
            return

        self.traveling = True
        self.force_reach = force_reach

        # lets move
        goal_msg = MoveBaseGoal()
        goal_msg.target_pose.header.frame_id = "map"
        goal_msg.target_pose.pose.position.x = point.x
        goal_msg.target_pose.pose.position.y = point.y
        goal_msg.target_pose.pose.position.z = 0.0
        goal_msg.target_pose.pose.orientation.x = quat.x
        goal_msg.target_pose.pose.orientation.y = quat.y
        goal_msg.target_pose.pose.orientation.z = quat.z
        goal_msg.target_pose.pose.orientation.w = quat.w
        goal_msg.target_pose.header.stamp = rospy.get_rostime()

        self.goal_position = goal_msg.target_pose.pose.position

        rospy.loginfo(f"NEXT GOAL: x={point.x:.3f}, y={point.y:.3f}, force={self.force_reach}")

        self.move_base_client.send_goal(goal_msg, self.on_goal_reached, None, self.on_goal_feedback) 


    # used to tell robot to follow his own path
    def follow_path(self):
        if self.is_following_path:
            return

        self.is_following_path = True
        self.move_to_next_point()
        

    def get_valid_point_near(self, point):
        # Try with different offsets
        for offset in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
            for x in [0, -offset, offset]:
                for y in [0, -offset, offset]:
                    temp = Point( point.x + x, point.y + y, 0)
                    if self.is_valid(temp):
                        return temp
        
        return False


    # stop following next goal (path or any other point)
    def stop_robot(self):

        #if self.is_following_path:
        #    self.path.revert_point()
        
        self.traveling = False
        self.is_following_path = False

        self.move_base_client.cancel_goal()

    
    def move_forward(self, forward_speed):
        self.move_twist_publisher.publish(Twist(Vector3(forward_speed, 0.0, 0.0), Vector3(0.0, 0.0, 0.0)))

    def rotate_deg(self, angle):
        self.move_twist_publisher.publish(Twist(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, (angle / 360.0) * 2 * 3.14)))

