import rospy
from time import sleep
from geometry_msgs.msg import Point, Quaternion
from move_manager.mover import Mover

if __name__ == '__main__':

    rospy.init_node("mover_client")

    m = Mover()

    print(m.get_pose())
    m.follow_path()

    sleep(4)

    m.stop_robot()

    sleep(1)
    m.move_to(Point(-0.052309419959783554, -1.1175440549850464,0), Quaternion(0,0,0,1))
    m.move_to(Point(-0.052309419959783554, -1.1175440549850464,0), Quaternion(0,0,0,1))
    sleep(1)
    m.move_to(Point(-0.052309419959783554, -1.1175440549850464,0), Quaternion(0,0,0,1))

    m.stop_robot()
    sleep(1)
    m.move_to(Point(-0.052309419959783554, -1.1175440549850464,0), Quaternion(0,0,0,1))
    m.follow_path()
    m.follow_path()

    print(m.get_pose())

    sleep(1)
    m.follow_path()

    rospy.spin()