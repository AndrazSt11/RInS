import rospy
from time import sleep
from move_manager.mover import Mover

if __name__ == '__main__':

    m = Mover()
    m.follow_path()

    sleep(10)

    m.stop_robot()
    m.move_to(-0.052309419959783554, -1.1175440549850464)

    sleep(4)
    m.follow_path()

    rospy.spin()