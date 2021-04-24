cmake_minimum_required(VERSION 2.8.3)
project(face_detector)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	cv_bridge
	sensor_msgs
	visualization_msgs
	tf2_ros
	nav_msgs
	std_msgs
	geometry_msgs
	message_generation
	message_filters
)

add_message_files(
  FILES
  FaceDetected.msg
  Detected.msg
  CylinderD.msg
)

generate_messages(DEPENDENCIES std_msgs)

find_package( OpenCV REQUIRED )

catkin_package(
	CATKIN_DEPENDS message_runtime roscpp rospy std_msgs message_filters
)
