# RInS - team Omnicorn

### How to run
```bash
roslaunch simulation rins_world.launch
roslaunch simulation amcl_simulation.launch 
roslaunch turtlebot_rviz_launchers view_navigation.launch
roslaunch face_detector start.launch 
rosrun sound_play soundplay_node.py
rosrun main_package Main.py
``` 

## How to run Cylinder segmentation and Ring detection
```bash
roslaunch robot_arm rins_world.launch
roslaunch simulation amcl_simulation.launch 
roslaunch turtlebot_rviz_launchers view_navigation.launch 
roslaunch object_detection find_cylinder.launch 
rosrun face_detector cylinder_markers 
rosrun object_detection detect_rings 
rosrun object_detection ring_markers 
rosrun robot_arm move_arm.py
rosrun main_package Main.py
```
