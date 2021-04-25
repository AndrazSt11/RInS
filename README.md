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

## How to run Cylinder segmentation 
```bash
roslaunch object_detection rins_world.launch 
roslaunch simulation amcl_simulation.launch 
roslaunch turtlebot_rviz_launchers view_navigation.launch 
roslaunch object_detection find_cylinder.launch 
rosrun face_detector cylinder_markers 
rosrun main_package Main.py
```
