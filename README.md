# RInS - team Omnicorn

### Run everything
```bash
# Simulation 
roslaunch simulation rins_world.launch 2>/dev/null
roslaunch simulation amcl_simulation.launch 2>/dev/null
roslaunch turtlebot_rviz_launchers view_navigation.launch 2>/dev/null

# Detectors
roslaunch face_detector start.launch 2>/dev/null
roslaunch object_detection find_cylinder.launch 2>/dev/null
rosrun object_detection detect_rings 2>/dev/null

# Markers
roslaunch markers start.launch

# Utility
rosrun robot_arm move_arm.py 
rosrun sound_play soundplay_node.py 

# QR and digit extractor 
rosrun data_viewer extract_qr
rosrun data_viewer extract_digits

# Main
rosrun main_package Main.py
```

### DEPRECATED
```bash
# roslaunch simulation rins_world.launch
# roslaunch simulation amcl_simulation.launch 
# roslaunch turtlebot_rviz_launchers view_navigation.launch
# roslaunch face_detector start.launch 
# rosrun sound_play soundplay_node.py
# rosrun main_package Main.py
``` 

```bash
# roslaunch simulation rins_world.launch 2>/dev/null
# # if robot arm is activated: roslaunch robot_arm rins_world.launch 2>/dev/null
# roslaunch simulation amcl_simulation.launch 2>/dev/null
# roslaunch turtlebot_rviz_launchers view_navigation.launch  2>/dev/null
# roslaunch object_detection find_cylinder.launch 2>/dev/null
# rosrun face_detector cylinder_markers 2>/dev/null
# rosrun object_detection detect_rings 2>/dev/null
# rosrun object_detection ring_markers 2>/dev/null
# # if robot arm is activated: rosrun robot_arm move_arm.py 
# rosrun sound_play soundplay_node.py
# rosrun main_package Main.py
```
