RIS: Observation model
In this observartion model we have compared dlib CNN face detector and openCv dnn face detector.

Experiment setup:
Testing enviroment consisted of an image of human face placed on a wall and additional items such as books, plant and workout mat.
Testing videos were captured from 50, 30, 0, -30 and 50 degrees relative to face position on the wall. They were also
captured in sun light and in artificial lighting conditions and with or without camera motion.
Example of a video name: <lighting_condition>_<Motion(optional)>_<degree_captured>.mp4 (ArtificialLight_Motion_30.mp4) 

Processing:
Each frame image is processed by both detectors. Face detection is evaluated, based on assumption that image is approximately at frame center.
If center of detected bounding box is approximately at image center, then face is detected. If it is not at the center of an image, it
is considered as false positive. If there was no detection at all, it is considered as false negative.
All statistics is kept on per meter basis.

Execution:
When running the experiment, statistic is printed to console and to two files ('face_detector_dnn.txt' and 'face_detector_dlib.txtx').
You can also see in real-time where face was detected. Please refer to 'ObservationModel.png' image for example(Dnn - red bounding box, Dlib - green bounding box).

Observations:
As expected dlib CNN was performing better that openCv face detector, but it was only a little bit better and it is quite slower to process
than openCv face detector.
I was suprised to see, that both detectors struggled to detect faces from 4 to 3 meters away. Here neural network based dlib detecotor
was more successful, thus it might be more useful for space exploration.
Motion blur introduced worse performance to both detectors, but they were not affected severely and introduced motion
blur was large, which should not occur on real robot.