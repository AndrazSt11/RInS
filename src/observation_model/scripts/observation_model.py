#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Observation:
# Dlib is much slower but a bit more accurate

import os
import math
import sys 
import cv2
import dlib
import numpy as np

class face_detector_dnn:
    def __init__(self):
        # The function for performin HOG face detection
        currentPath = os.path.dirname(os.path.abspath(__file__));
        self.face_net = cv2.dnn.readNetFromCaffe(currentPath + '/deploy.prototxt.txt',
                                                 currentPath + '/res10_300x300_ssd_iter_140000.caffemodel')

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)


    def find_faces(self, orgFrame, writeToFrame):
        detectionData = detection_data(writeToFrame)

        # Set the dimensions of the image
        self.dims = orgFrame.shape
        h = self.dims[0]
        w = self.dims[1]

        # Detect the faces in the image
        blob = cv2.dnn.blobFromImage(cv2.resize(orgFrame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        face_detections = self.face_net.forward()

        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]
            if confidence>0.5:

                box = face_detections[0,0,i,3:7] * np.array([w,h,w,h])
                box = box.astype('int')
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
                # Bounding box
                cv2.rectangle(detectionData.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                xCenter = (x2 + x1) / 2
                yCenter = (y2 + y1) / 2
                
                detectionData.xCenters.append(xCenter)
                detectionData.yCenters.append(yCenter)

        return detectionData


class face_detector_dlib:
    def __init__(self):
        # The function for performin HOG face detection
        self.face_detector = dlib.get_frontal_face_detector()

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)
    
    def find_faces(self, orgFrame, writeToFrame):
        detectionData = detection_data(writeToFrame)

        # Set the dimensions of the image
        self.dims = orgFrame.shape

        # Detect the faces in the image
        face_rectangles = self.face_detector(orgFrame, 0)
        for face_rectangle in face_rectangles:

            x1 = face_rectangle.left()
            x2 = face_rectangle.right()
            y1 = face_rectangle.top()
            y2 = face_rectangle.bottom()

            # Bounding box
            cv2.rectangle(detectionData.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            xCenter = (x2 + x1) / 2
            yCenter = (y2 + y1) / 2
            
            detectionData.xCenters.append(xCenter)
            detectionData.yCenters.append(yCenter)

        return detectionData


class detection_data:
    def __init__(self, frame):
        self.xCenters = []
        self.yCenters = []
        self.frame = frame

# distance: starting distance in meters - round to meters!
class detection_statistic:
    def __init__(self, videoName, distance, frameCount):
        self.centerOffsetMargin = 0.2
        

        self.videoName = videoName
        self.distance = distance
        self.frameCount = frameCount
        self.measureArea = round(frameCount / distance) + 1 # measure for each meter

        self.turePositive = []
        self.falsePositive = []
        self.falseNegative = []
        self.missedFrameIndices = []

        for i in range(0, self.distance, 1):
            self.turePositive.append(0)
            self.falsePositive.append(0)
            self.falseNegative.append(0)
            

    def process_detection(self, frameIndex, detectionData):
        measureAreaIndex = math.floor(frameIndex / self.measureArea)

        # Image dimension
        imageY = detectionData.frame.shape[0]
        imageX = detectionData.frame.shape[1]

        # Assume true face is at the center of an image
        imageCenterY = imageY / 2
        imageCenterX = imageX / 2

        if(len(detectionData.xCenters) > 0):
            for i in range(0, len(detectionData.xCenters), 1):
                diffX = abs(imageCenterX - detectionData.xCenters[i])
                diffY = abs(imageCenterY - detectionData.yCenters[i])
                
                # True positive
                if((diffX < imageX * self.centerOffsetMargin ) or (diffY < imageY * self.centerOffsetMargin)):
                    self.turePositive[measureAreaIndex] += 1
                else:
                    self.falsePositive[measureAreaIndex] += 1
        else:
            self.falseNegative[measureAreaIndex] += 1
            self.missedFrameIndices.append(frameIndex)

    def console_output(self):
        print("Video name: ", self.videoName)
        print("Distance", self.distance)
        print("Frame count: ", self.frameCount)
        print("True positives: ", self.turePositive)
        print("False positive: ", self.falsePositive)
        print("False negative: ", self.falseNegative)
        print("Missed frames:", self.missedFrameIndices)


# Dnn - RED bounding box
# Dlib - GREEN bounding box
def main():
    statisticsDnn = []
    statisticsDlib = []
    faceDetectorDnn = face_detector_dnn()
    faceDetectorDlib = face_detector_dlib()

    # TODO: parse distances -> 0, 30 = 4m, 50 = 3m
    resourceVideos = ["../video/ArtificialLight_0.mp4",
                      "../video/ArtificialLight_30.mp4",
                      "../video/ArtificialLight_-30.mp4",
                      "../video/ArtificialLight_50.mp4",
                      "../video/ArtificialLight_-50.mp4",
                      "../video/ArtificialLight_Motion_0.mp4",
                      "../video/ArtificialLight_Motion_30.mp4",
                      "../video/ArtificialLight_Motion_-30.mp4",
                      "../video/ArtificialLight_Motion_50.mp4",
                      "../video/ArtificialLight_Motion_-50.mp4",
                      "../video/SunLight_0.mp4",
                      "../video/SunLight_30.mp4",
                      "../video/SunLight_-30.mp4",
                      "../video/SunLight_50.mp4",
                      "../video/SunLight_-50.mp4",
                      "../video/SunLight_Motion_0.mp4",
                      "../video/SunLight_Motion_30.mp4",
                      "../video/SunLight_Motion_-30.mp4",
                      "../video/SunLight_Motion_50.mp4",
                      "../video/SunLight_Motion_-50.mp4"]

    for resource in resourceVideos:
        # Open video stram
        print ("Trying to open resource: " + resource)
        cap = cv2.VideoCapture(resource)

        if not cap.isOpened():
            print("Error opening resource: " + str(resource))
            exit(0)

        print("Correctly opened resource, starting to show feed.")

        winName = "Stream: " + resource
        fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Init statistic
        statisticDnn = detection_statistic(resource, 4, frameCount)
        statisticDlib = detection_statistic(resource, 4, frameCount)
        
        frameIndex = 1
        rval, frame = cap.read()
        while rval:
            # Face detection
            detectionData = faceDetectorDnn.find_faces(frame, frame)
            statisticDnn.process_detection(frameIndex, detectionData)

            detectionData = faceDetectorDlib.find_faces(frame, detectionData.frame)
            statisticDlib.process_detection(frameIndex, detectionData)

            # Show image preview
            cv2.imshow(winName, detectionData.frame)
            key = cv2.waitKey(1)

            # Next frame
            rval, frame = cap.read()
            frameIndex += 1

        statisticsDnn.append(statisticDnn)
        statisticsDlib.append(statisticsDlib)
        statisticDnn.console_output()
        statisticDlib.console_output()
        
        cv2.destroyWindow(winName)


    # TODO: output to file


if __name__ == '__main__':
    main()