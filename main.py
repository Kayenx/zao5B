import sys
import time
from tkinter import font
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob

def face_detect():
    cv2.namedWindow("face_detect",0)
    video_cap = cv2.VideoCapture("fusek_face_car_01.avi")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    profile_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
    eye_cascade= cv2.CascadeClassifier("eye_cascade_fusek.xml")

    desired_size = (100, 100)  
    
    all_images = []
    for img_path in glob.glob("eyes/*.png"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, desired_size)
        label = int(img_path.split('_')[4])
        all_images.append((img, label)) 
      
    images = np.array([data[0] for data in all_images])
    labels = np.array([data[1] for data in all_images])    
    recognizer = cv2.face.LBPHFaceRecognizer.create(1,12,8,8) 
    recognizer.train(images, labels)    
    while True:
        ret, frame = video_cap.read()
        paint_frame = frame.copy()
        if ret is True:
            faces,reject_levels,level_weights = face_cascade.detectMultiScale3(frame,
                                                   1.2,
                                                   3,
                                                   minSize=(100,100),
                                                   maxSize=(500,500),
                                                   outputRejectLevels=True
                                                   )
            i = 0
            for one_face in faces:
                if(level_weights[i] >= 3):        
                    cv2.rectangle(paint_frame,one_face,(0,0,0),12)
                    cv2.rectangle(paint_frame,one_face,(255,255,255),4)
                    x, y, w, h = one_face
                    roi_face = frame[y:y+h, x:x+w]
                    
                    eyes,reject_levels,level_weights_eyes = eye_cascade.detectMultiScale3(roi_face,
                                                   1.2,
                                                   3,
                                                   minSize=(100,100),
                                                   maxSize=(500,500),
                                                   outputRejectLevels=True
                                                   )
                    j = 0
                    for one_eye in eyes: 
                        if(level_weights_eyes[j] >= 6):
                            eye_x, eye_y, eye_w, eye_h = one_eye
                            eye_image = roi_face[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
                            eye_image = cv2.resize(eye_image, desired_size)
                            eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)                           
                            label, confidence = recognizer.predict(eye_image)
                            if label == 1:
                                cv2.rectangle(paint_frame, one_eye + (x, y, 0, 0), (0, 255, 0), 12)
                                cv2.rectangle(paint_frame, one_eye + (x, y, 0, 0), (255, 255, 255), 4) 
                            else:
                                cv2.rectangle(paint_frame, one_eye + (x, y, 0, 0), (0, 0, 255), 12)
                                cv2.rectangle(paint_frame, one_eye + (x, y, 0, 0), (255, 255, 255), 4)                                
                        j += 1                                                              
                i= i + 1
                
            cv2.imshow("face_detect",paint_frame)
            if cv2.waitKey(2) == ord("q"):
                break
face_detect()