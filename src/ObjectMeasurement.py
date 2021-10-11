import cv2
import numpy as np
import utils

BRIGHTNESS = 10
WIDTH = 3
HEIGHT = 4

webcam = False
path = '../assets/1.jpg'
cap = cv2.VideoCapture(0)
cap.set(BRIGHTNESS,55) 
cap.set(WIDTH,1280)
cap.set(HEIGHT,720)

while True:
    
    if webcam:success,img = cap.read()
    else: img = cv2.imread(path)
    
    img , finalContours = utils.getContours(img,showCanny=True,draw=True)
    
    img = cv2.resize(img,(0,0),None,0.4,0.4)
    cv2.imshow('Original',img)
    cv2.waitKey(1)
