import cv2
import numpy as np

BRIGHTNESS = 10
WIDTH = 3
HEIGHT = 4

webcam = False
path = './1.jpg'
cap = cv2.VideoCapture(0)
cap.set(BRIGHTNESS,55) 
cap.set(WIDTH,1280)
cap.set(HEIGHT,720)

while True:
    success,img = cap.read()
    
    cv2.imshow('Original',img)
    cv2.waitKey(1)
