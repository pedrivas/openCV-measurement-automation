import cv2
import numpy as np

def getContours(img,threshold=[100,100],showCanny=False,minArea=1000,filter=0,draw=False):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,threshold[0],threshold[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)
    imgThreshold = cv2.erode(imgDial,kernel,iterations=2)
    if showCanny:cv2.imshow('Canny', imgThreshold)
    
    contours,hierarchy = cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    finalContours = []
    
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            perimeter = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*perimeter,True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx),area,approx,bbox,i])
            else:
                finalContours.append([len(approx),area,approx,bbox,i])
    
    finalContours = sorted(finalContours,key = lambda x:x[1],reverse=True)
    if draw:
        for contours in finalContours:
            cv2.drawContours(img,contours[4],-1,(0,0,255),3)
    
    return img,finalContours
            
