import cv2
import numpy as np
import utils

BRIGHTNESS = 10
WIDTH = 3
HEIGHT = 4

webcam = True
path = '../assets/1.jpg'
cap = cv2.VideoCapture(0)
cap.set(BRIGHTNESS,50) 
cap.set(WIDTH,1280)
cap.set(HEIGHT,720)
scale = 3
paperWidth = 210 * scale
paperHeight = 297 * scale


while True:
    
    if webcam:success,img = cap.read()
    else: img = cv2.imread(path)
    
    img , contours = utils.getContours(img,minArea=50000,filter=4)
    
    if len(contours) != 0:
        biggest = contours[0][2]
        #print(biggest)
        imgWarp = utils.warpImg(img,biggest,paperWidth,paperHeight)
        
        #IMAGEM COM OS CONTORNOS A SEREM MEDIDOS
        img2 , contours2 = utils.getContours(imgWarp,minArea=2000,filter=4,threshold=[50,50],draw=True)
        
        if len(contours) != 0:
            for obj in contours2:
                cv2.polylines(img2,[obj[2]],True,(0,255,0),2)
                newPoints = utils.reorder(obj[2])
                # ACHANDO AS DISTANCIAS ESTRE UM OBJETO RETANGULAR
                newW = round( utils.findDistance(newPoints[0][0]//scale,nPoints[1][0]//scale),1) #Dividir pelo scale pra ficar correto )
                newH = round( utils.findDistance(newPoints[0][0]//scale,nPoints[2][0]//scale),1) #Dividir pelo scale pra ficar correto )
                
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

        cv2.imshow('A4',img2)

    
    img = cv2.resize(img,(0,0),None,0.4,0.4)
    cv2.imshow('Original',img)
    cv2.waitKey(1)
