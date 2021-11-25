import cv2
import utils

BRIGHTNESS = 10
WIDTH = 3
HEIGHT = 4

webcam = True
path = '../../assets/1.png'
cap = cv2.VideoCapture(0)
cap.set(BRIGHTNESS,10)
cap.set(WIDTH,1920)
cap.set(HEIGHT,1080)
scale = 3
paperWidth = 210 * scale
paperHeight = 297 * scale


while True:
    
    if webcam:success,img = cap.read()
    else: img = cv2.imread(path)
    
    img , contours = utils.getContours(img, minArea=50000, filter=4)
    
    if len(contours) != 0:
        biggest = contours[0][2]
        #print(biggest)
        imgWarp = utils.warpImg(img, biggest, paperWidth, paperHeight)
        
        #IMAGEM COM OS CONTORNOS A SEREM MEDIDOS
        img2 , contours2 = utils.getContours(imgWarp, minArea=2000, filter=4, threshold=[50, 50], draw=True)
        
        if len(contours) != 0:
            for obj in contours2:
                cv2.polylines(img2,[obj[2]],True,(0,255,0),2)
                newPoints = utils.reorder(obj[2])
                # ACHANDO AS DISTANCIAS ESTRE UM OBJETO RETANGULAR
                newW = round(utils.findDistance(newPoints[0][0] // scale, newPoints[1][0] // scale), 1) #Dividir pelo scale pra ficar correto )
                newH = round(utils.findDistance(newPoints[0][0] // scale, newPoints[2][0] // scale), 1) #Dividir pelo scale pra ficar correto )
                
                cv2.arrowedLine(img2, (newPoints[0][0][0], newPoints[0][0][1]), (newPoints[1][0][0], newPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(img2, (newPoints[0][0][0], newPoints[0][0][1]), (newPoints[2][0][0], newPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(img2, '{}cm'.format(newW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(img2, '{}cm'.format(newH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

        cv2.imshow('A4',img2)

    
    img = cv2.resize(img,(0,0),None,0.4,0.4)
    cv2.imshow('Original',img)
    cv2.waitKey(1)
