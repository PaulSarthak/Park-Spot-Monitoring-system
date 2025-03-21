# Main Project file
#Made with ❤️ by Sarthak Paul

import cv2
import numpy as np
import pickle
import cvzone

#Set width and height of rectangle or import it from already created file
width, height = 107, 48
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

#open video file
cap=cv2.VideoCapture('carPark.mp4')

# Space counter function as well as display other metrics on the output
def checkParkingSpace(imgPro):
    spaceCounter=0
    for pos in posList:
        x,y=pos


        imgCrop = imgPro[y:y+height, x:x+width]
        # cv2.imshow(str(x+y), imgCrop)
        count=cv2.countNonZero(imgCrop)

        if count<930:
            color=(0,255,0)
            thickness=5
            spaceCounter +=1
        else:
            color=(0,0,255)
            thickness = 2
        # for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 5), scale=1.5, thickness=2, offset=0,colorR=color)
    cvzone.putTextRect(img,f'Empty Spots: {str(spaceCounter)}', (50,50), scale=3, thickness=5, offset=20, colorR=(255,0,0))
    #Stamp
    cvzone.putTextRect(img, "Made by Sarthak Paul and soham", (300, 700), scale=2, thickness=2, offset=5)

# Image processing
while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(3,3),1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)

    imgDilate=cv2.dilate(imgMedian,np.ones((3,3),np.uint8),iterations=1)


    checkParkingSpace(imgDilate)   #Send the processed frame back to counter function

    cv2.imshow('image',img)
    # cv2.imshow('ImageBlur', imgMedian)
    cv2.waitKey(10)


    # Resize all images to the same size for concatenation
    imgGray_resized = cv2.resize(imgGray, (300, 300))
    imgBlur_resized = cv2.resize(imgBlur, (300, 300))
    imgThreshold_resized = cv2.resize(imgThreshold, (300, 300))
    imgMedian_resized = cv2.resize(imgMedian, (300, 300))

    # Concatenate images horizontally
    imgStack = cv2.hconcat([imgGray_resized, imgBlur_resized, imgThreshold_resized, imgMedian_resized])

    # Display the concatenated image
    cv2.imshow('Image Processing Steps', imgStack)

    # Process the final dilated image
    imgDilate = cv2.dilate(imgMedian, np.ones((3, 3), np.uint8), iterations=1)
    checkParkingSpace(imgDilate)  # Send the processed frame back to counter function

    # Display the original image with parking space info
    cv2.imshow('Parking Space Detection', img)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




