"""
Face detection Module

@auther : alisharify
2023/9/2 - 1402/6/8

"""


import time
import datetime
from threading import Thread

import cv2 as cv
from deepface import DeepFace


global faceMatch
global FaceData
global CheckFlag

CheckFlag = True
faceMatch = False
FaceData = []
sourceImage = cv.imread("./Media/alisher.jpg")
sourceName = "Alisher"

# use OpenCV Haar cascade to detect multiple faces quickly
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)


def checkFace(frame):
    """
    Detect multiple faces in the frame (using Haar cascade) and verify each
    against the `sourceImage` using DeepFace.verify.

    Produces a list `FaceData` where each entry is a dict:
      {'rect': (x,y,w,h), 'verified': bool, 'distance': float|None}
    """
    global faceMatch
    global CheckFlag
    global FaceData

    FaceData = []
    faceMatch = False
    try:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # crop the face region and verify against source image
            roi = frame[y:y + h, x:x + w]
            try:
                # enforce_detection=False reduces errors for small / partial crops
                res = DeepFace.verify(roi, sourceImage, enforce_detection=False)
                verified = bool(res.get('verified', False))
                distance = res.get('distance', None)
            except Exception:
                verified = False
                distance = None

            FaceData.append({'rect': (x, y, w, h), 'verified': verified, 'distance': distance})

        # if any face verified, set faceMatch True
        faceMatch = any([d['verified'] for d in FaceData])
    except Exception:
        FaceData = []
        faceMatch = False

    CheckFlag = True


# def AnalizyeFace(frame):
#  Analizye's face motion and race and gender
#     global Counter
#     Counter += 1
#     if Counter  == 60:
#         Counter = 0
#         result = DeepFace.analyze(frame)[0]
#         print(json.dumps(result, indent=1))

        

timer = time.time()
imageCounter = 0
FPS = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    if CheckFlag:
        counter = 0
        print(f"Thread Started !{imageCounter}")
        Thread(target=checkFace, args=(frame.copy(), )).start()
        CheckFlag = False

    if faceMatch:
        frame = cv.putText(img=frame,
            org=(0, 690) ,text="Face Match",  color=(0, 0, 255),
            fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=3, fontScale=1.5)

        frame = cv.rectangle(frame, (0,698), (300,698), color=(0 , 0, 255), thickness=2)


        x = "img1"
        startPoint = (FaceData[x]['x'], FaceData[x]['y'],)
        endPoint = (FaceData[x]['x']+FaceData[x]['w'], FaceData[x]['y']+FaceData[x]['h'],)
        
        # draw a rectangle around the face in the frame
        frame = cv.rectangle(frame, startPoint, endPoint, color=(0 , 255, 0), thickness=3)

        # add person name to frame
        frame = cv.putText(img=frame,
            org=(endPoint[0]-100, endPoint[1]+20), text=sourceName,
            color=(0, 0, 255), fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=2, fontScale=0.6)

        frame = cv.putText(img=frame,
            org=(endPoint[0]-100, endPoint[1]+40), text=f"x:{startPoint[0]},y:{startPoint[1]}",
            color=(0, 0, 255), fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=1, fontScale=0.6)


    else:
        frame = cv.putText(img=frame,
            org =(0, 690) , text="No Match", color=(0, 255, 0), 
            fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=3, fontScale=1.5)
        frame = cv.rectangle(frame, (0,698), (225, 698), color=(0 , 255, 0), thickness=2)

    frame = cv.putText(img=frame,
        org=(0, 715), text=f"{datetime.datetime.utcnow()}", color=(0, 0, 255),
        fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=1, fontScale=0.6)


    # FPS
    imageCounter += 1
    now = time.time()
    if now - timer > 1:
        FPS = imageCounter / 1
        imageCounter = 0
        timer = time.time()
    
    frame = cv.putText(img=frame,
        org=(0, 20), text=f"FPS:{FPS}",
        color=(0, 0, 255), fontFace=cv.FONT_HERSHEY_DUPLEX, thickness=1, fontScale=0.6)

    frame = cv.resize(frame, (920, 640))
    cv.imshow("video", frame)
    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()

