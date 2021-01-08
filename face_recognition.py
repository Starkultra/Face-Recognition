import cv2 as cv
import numpy as np
import os 

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascadePath);

font = cv.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

names = ['None', 'Prasath', 'Elon musk', 'Gates', 'Mark Zuckerberg', 'W'] 

# Initialize and start realtime video capture
cap = cv.VideoCapture(0)
cap.set(3, 640) # set video widht
cap.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)

while True:

    ret, img =cap.read()
   

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv.imshow('camera',img) 

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Do a bit of cleanup
print("\n Exiting Program")
cap.release()
cv.destroyAllWindows()