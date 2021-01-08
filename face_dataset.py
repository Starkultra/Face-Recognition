import cv2 as cv
import os

cap = cv.VideoCapture(0)
cap.set(3, 640) # set video width
cap.set(4, 480) # set video height

face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id')

print("\n Initializing face capture....")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cap.read()
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:

        cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 4)     
        count += 1

        # Save the captured image into the datasets folder
        cv.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv.imshow('image', img)

    if cv.waitKey(100) & 0xff == ord("q"):
        break
    elif count >= 50: 
         break

# Do a bit of cleanup
print("\n Exiting Program")
cap.release()
cv.destroyAllWindows()