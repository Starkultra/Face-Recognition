import cv2 as cv
import numpy as np
import os
from PIL import Image

# Path for face image database(storage to train classifier)
path = 'dataset'

recognizer = cv.face.LBPHFaceRecognizer_create()
detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def ImageLabeling(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:  #Co-ordinates around the face
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = ImageLabeling(path)

recognizer.train(faces, np.array(ids))

# Save the model into trainner/trainer.yml

recognizer.write('trainer/trainer.yml') 

# Print the numer of faces trained and end program

print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))