# Face-Detection-and-Recognition
## Introduction
Face detection is a computer vision technology that helps to locate/visualize human faces in digital images. This technique is a specific use case of object detection     technology that deals with detecting instances of semantic objects of a certain class (such as humans, buildings or cars) in digital images and videos. With the advent of technology, face detection has gained a lot of importance especially in fields like photography, security, and marketing
There are a number of detectors other than the face, which can be found in the library. Feel free to experiment with them and create detectors for eyes, license plates, etc.

## Pre-requisites
- [x] Python
- [x] Numpy
- [x] Opencv
## Explanation 
 There are three .py file(face_dataset,training,face_Recognition), two folder(dataset,trainer) and one .xml file.
 face_dataset.py is used for creating database to store the captured face for training. Once face_dataset.py file is executed it caputures the image and stored in dataset folder
## Technique Used
- [x] Local Binary Pattern(LBP) classifier(for recognition)
- [x] Haarcascade classifier(for detection)
### How Haar cascade Classifier works(Feature detection)
<p align="center"><img src="https://github.com/Starkultra/Face-Recognition-using-Opencv/blob/main/Asset/haar.png" width=40%></p>
<p align="center"><img src="https://github.com/Starkultra/Face-Recognition-using-Opencv/blob/main/Asset/haar2.png" width=40%></p>

The dataset captured were passed to trainer.xml file for training 
<p align="center"><img src="https://github.com/Starkultra/Face-Recognition-using-Opencv/blob/main/Asset/trainer.png" alt="Trainer image" width=60%></p>

## Result
<p align="center"><img src="https://github.com/Starkultra/Face-Recognition-using-Opencv/blob/main/Asset/result.png" width=50% alt="Face recognition image"></p>

## Conclusion 
Feel free to try different classifier to detect smile,eyes,etc and also try different Algorithm for [Face recognition](https://github.com/opencv/opencv/tree/master/data/haarcascades)

