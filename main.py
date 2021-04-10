import cv2
import numpy as np
import face_recognition
# print("Hello coders")

# Loading image from Folder
imgRobert = face_recognition.load_image_file('imageTest/disha.jpeg')
# Convert image to RGB format
imgRobert = cv2.cvtColor(imgRobert,cv2.COLOR_BGR2RGB)

# Loading image (Testing)
imgRobertTest = face_recognition.load_image_file('imageTest/dishaTest.jpg')
imgRobertTest = cv2.cvtColor(imgRobertTest,cv2.COLOR_BGR2RGB)

#To Locate face in image
faceLoc = face_recognition.face_locations(imgRobert)[0]
encodeimgRobert = face_recognition.face_encodings(imgRobert)[0]
cv2.rectangle(imgRobert,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#To Locate face in image
faceLocTest = face_recognition.face_locations(imgRobertTest)[0]
encodeimgRobertTest =  face_recognition.face_encodings(imgRobertTest)[0]
cv2.rectangle(imgRobertTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLoc[2]),(255,0,255),2)

#Compare face's
result = face_recognition.compare_faces([encodeimgRobert],encodeimgRobertTest)
#Compare face distance
faceDist = face_recognition.face_distance([encodeimgRobert],encodeimgRobertTest)
print(result,faceDist)

#Display result and face distance on screen
cv2.putText(imgRobertTest,f'{result} {round(faceDist[0],3)}',(10,10), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(129,178,20),1)

#Open images
cv2.imshow('Robert Jr',imgRobert)
cv2.imshow('Robert Jr Test',imgRobertTest)
cv2.waitKey(0)