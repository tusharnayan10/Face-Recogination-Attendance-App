import cv2
import numpy as np
import face_recognition
import os
from datetime import  datetime
# print("Hello coders")

path = 'imageTest'
images = []
className = []
mylist = os.listdir(path)

#Dynamically read files from folder
for classList in mylist:
    curImg = cv2.imread(f'{path}/{classList}')
    images.append(curImg)
    #split extendtion from image .jpg
    className.append(os.path.splitext(classList)[0])
print(className)

def findEncoding(images):
    encodeList = []
    for img in images:
        #Conver to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Mark attendance in Attendance.csv folder
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H : %M : %S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnow = findEncoding(images)
print('Encoding Processing !!')

#Open Camera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    #Resize image pixel by 1/4
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    # Conver to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # To Locate face's in current frame
    faceInCurrFrame = face_recognition.face_locations(imgS)
    #Encode current face's in Frame
    encodeCurrFrame = face_recognition.face_encodings(imgS,faceInCurrFrame)

    for encodeFace,faceLoc in zip(encodeCurrFrame,faceInCurrFrame):
        #Performing face match with known list
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        #Check Distance( Smaller the distance more the accuracy)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        #print(faceDis)

        matchIndex = np.argmin(faceDis)

        #Diplay if match found
        if matches[matchIndex]:
            # Diplay name if match found
            name = className[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            #resize to original
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            # Draw rectangle on webcam Image
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
            # Draw rectangle on original Image
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
            markAttendance(name)
    #Show Webcam
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

#To Locate face in image
#faceLoc = face_recognition.face_locations(imgRobert)[0]
#encodeimgRobert = face_recognition.face_encodings(imgRobert)[0]
#cv2.rectangle(imgRobert,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#To Locate face in image
#faceLocTest = face_recognition.face_locations(imgRobertTest)[0]
#encodeimgRobertTest =  face_recognition.face_encodings(imgRobertTest)[0]
#cv2.rectangle(imgRobertTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLoc[2]),(255,0,255),2)

#Compare face's
#result = face_recognition.compare_faces([encodeimgRobert],encodeimgRobertTest)
#Compare face distance
#faceDist = face_recognition.face_distance([encodeimgRobert],encodeimgRobertTest)