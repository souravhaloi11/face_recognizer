import cv2
import numpy as np
import face_recognition
import PIL
from PIL import Image
#here first we call the elon musk image for recoginition
imgSourav = face_recognition.load_image_file('sourav.jpg')
#here convert the color RGB to BGR
imgSourav = cv2.cvtColor(imgSourav,cv2.COLOR_BGR2RGB)
#here first one face recognition is apply in diffrent pic
imgTest = face_recognition.load_image_file('Bidisha.jpg')
#imgTest = face_recognition.load_image_file('bill gates.jpg')
#here covnvert the img into RGB to BGR
imgTest = cv2.cvtColor(imgSourav,cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgSourav)[0]
encodeSourav = face_recognition.face_encodings(imgSourav)[0]
#print(faceLoc) 
cv2.rectangle(imgSourav,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodeSourav],encodeTest)

# = face_recognition.compare_faces(data["encodings"],encoding)



cv2.imshow("sourav",imgSourav)
#for showing the img
cv2.imshow("sourav",imgTest)
#for showing the img
#cv2.imread('Elon musk.jpg')

cv2.waitKey(0)