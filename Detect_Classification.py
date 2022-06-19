from time import sleep
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

facedetect = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

model = load_model('./BC1.h5',compile=False)
#L = os.listdir('./input/train')
L = {0: 'B', 1: 'C', 2: 'D',3:'A' }

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    faces = facedetect.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        crop_img=frame[y:y+h,x:x+h]
        img = cv2.resize(crop_img,[224,224])
        img=img_to_array(img)
        img=img/255
        img=np.expand_dims(img,[0])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        answer=model.predict(img)
        y_class = answer.argmax(axis=-1)
        y = " ".join(str(x) for x in y_class)
        res = L[int(y)]
        cv2.putText(frame,res+'   '+str(np.max(answer)),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FONT_HERSHEY_SIMPLEX,2,cv2.LINE_AA)
   
    cv2.imshow('VIDEO',frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()