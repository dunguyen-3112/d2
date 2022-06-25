import cv2
import os
import uuid
# Establish a connection to the webcam
ANC_PATH = 'Toan'
cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
while cap.isOpened(): 
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        crop_img=frame[y:y+h,x:x+h]
        img = cv2.resize(crop_img,[224,224])
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        break
    
    cv2.imwrite(imgname, img)

    cv2.imshow('Image', frame)
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        

cap.release()

cv2.destroyAllWindows()