from srcTensorflow import BEN_predict as model
import BEN_detectFinger as finger 
import cv2 
import os 
import time 
ben = finger.handLandmarks() 
JSON_FILE = "modelTensorflow/model3.json"
#WEIJSON_FILE = "jsonEyeModelV1.json"
WEIGHTS_FILE = "modelTensorflow/model3.h5"
#GHTS_FILE = "eyeModelV1.h5"

pTime = 0 
cTIme = 0 

model = model.runModel( JSON_FILE, WEIGHTS_FILE )  

MODEL = model.wakeUpModel() 

video = 0 
cap = cv2.VideoCapture( video ) 
predict = ' ' 
acc = 0 

while True : 
    success, img = cap.read()
    ben.showFinger( img )
    pointList, box  = ben.storePoint ( img )
    check , img1 , img2 = ben.drawAndResize( img , box )
    if success == False :
        break

    if len ( box ) != 0 :

        cv2.rectangle( img , ( box[0] - 20 , box[1] - 20  ) , ( box[2] + 20 , box[3]+ 20  ) , (0,255,0),2 )

    cTime = time.time()
    fps = 1/( cTime - pTime )
    pTime = cTime
    cv2.putText( img , f"Fps: {int(fps)}" , (6,50) , cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255 ) ,3 )
    if check and img2.size == 100*100 :
        predict, acc  = model.getResult(img2, MODEL , size = 26)
        if predict != False :
            try:

                cv2.putText( img , f"Tensorflow: {predict} {int(acc*100)} %" , (200,50) , cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255 ) ,3 )
            except : 

                cv2.putText( img , f"Tensorflow:          %" , (200,50) , cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255 ) ,3 )
        else:
            cv2.putText( img ,     f"Tensorflow:          %" , (200,50) , cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255 ) ,3 )
    else : 
        cv2.putText( img ,         f"Tensorflow:          %" , (200,50) , cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255 ) ,3 )
    cv2.imshow("image", img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()



    
