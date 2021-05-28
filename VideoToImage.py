import BEN_detectFinger as finger 
import cv2
import time 
import os
ben = finger.handLandmarks()
pTime = 0
cTime = 0
video ="handBen2.mp4"
video = '/home/pcwork/ai/ftech/finger/DatasetToHandVitual/helloVideo/hello_01'
video = '/home/pcwork/depthai-python/examples/vaytay.avi' 
video = 0
video = 'rtsp://ftech:ad1235min@192.168.130.27/live0'
#video = '/home/pcwork/ai/ftech/finger/CNN/quang/My video - Ngày (2).mp4' 
cap = cv2.VideoCapture(video)
name = "like1" 
try: 
    os.mkdir ("data") 
except :
    pass 

try: 
    os.mkdir (f"data/{name}") 
except :
    pass 


count = 0 
while True :

    success, img = cap.read()
    ben.showFinger( img )
    pointList, box  = ben.storePoint ( img )
    check , img1 , img2 = ben.drawAndResize( img , box ) 
    if success == False :
        break 
    # ben.findFingerUp(pointList)
    #print ( box  )
    #print ( pointList[0] )
    print ( len ( box ) )
    if len ( box ) != 0 :

        cv2.rectangle( img , ( box[0] - 20 , box[1] - 20  ) , ( box[2] + 20 , box[3]+ 20  ) , (0,255,0),2 )

    #if len ( box ) != 0 :

        #cv2.rectangle( img2 , ( box[0] - 20 , box[1] - 20  ) , ( box[2] + 20 , box[3]+ 20  ) , (0,255,0),2 )

    cTime = time.time()
    fps = 1/( cTime - pTime )
    pTime = cTime
    cv2.putText( img , str( int ( fps ) ) , (10,70) , cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 255 ) ,3 )
    cv2.imshow("ben", img )
    cv2.imwrite(f"data/{name}/{name}_{count}.jpg", img2 )
    print ( f"data/{name}/{name}_{count}.jpg") 
    count +=1 
    #if check  == True :
    #    cv2.imshow("image", img2  )
    #    cv2.imshow("ben1", img1)
    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
cap.release()

