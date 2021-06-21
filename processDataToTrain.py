import BEN_detectFinger as finger 
import cv2
import time 
import os
import depthai as dai 
import argparse 

#create add the name label 

parser = argparse.ArgumentParser() 
parser.add_argument('-name','--name',type=str, help="create name of labels") 
parser.add_argument('-camera','--camera', type=int, help="chosse the source camera")  
parser.add_argument('-depth','--depthai',action="store_true", help="using the source camera depthai" ) 


args = parser.parse_args()  
print (args.camera )
if args.name == None :
    raise RuntimeError ("please write name of label, --name __ ")
else :
    name = args.name 

if args.depthai == False and args.camera == None :
    raise RuntimeError ("No source selected, --camera or --depthai") 

if args.depthai == True  and args.camera !=  None : 
    raise RuntimeError ("Please chosse 1 source camera") 

if args.depthai: 
    #create pipeline
    pipeline = dai.Pipeline() 
    camRgb = pipeline.createColorCamera()  
    xoutVideo = pipeline.createXLinkOut() 
    
    xoutVideo.setStreamName("video") 
    
    # Properties 
    camRgb.setBoardSocket( dai.CameraBoardSocket.RGB) 
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) 
    camRgb.setVideoSize( int(1920*2/3), int(1080*2/3))  
    xoutVideo.input.setBlocking(False) 
    xoutVideo.input.setQueueSize(1)  
    # linking  
    camRgb.video.link(xoutVideo.input) 
elif args.camera != None : 
    cap = cv2.VideoCapture( args.camera ) 


#set up API handlandmarks  

ben = finger.handLandmarks()
pTime = 0
cTime = 0
try: 
    os.mkdir ("data") 
except :
    pass 

try: 
    os.mkdir (f"data/{name}") 
except :
    pass 


# ussing camera source depthai 
if args.depthai: 
    with dai.Device(pipeline) as device:
        device.startPipeline() 
        video = device.getOutputQueue(name="video", maxSize=4, blocking=False)  
        count = 0 
        while True :
            videoIn = video.get() 
            #get BGR video from camera 
            img = videoIn.getCvFrame() 
    
            ben.showFinger( img )
            pointList, box  = ben.storePoint ( img )
            check , img1 , img2 = ben.drawAndResize( img , box ) 
            # ben.findFingerUp(pointList)
            #print ( box  )
            #print ( pointList[0] )
            #print ( len ( box ) )
            if len ( box ) != 0 :
        
                cv2.rectangle( img , ( box[0] - 20 , box[1] - 20  ) , ( box[2] + 20 , box[3]+ 20  ) , (0,255,0),2 )
        
            #if len ( box ) != 0 :
        
                #cv2.rectangle( img2 , ( box[0] - 20 , box[1] - 20  ) , ( box[2] + 20 , box[3]+ 20  ) , (0,255,0),2 )
        
            cTime = time.time()
            fps = 1/( cTime - pTime )
            pTime = cTime
            cv2.putText( img , str( int ( fps ) ) , (10,70) , cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 255 ) ,3 )
            cv2.imshow("ben", img )
            if check and img2.size == 10000 :
                cv2.imwrite(f"data/{name}/{name}_{count}.png", img2 )
                img2 = cv2.flip(img2, 1 )
                count +=1 
                cv2.imwrite(f"data/{name}/{name}_{count}.png", img2 )
                count += 1
                #print ( f"data/{name}/{name}_{count}.jpg") 
            #if check  == True :
            cv2.imshow("image", img2  )
            #    cv2.imshow("ben1", img1)
            if cv2.waitKey(1) == 27:
                break
        
        cv2.destroyAllWindows()
        cap.release()
#ussing another source camera such as webcam, rtsp , ....

elif args.camera != None : 
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
        #print ( len ( box ) )
        if len ( box ) != 0 :
    
            cv2.rectangle( img , ( box[0] - 20 , box[1] - 20  ) , ( box[2] + 20 , box[3]+ 20  ) , (0,255,0),2) 
        #if len ( box ) != 0 :
    
            #cv2.rectangle( img2 , ( box[0] - 20 , box[1] - 20  ) , ( box[2] + 20 , box[3]+ 20  ) , (0,255,0)
    
        cTime = time.time()
        fps = 1/( cTime - pTime )
        pTime = cTime
        cv2.putText( img , str( int ( fps ) ) , (10,70) , cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 255 ) ,3 )
        cv2.imshow("ben", img )
        if check and img2.size == 10000 :
            cv2.imwrite(f"data/{name}/{name}_{count}.png", img2 )
            img2 = cv2.flip(img2, 1 )
            count +=1
            cv2.imwrite(f"data/{name}/{name}_{count}.png", img2 )
            count += 1
            #print ( f"data/{name}/{name}_{count}.jpg")
        #if check  == True :
        cv2.imshow("image", img2  )
        #    cv2.imshow("ben1", img1)
        if cv2.waitKey(1) == 27:
            break
    
