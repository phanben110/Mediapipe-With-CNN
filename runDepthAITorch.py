import torch 
from torch.utils.data import DataLoader  
from torch.utils.data import Dataset  
import torchvision.transforms as transforms 
from torch import nn  
import cv2 
from srcTorch.BEN_processingData import imageDataset, processingDataset 
from srcTorch.BEN_modelCNN import CNN 
import BEN_detectFinger as finger 
import time
import numpy as np

import depthai as dai 

#Create pipeline  
pipeline = dai.Pipeline() 
#Define source and output 
camRgb = pipeline.createColorCamera() 
xoutVideo = pipeline.createXLinkOut()

xoutVideo.setStreamName("video") 

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(int ( 1920*2/3) ,int (1080 *2/3))

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Linking
camRgb.video.link(xoutVideo.input)


ben = finger.handLandmarks()
pTime = 0
cTime = 0

labels  = ['Ok', 'Silent', 'Dislike', 'Like', 'Hi', 'Hello', 'Stop' , ' ' ]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PATH = '/home/pcwork/ai/ftech/finger/CNN/modelPytorch/model2.pt' 
numClasses = 7 
model = CNN (numClasses) 
model.load_state_dict(torch.load(PATH))
model.eval()

print ( model )

dataTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(26),
    transforms.ToTensor()
    ])



#video = 'rtsp://ftech:ad1235min@192.168.130.27/live0'
#video = 'rtsp://ftech:ad1235min@192.168.130.81/live0'
#command line to scan IP ("nmap -sP 192.168.130.0/24") 
#video = 0

predict = ''
acc = 0


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    video = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        videoIn = video.get()

        # Get BGR frame from NV12 encoded video frame to show with opencv
        # Visualizing the frame on slower hosts might have overhead
        img = videoIn.getCvFrame()

        ben.showFinger( img )
        pointList, box  = ben.storePoint ( img )
        check , img1 , img2 = ben.drawAndResize( img , box )
        
        if len ( box ) != 0 :

            cv2.rectangle( img , ( box[0] - 20 , box[1] - 20  ) , ( box[2] + 20 , box[3]+ 20  ) , (0,255,0),2) 
        cTime = time.time()
        fps = 1/( cTime - pTime )
        pTime = cTime
        cv2.putText( img , f"Fps: {int(fps)}" , (6,50) , cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255 ) ,3 )
        if check and img2.size == 100*100 :

            img2 = np.array ( img2 )
            image = dataTransform ( img2 )
            image = image.reshape([1,1,26,26])
            outputs = model ( image )
            m = nn.Softmax(dim=1)
            output = m( outputs )
            _, predicted = torch.max( output.data ,1 )
            predict = labels[predicted]
            acc = int( output.data[0][predicted]*100)
        
            if acc > 75 :
                cv2.putText( img , f"Pytorch: {predict} {acc} %" , (200,50) , cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255 ) ,3 )
            else:
                cv2.putText( img ,     f"Pytorch:          %" , (200,50) , cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255 ) ,3 )
        else :
            cv2.putText( img ,         f"Pytorch:          %" , (200,50) , cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255 ) ,3 ) 

        cv2.imshow("video",img)

        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

