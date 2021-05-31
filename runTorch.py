import torch 
from torch.utils.data import DataLoader  
from torch.utils.data import Dataset  
import torchvision.transforms as transforms 
import torch  
import cv2 
from srcTorch.BEN_processingData import imageDataset, processingDataset 
from srcTorch.BEN_modelCNN import CNN 
import BEN_detectFinger as finger 
import time 
ben = finger.handLandmarks()
pTime = 0
cTIme = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PATH = '/home/pcwork/ai/ftech/finger/CNN/modelPytorch/model.pt' 
numClasses = 7 
model = CNN (numClasses) 
model.load_state_dict(torch.load(PATH))
model.eval()
#print ( model )


def predict(model, image ) : 


    dataTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(26),
    transforms.ToTensor()
    ])
    
    image = dataTransform ( img )
    
    image = image.reshape([1,1,26,26])
    outputs = model ( image )  
    _, predicted = torch.max( outputs.data ,1 )  

video = 0 
cap = cv2.VideoCapture(video )  
predict = '' 
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
        predict   = predict ( img2 ) 
        print ( predict ) 
        #if predict != False :
        #    try:

        #        cv2.putText( img , f"Status: {predict} {int(acc*100)} %" , (200,50) , cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255 ) ,3 )
        #    except :

        #        cv2.putText( img , f"Status:          %" , (200,50) , cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255 ) ,3 )
        #else:
        #    cv2.putText( img ,     f"Status:          %" , (200,50) , cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255 ) ,3 )
    else :
        cv2.putText( img ,         f"Status:          %" , (200,50) , cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255 ) ,3 )
    cv2.imshow("image", img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()


