import BEN_detectFinger  as finger  
import os  
import numpy as np 
import cv2
import time 

finger = finger.handLandmarks() 
class extractFeature() : 
    def __init__(self)  : 
        self.pointList = ''
        self.points = '' 
    # this def to extract feature for each video  
    
    def extractFeature(self,video) : 
        cap = cv2.VideoCapture(video) 
        self.pointList=''
        pTime = 0 
        cTime = 0
        count = 0  

        while True : 
            success, img = cap.read() 
            if success != True : 
                print (f"This is the end video {video}.") 
                break  
            else: 
                finger.showFinger(img) 
                point = finger.extractFeature(img )
                

                #print ( point ) 
                print ( len ( point )) 
                count += 1 
                if len ( point ) > 10 : 
                    cTime = time.time()  
                    fps = 1/(cTime - pTime ) 
                    pTime = cTime 
                    self.pointList = self.pointList + ' ' + point 
                    cv2.putText( img ,"fps" +  str( int ( fps ) ) , (10,70) , cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255 ) ,2 )
                    cv2.imshow("image" , img ) 
                    cv2.waitKey(1)
                else : 
                    pass
        print (f"number of frame {count} ") 
        return self.pointList

    def extractFeatureToRun(self, frame, debug=False ) :
        self.points = '' 
        for i in range(len(frame)): 
            finger.showFinger(frame[i]) 
            point  = finger.extractFeature(frame[i]) 

            if len(point) > 10 :  
                self.points = self.points + ' ' + point 
            else : 
                self.points = ''
                #self.points.append( point ) 
        print ( "*************************************************") 
        #print ( self.points)  
        return self.points 

                
    
    # this def to save to file txt  
    def saveFileTxt(self, nameFile, data) : 
        file = open(nameFile, 'w' ) 
        #data = np.array( dataList )  
        #data = str ( data ) 
        n = len ( data )
        data = data[1:n-1] 
        file.write(data) 
        file.close()  

# this function to processing file 
if __name__ == "__main__" :
    video = 'handBen2.mp4'  
    extract = extractFeature()  
    #data = extract.extractFeature(video) 
    #print ( data ) 
    extract.saveFileTxt('demo.txt' , data ) 

