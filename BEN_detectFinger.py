# this class to detect finger ussing handlamdmasks
import cv2
import mediapipe as mp
import time
import math
from typing import List, Tuple, Union
import numpy as np


class handLandmarks:
    def __init__(self, mode=False , maxHands = 2 , detectionCon = 0.65 , trackCon=0.65) :
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode , self.maxHands, self.detectionCon, self.trackCon )
        self.mpDraw = mp.solutions.drawing_utils
        self.pointStore = []
        self.statusFinger = [None,None,None,None,None]
        self.results = None

    # function to store value point x,y each finger
    def choosePoint ( self) : 
        rightFinger = []
        leftFinger = []
        twoPoint  = []
        if self.results.multi_hand_landmarks != None:
            for handLandmarks in self.results.multi_hand_landmarks:
                for point in self.mpHands.HandLandmark:
                    normalizedLandmark = handLandmarks.landmark[point]


                    twoPoint.append( normalizedLandmark )

        #print ( len ( twoPoint  ))
        if len( twoPoint ) ==21 : 
            return twoPoint 
        elif len( twoPoint ) == 42 :
            point1 = twoPoint[:21]
            point2 = twoPoint[21:]
            yList1 = []
            yList2 = []
            if point1 and point2 :

                for id, lm in enumerate (point1):
                    yList1.append(lm.y)
                for id, lm in enumerate ( point2):
                    yList2.append( lm.y  )
                y1 = max(yList1)
                y2 = max(yList2)

                if y1 > y2 :
                    return point2 
                else :
                    return point1 
        else :
            return False 



    def storePoint( self, img  , handNo= 0 , draw=True ) :
        xList = []
        yList = []
        box = []

        self.pointStore = [] 
        hand = self.choosePoint() 
        if hand : 
            for id, lm in enumerate (hand):
                h,w,c = img.shape
                cx,cy = int ( lm.x * w ) ,  int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.pointStore.append([id,cx,cy])
                if draw:
                    cv2.circle( img , (cx,cy), 2 , (200,0,255), cv2.FILLED )
                xMin, xMax = min(xList)  , max (xList)
                yMin, yMax = min(yList)  , max (yList)
                box = xMin, yMin, xMax , yMax

        return self.pointStore  , box



    def extractFeature(self , img , handNo = 0, draw=True ): 
        self.pointStore = ''
        xList = []
        yList = []
        box = []
 
        hand = self.choosePoint()
        if hand :
            for id, lm in enumerate (hand):
                h,w,c = img.shape
                cx1,cy1 = int ( lm.x * w ) ,  int(lm.y * h)
                xList.append(cx1)
                yList.append(cy1)

                cx,cy =  lm.x  ,  lm.y 
                cx = round(cx , 6 ) 
                cy = round(cy , 6 ) 
                self.pointStore = self.pointStore + " " +  str(cx) +" "+ str(cy)

                if draw:
                    cv2.circle( img , (cx1,cy1), 2 , (200,0,255), cv2.FILLED )


        n = len ( self.pointStore  )
        self.pointStore = self.pointStore [1:n-1] 

        return self.pointStore

        
    def drawAndResize(self , img , box , size = 120 , draw = True ):
        import numpy
        img2 = numpy.ones((img.shape[0], img.shape[1], 3), numpy.uint8) * 0
        img3 = numpy.ones((size, size, 3), numpy.uint8) * 0

        if len ( box ) != 0 :
            
            y = box[1]
            x = box[0]
            w = box[2] - box[0]
            h = box[3] - box[1]
            s =int ( w*h*0.0004)
            if s > 20 :
                s = 20
            elif s < 3 :
                s = 3

            print (f"s: {s}" ) 

            crop_img = img2[y-s:y + h + s , x-s:x + w + s ]
            #cv2.rectangle( img2 , ( box[0] - s , box[1] - s ) , ( box[2] + s , box[3]+ s  ) , (0,255,0),2 )
            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img2, handLms, self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=s-s),
                                               self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=s))
            # resize 130 x 130
            k=1
            if w > h : 
                k = w/100
            else :
                k = h/100

            w1 = int ( w/k ) 
            h1 = int ( h/k ) 

            
            
            dim = (w1 ,h1)
            #print (f" size {w}, {h} , dim {dim} ")
            try:

                crop_img  = cv2.cvtColor(crop_img , cv2.COLOR_BGR2GRAY)
                img3  = cv2.cvtColor(img3 , cv2.COLOR_BGR2GRAY)
                crop_img = cv2.resize(crop_img , dim, interpolation = cv2.INTER_AREA)
                x_offset=int ( (size - crop_img.shape[0] )/2) 
                y_offset=int ( (size - crop_img.shape[1] )/2)
                #print ( y_offset ) 
                img3[x_offset:crop_img.shape[0]+x_offset , y_offset:crop_img.shape[1]+y_offset] = crop_img
                #print ( "ok ") 

                return True , img2 , img3
            except :
                return False, False, False
        return False , False , False 

    # function to show finger in screen
    def showFinger( self, img , draw=True  ) :
        # import numpy
        imgRGB = cv2.cvtColor( img , cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB )
        #print( self.results.multi_hand_landmarks)
        # img2 = numpy.ones((img.shape[0] , img.shape[1],3  ) , numpy.uint8)*0

        # if self.results.multi_hand_landmarks:
        #     for handLms in self.results.multi_hand_landmarks:
        #         if draw:
        #             self.mpDraw.draw_landmarks( img2 , handLms, self.mpHands.HAND_CONNECTIONS ,self.mpDraw.DrawingSpec(color=(255,255,255), thickness=20), self.mpDraw.DrawingSpec(color=(255,255,255), thickness=20) )
        return img


    def distance(self, pointA , pointB ) :
        x1 = pointA[1]
        y1 = pointA[2]
        x2 = pointB[1]
        y2 = pointB[2]
        return math.hypot(x2-x1,y2-y1)

    def findFingerUp(self, pointStore ) :
        # set example finger 1, index (8,7,6,5,0). put A = 0 , B = 5, C = 6, D = 7, E = 8
        listFinger = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
        if len(pointStore) != 0 :

            for i in range (5) :
                if i == 0 :
                    # call A is point 3, B is point 5, C is point 9 ,
                    F35 = self.distance( pointStore[3], pointStore[5] )
                    F59 = self.distance( pointStore[5], pointStore[9] )
                    F45 = self.distance( pointStore[4] , pointStore[5] )

                    if F35 > F59 and  F45 > F59*1.5  :
                        self.statusFinger[i] = 1
                    else:
                        self.statusFinger[i] = 0

                else :
                    AE = self.distance( pointStore[listFinger[i][0]], pointStore[listFinger[i][4]] )
                    AB = self.distance( pointStore[listFinger[i][0]], pointStore[listFinger[i][1]] )
                    AC = self.distance( pointStore[listFinger[i][0]], pointStore[listFinger[i][2]] )
                    AD = self.distance( pointStore[listFinger[i][0]], pointStore[listFinger[i][3]] )

                    if AE > AD and AD > AC and AC > AB :
                        self.statusFinger[i] = 1
                    elif AE < AD or AE < AC :
                        self.statusFinger[i] = 0
                    else :
                        self.statusFinger[i] = 2

            #print ( f"status of finger  {self.statusFinger} " )
            total = sum(self.statusFinger)


            return self.statusFinger
        else :
            return 0
        #if ( len( pointStore ) != 0 ):
            #print ( pointStore )



if __name__ =="__main__" :

    import depthai as dai

    ben = handLandmarks()
    pTime = 0
    cTime = 0
    video ="handBen2.mp4"
    video = '/home/pcwork/ai/ftech/finger/DatasetToHandVitual/helloVideo/hello_01'

    #video = 0
    #video = 'rtsp://ftech:ad1235min@192.168.130.27/live0'
    cap = cv2.VideoCapture(video)
    while True :

        success, img = cap.read()
        ben.showFinger( img )
        pointList, box  = ben.storePoint ( img )
        check , img1 , img2 = ben.drawAndResize( img , box )
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
        if check  == True : 
            cv2.imshow("image", img2  )
            cv2.imshow("ben1", img1)
        if cv2.waitKey(1) == 27: 
            break 


    cv2.destroyAllWindows() 
    cap.release()
# vim file1 file2 file3
