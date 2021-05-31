import cv2
import numpy as np
import math
import BEN_detectFinger as Finger
 
 
ben = Finger.handLandmarks()
cap = cv2.VideoCapture(0)
 
while (True):
    ret, frame = cap.read()
    # print(ret)
    frame = cv2.flip(frame, 1)
    newFrame = cv2.resize(frame, (1500, 800))
 
    cv2.rectangle(newFrame, (900, 100), (1400, 600), (0, 255, 0), 0)
    roi = newFrame[100:600, 900:1400]
 
    gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    kernel = np.ones((3, 3), np.uint8)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
 
    # define range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
 
    # extract skin colur imagw
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
 
    # extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask, kernel, iterations=4)
 
    # blur the image
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n_contours = len(contours)
    if n_contours>0 :
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
 
        cv2.drawContours(roi, cnt, -1, (252, 3, 49), thickness=5)
        # approx the contour a little
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
 
        # make convex hull around hand
        hull = cv2.convexHull(cnt)
        # define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
 
        # print("hull", areahull)
        # print("contour", areacnt)
        arearatio = ((areahull - areacnt) / areacnt) * 100
        #print("Rate not cover",arearatio)
 
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        cnt_defects=0
 
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            # print("start",start)
            # print("end",end)
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
            # distance between point and convex hull
            d = (2 * ar) / a
            # print(d)
            # print(d)
            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
 
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 10:
                cnt_defects += 1
                cv2.circle(roi, far, 4, color=4,thickness= -2)
            # draw lines around hand
            cv2.line(roi, start, end, [0, 255, 0], 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if(cnt_defects == 4):
            cv2.putText(newFrame, 'Hello', (0, 50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        elif cnt_defects == 1:
            if(arearatio>30):
                cv2.putText(newFrame, 'Hi', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                ben.showFinger(roi, draw=False)
                pointList, box = ben.storePoint(roi, draw=False)
                if len(pointList) > 0:
                    x1, y1 = pointList[4][1], pointList[4][2]
                    x2, y2 = pointList[8][1], pointList[8][2]
                    if x1-x2>100:
                        cv2.putText(newFrame, 'Love', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(newFrame, 'Dislike', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif cnt_defects == 0:
            ben.showFinger(roi, draw=False)
            pointList, box = ben.storePoint(roi, draw=False)
            if len(pointList) > 0 :
                x1, y1 = pointList[4][1], pointList[4][2]
                x2, y2 = pointList[8][1], pointList[8][2]
                # print(x1,x2)
                if(x1>x2):
                    cv2.putText(newFrame, 'like', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif cnt_defects == 3 :
            cv2.putText(newFrame, 'ok', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
 
    else:
        cv2.putText(newFrame, 'No object detection', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
 
    cv2.imshow("frame", newFrame)
    cv2.waitKey(1)
 
cv2.destroyAllWindows()
cap.release()
