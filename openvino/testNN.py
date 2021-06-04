#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import BEN_detectFinger as finger  
import time  
import torchvision.transforms as transforms

ben = finger.handLandmarks()  
pTime = 0  
cTime = 0  

labels  = ['Ok', 'Silent', 'Dislike', 'Like', 'Hi', 'Hello', 'Stop' , ' ' ]


#blobPath = '/home/pcwork/ai/ftech/finger/CNN/modelPytorch/openvino/model2.blob' 
blobPath = '/home/pcwork/ai/ftech/finger/CNN/modelTensorflow/modelOpenvino/modelHand.blob'  

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError("No source selected. Please use either \"-cam\" to use RGB camera as a source or \"-vid <path>\" to run on video")

debug = not args.no_debug
camera = not args.video

# Start defining a pipeline
pipeline = dai.Pipeline()

# NeuralNetwork
print("Creating Neural Network...")
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str(Path(blobPath).resolve().absolute()))
if camera:
    print("Creating Color Camera...") 


    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(480, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("rgb")
    cam_rgb.preview.link(cam_xout.input)

    print("Creating ImageManip node...")
    manip = pipeline.createImageManip()
    manip.initialConfig.setResize(26, 26)
    cam_rgb.preview.link(manip.inputImage)
    manip.out.link(detection_nn.input)

else:
    face_in = pipeline.createXLinkIn()
    face_in.setStreamName("in_nn")
    face_in.out.link(detection_nn.input)

# Create outputs
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

img = None
bboxes = []


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frame_norm(in_frame, bbox):
    norm_vals = np.full(len(bbox), in_frame.shape[0])
    norm_vals[::2] = in_frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def to_tensor_result(packet):
    return {
        tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        for tensor in packet.getRaw().tensors
    }


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    ben =  e_x / e_x.sum(axis=0)   
    print ( ben  ) 

    return e_x / e_x.sum(axis=0)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1)


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    if camera:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        #detection_in = device.getInputQueue("in_nn") 
    else:
        cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
        detection_in = device.getInputQueue("in_nn")
    q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=False)


    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame():
        if camera:
            in_rgb = q_rgb.get()
            new_frame = np.array(in_rgb.getData()).reshape((3, in_rgb.getHeight(), in_rgb.getWidth())).transpose(1, 2, 0).astype(np.uint8)
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            return True, np.ascontiguousarray(new_frame)
        else:
            return cap.read()

    result = None

    while should_run():
        read_correctly, img  = get_frame()

        if not read_correctly:
            break
        
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

            image = cv2.resize ( img2 , (26, 26))
            image = image.T
            image = np.reshape(image , [1,26,26,1])


            nn_data = dai.NNData()
            #nn_data.setLayer("input", to_planar(image, (26,26)))

            nn_data.setLayer("0", image )
            detection_in.send(nn_data)

            in_nn = q_nn.tryGet() 

            if in_nn is not None:
                data = softmax(in_nn.getFirstLayerFp16())
                result_conf = np.max(data)
                if result_conf > 0.1:
                    result = {
                        "name": labels[np.argmax(data)],
                        "conf": round(100 * result_conf, 2)
                    }
                else:
                    result = None

            if debug:
                # if the frame is available, draw bounding boxes on it and show the frame
                if result is not None:
                    cv2.putText(img, "{} ({}%)".format(result["name"], result["conf"]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                cv2.imshow("rgb", img)

                if cv2.waitKey(1) == ord('q'):
                    break
            elif result is not None:
                print("{} ({}%)".format(result["name"], result["conf"]))



