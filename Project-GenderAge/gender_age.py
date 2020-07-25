# Import required modules
import cv2 as cv
import math 
import time
import argparse #for command line arguments

def getFaceBox(net, frame, conf_threshold=0.7): #threshold is for min limit for detectedface
    frameOpencvDnn = frame.copy() #dnn is for using the pretrained model
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward() #for detecting the face 
    bboxes = [] #boundary boxes
    for i in range(detections.shape[2]): #for loop for detection the face and to know confidence
        confidence = detections[0, 0, i, 2]#Extracing the confidence
        if confidence > conf_threshold: #if conf >0.7 then execute if block
            x1 = int(detections[0, 0, i, 3] * frameWidth)#Start X
            y1 = int(detections[0, 0, i, 4] * frameHeight)#Start Y
            x2 = int(detections[0, 0, i, 5] * frameWidth)#End X
            y2 = int(detections[0, 0, i, 6] * frameHeight)#End Y
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)  #for drawing rectangle box
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='To get input from user')
parser.add_argument("-i", help='Path to input image or video file.We can skip this argument to capture frames from a camera live feed.')

args = parser.parse_args()

#Network Archietecure
faceProto = "opencv_face_detector.pbtxt"#Holds the network of nodes each represents one operation,
#connected to each other as inputs and outputs.
faceModel = "opencv_face_detector_uint8.pb"#Stores actual tensorflow program

ageProto = "age_deploy.prototxt"# used to describe the structure of the data to be serialized.
ageModel = "age_net.caffemodel"# This contains the information of the trained neural network.
# definez the internal states of the parameters/gradients of the layers.


genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)# Mean "RGB" values
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Loading and reading the network

ageNet = cv.dnn.readNetFromCaffe(ageProto,ageModel)#Reads a network model stored in Caffe framework's format.
genderNet = cv.dnn.readNetFromCaffe(genderProto,genderModel)
faceNet = cv.dnn.readNet(faceModel,faceProto)#Reading deep learning network 

# Open a video file or an image file or a camera stream
cap = cv.VideoCapture(args.i if args.i else 0)
padding = 20
while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]# extract the ROI of the face and then construct a blob from ROI

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        print("Gender : {}, confidence = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        print("Age : {}, confidence = {:.3f}".format(age, agePreds[0].max()))

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0]-5, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2, cv.LINE_AA)
        cv.imshow("Age Gender Demo", frameFace)
        name = args.i
        cv.imwrite('./detected faces/'+name,frameFace)
     
    print("Time : {:.3f}".format(time.time() - t))
