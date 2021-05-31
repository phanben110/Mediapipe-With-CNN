import os
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import torch
import cv2
import pandas as pd
import time  

from BEN_processingData import imageDataset 
from BEN_processingData import processingDataset 

import BEN_modelCNN as model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

rootDir = './../data/raw/'

# hyper parameters
inChannel = 1
numClasses = 7
learningRate = 0.001
batchSize = 100
numEpochs = 10

dataTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(26),
    transforms.ToTensor()
    ])


data = processingDataset( rootDir )
df = data.makeData ( draw = False  )

#Load data

dataset =imageDataset( df = df , rootDir = rootDir , transform = dataTransform )
lenDataset = len (dataset)
#split dataset to train and valid
lenTrain = int (lenDataset*0.75)
lenValid = lenDataset - lenTrain
trainset , validset = torch.utils.data.random_split(dataset, [lenTrain,lenValid])
trainLoader = DataLoader(dataset=trainset , batch_size=batchSize, shuffle=True) # shuffle is mean is mix label
testLoader = DataLoader(dataset=validset)

model = model.CNN( numClasses ).to(device) 

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

# Train the model
totalStep = len(trainLoader)
Ptime = time.time() 
beginTime = time.time() 
for epoch in range(numEpochs):
    for i, (images, labels) in enumerate(trainLoader):
        images = images.to(device)
        labels = labels.to(device)
        #print ( labels.shape)

        # Forward pass
        #print ( f"shape image {images.shape} and label shape {labels.shape} ")
        #print ( labels )
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sec = (time.time() - Ptime) 
        if (i+1) % 100 == 0:

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time: {:.2f} s'
                   .format(epoch+1, numEpochs, i+1, totalStep, loss.item(), sec))
            Ptime = time.time() 


# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testLoader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item() 


    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), './../modelPytorch/model.pt')
hour = ( time.time()  - beginTime ) / (60*60) 

print ("training Done , total time during training is: {:.3f} h ".format(hour) ) 
