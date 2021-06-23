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

from sklearn.model_selection import train_test_split

from BEN_processingData import imageDataset 
from BEN_processingData import processingDataset
from BEN_processingData import confusionMatrix 

import BEN_modelCNN as model

import itertools
import numpy as np
import matplotlib.pyplot as plt

from processData import Dataset
# prepare dataset
dataset = Dataset()

def getTrainData(test_size):
    # load data
    X_train, X_test, y_train, y_test = train_test_split(dataset.data['image'], dataset.data['labels'],
                                                        test_size=test_size)
    #X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    print(f"print {y_train.shape} ")
    print(f"printd {y_test.shape}")
    return X_train, X_test, y_train, y_test

def imshow(data):
    data.reshape(-1, 1, 50, 50)
    data = np.array(data, dtype=np.uint8)
    img = Image.fromarray(data)
    img.show()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
sizeImg = 26 
numEpochs = 250
numClasses = 12
batchSize = 100
learningRate = 0.00001

X_train, X_test, y_train, y_test = getTrainData(0.3)

classes = dataset.data['mapping']


X_train = X_train.reshape((-1, 1, sizeImg, sizeImg))

X_train = torch.tensor(X_train)

img_test = X_test

X_test = X_test.reshape((-1, 1, sizeImg, sizeImg))

X_test = torch.tensor(X_test)
# print(type(X_train[0][0][0][0].item()))
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

trainLoader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=True)


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
torch.save(model.state_dict(), './../modelPytorch/model12ClassSize50.pt')
hour = ( time.time()  - beginTime ) / (60*60) 

print ("training Done , total time during training is: {:.3f} h ".format(hour) )


from sklearn.metrics import confusion_matrix

nb_classes = 12

# Initialize the prediction and label lists(tensors)
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

# that is print confusion matrix 

with torch.no_grad():
    for i, (inputs, classes) in enumerate(testLoader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Append batch prediction results
        predlist=torch.cat([predlist,preds.view(-1).cpu()])
        lbllist=torch.cat([lbllist,classes.view(-1).cpu()])



# Confusion matrix
conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
print(conf_mat)

# Per-class accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print(class_accuracy)

plt.figure(figsize=(12,12))

#plot_confusion_matrix(conf_mat , testLoader.)

conMax = confusionMatrix() 
#conMax.plot_confusion_matrix(conf_mat , ['Ok', 'Silent', 'Dislike', 'Like', 'Hi', 'Hello', 'Stop'])
#conMax.plot_confusion_matrix(conf_mat ,['3', '9', '5', '0', '11', '10', '2', '1', '7', '6', '4', '8']) 
conMax.plot_confusion_matrix(conf_mat , ['Three','Stop','Hello','Zero','Tym','Nothing','Two','One','Dislike', 'Like' ,'Four','Ok']) 
