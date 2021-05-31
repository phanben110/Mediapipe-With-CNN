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
import itertools
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class imageDataset( Dataset): 
    def __init__(self, df , rootDir , transform= None ) :

        self.annotations = df  

        self.rootDir = rootDir  

        self.transform = transform  

    def __len__( self ): 
        return len ( self.annotations) 

    def __getitem__(self, index ): 

        imgPath = os.path.join(self.rootDir, self.annotations.iloc[index,0]) 
        image = cv2.imread( imgPath , 0 ) 
        yLabel = torch.tensor( int(self.annotations.iloc[index, 1] )) 

        if self.transform : 
            image = self.transform( image ) 
        return (image , yLabel ) 

class processingDataset () : 
    def __init__( self, rootDir ) : 
        self.rootDir = rootDir  

    def makeData (self , draw = True  ) : 
        categories = []
        trainImage = []
        label = [] 
        trainImgNames = os.listdir("./../data/raw")  
        for i in range ( len (trainImgNames)):
            fileImage = os.listdir(f"{self.rootDir}{trainImgNames[i]}")
            for img in fileImage :
                categories.append(i)
                label.append ( trainImgNames[i] ) 
                trainImage.append ( f"{trainImgNames[i]}/{img}")
        #print ( len (categories) )
        #print ( trainImage )
       
        df = pd.DataFrame({ 
            'fileName' : trainImage,  
            'category' : categories 
            }) 
        df.head() 
        if draw: 
            df1 = pd.DataFrame({ 
                'fileName' : trainImage, 
                'label'    : label 
                })  

            df1['label'].value_counts().plot.bar()
            plt.show () 

        return df

class confusionMatrix(): 
    def __init__(self) : 
        pass 
    def plot_confusion_matrix(self , cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues , draw = True  ):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
 
 
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
 
 
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
 
        if draw : 

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show ()

if __name__ == "__main__": 
    
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

    dataset = imageDataset( df = df , rootDir = rootDir , transform = dataTransform ) 
    lenDataset = len (dataset)  
    #split dataset to train and valid
    lenTrain = int (lenDataset*0.75) 
    lenValid = lenDataset - lenTrain 
    trainset , validset = torch.utils.data.random_split(dataset, [lenTrain,lenValid])
    trainLoader = DataLoader(dataset=trainset , batch_size=batchSize, shuffle=True) # shuffle is mean is mix label 
    validLoader = DataLoader(dataset=validset)
    
    for batch_idx, (data, targets) in enumerate(trainLoader):
    # get data to cuda
        data = data.to(device)
        targets = targets.to(device)
        print ( data.shape )
        print ( targets.shape ) 
        break
