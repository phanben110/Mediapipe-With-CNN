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
