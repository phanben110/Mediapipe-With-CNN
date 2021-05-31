import torch 
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d , BatchNorm, Loss functions  
import torch.optim as optim # For all optimization algorithms , SGD , Adan, etc. 
import torch.nn.functional as F # All functions that don't have any parameters 
from torch.utils.data import DataLoader  # Gives easier dataset managment and creates mini batches  
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way 
import torchvision.transforms as transforms #Transformations we can perform on our dataset  


# Create Simple Fully Connected Neural Network 

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Convolutional neural network (two convolutional layers) 

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # O = (26-5 + 2*2)/1 = 26 => ( 16, 28, 28 ) 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))   # O = 26/2 = 13 ( 16, 13,13 ) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # O = (13 - 5 +2*2)/1 +1 = 13 => (32,13,13) 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) #O = 13/2 = 6 =>(32,6,6)  
        self.fc = nn.Linear(6*6*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

