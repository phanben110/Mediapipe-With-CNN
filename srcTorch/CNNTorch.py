import torch 
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d , BatchNorm, Loss functions  
import torch.optim as optim # For all optimization algorithms , SGD , Adan, etc. 
import torch.nn.functional as F # All functions that don't have any parameters 
from torch.utils.data import DataLoader  # Gives easier dataset managment and creates mini batches  
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way 
import torchvision.transforms as transforms #Transformations we can perform on our dataset  

# Create Simple Fully Connected Neural Network

dataLoader = DataLoader( 
