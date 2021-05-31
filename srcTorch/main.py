import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np 
import json
from torch.utils.data import DataLoader 

DATA_PATH = ".././data/processed/dataHandSize26.json"

def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["image"])
    y = np.array(data["labels"])

    print (X.shape)
    print (y.shape)
    return X, y
def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    print ( f"print {X_train.shape} " )
    print ( f"printd {y_train.shape}") 
    # add an axis to input sets
    #X_train = X_train[..., np.newaxis]
    #X_validation = X_validation[..., np.newaxis]
    #X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
#train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=True, 
#                                           transform=transforms.ToTensor(),
#                                           download=True)
#
#test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                          train=False, 
#                                          transform=transforms.ToTensor())
#
## Data loader
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                           batch_size=batch_size, 
#                                           shuffle=True)
#
#test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                          batch_size=batch_size, 
#                                          shuffle=False)
# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)
# get train, validation, test splits
X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.2, 0.1)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

X_train = torch.Tensor ( X_train ) 
y_train = torch.Tensor ( y_train )  

X_test = torch.Tensor ( X_test ) 
y_test = torch.Tensor ( y_test )
datasetTrain = torch.TensorDataset( X_train , y_train ) 
#trainDataLoader = DataLoader(X_train, batch_size = 1 , shuffle = True )  
#print ( trainDataLoader)

for data, label in enumerate( datasetTrain  ): 
    print ( data.size )  
    print ( label.size  ) 

## Train the model
#total_step = len(X_train )
#for epoch in range(num_epochs):
#    for i in range ( len ( X_train ) ) :
#        images = X_train[i]
#        labels = y_train[i]
#        print ( labels.shape) 
#        print ( images.shape )   
#        # Forward pass
#        outputs = model(images)
#        loss = criterion(outputs, labels)
#        
#        # Backward and optimize
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        
#        if (i+1) % 100 == 0:
#            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
#
## Test the model
#model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
#with torch.no_grad():
#    correct = 0
#    total = 0
#
#    for i  in range ( len ( X_test ) ) :
#        images = X_test[i].to(device)
#        labels = y_test[i].to(device)
#        outputs = model(images)
#        _, predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted == labels).sum().item()
#
#    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
#
## Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')
