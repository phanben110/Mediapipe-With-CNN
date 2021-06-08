import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
from processData import Dataset
from sklearn.model_selection import train_test_split
from modelCNN import CNN
import numpy as np
from PIL import Image
import cv2

# prepare dataset
dataset = Dataset()



def getTrainData(test_size, validation_size):
    # load data
    X_train, X_test, y_train, y_test = train_test_split(dataset.data['image'], dataset.data['labels'],
                                                        test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    # print(f"print {X_train.shape} ")
    # print(f"printd {y_train.shape}")
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def imshow(data):
    data.reshape(-1, 1, 26, 26)
    data = np.array(data, dtype=np.uint8)
    img = Image.fromarray(data)
    img.show()


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 7
batch_size = 64
learning_rate = 0.001

X_train, X_validation, X_test, y_train, y_validation, y_test = getTrainData(0.2, 0.1)

classes = dataset.data['mapping']
print(classes)
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

X_train = X_train.reshape((-1, 1, 26, 26))

X_train = torch.tensor(X_train)

img_test = X_test

X_test = X_test.reshape((-1, 1, 26, 26))

X_test = torch.tensor(X_test)
# print(type(X_train[0][0][0][0].item()))
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
n_total_steps = len(train_loader)
loss = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = torch.tensor(labels.to(device), dtype=torch.long)
        # print(labels)
        # Forward pass
        outputs = model(images)

        # print("Outputs",outputs)
        # print("Label",labels)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 64 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

imshow(img_test[0])
img = test_dataset[1500][0]
print("one",type(img))
print(classes[test_dataset[1500][1]])
img = img.reshape(-1, 1, 26, 26)
print("two",type(img))
print(img.shape)
with torch.no_grad():
    output = model.forward(img)
    predict = torch.max(output, 1)[1]
    print(classes[predict.item()])
PATH = 'model.pth'
torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)
