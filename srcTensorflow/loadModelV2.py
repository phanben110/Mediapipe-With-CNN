from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import json
import numpy as np
fileJson = "model1.json" 
fileWeight = "model1.h5" 
# 2, Khơi tạo dữ liệu để dự đoán
(xTrain, yTrain) , (xTest, yTest) = mnist.load_data()
xTest = xTest.reshape(xTest.shape[0], 28, 28, 1)
yTest = np_utils.to_categorical(yTest, 10)


# 3, mở file weights model
# with open('myDepGai.json' ,  'r') as f:

with open(fileJson  ,'r') as f:
	model_json = json.load(f)
	print( model_json )
	print("+++++++++++++++++++++++++++")
model = model_from_json(model_json)
model.load_weights(fileWeight)


# 4, dự doán dữ liệu


print(model.summary())
plt.imshow(xTest[1].reshape(28,28) , cmap = 'gray')
yPredict = model.predict(predict)
print("data  is")
print (predict)
print("gia tri du doan: ", np.argmax(yPredict))

# 5, vẽ đồ thị của hình ảnh
plt.imshow(predict , cmap = 'gray')
plt.show()

# 6, đánh giá dữ liệu xác suất
score = model.evaluate(xTest[0].reshape(28,28) )
print(score)


