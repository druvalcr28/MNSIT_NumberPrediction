#!/usr/bin/env python
# coding: utf-8
import keras
import PIL
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

imgName=raw_input("enter name of image : ")
model = load_model('model_keras_mnist.h5')

##testIMG = Image.open('testIMG/'+str(imgName)+'.png').convert('LA')
testIMG = image.load_img(path="testIMG/"+str(imgName)+".png",color_mode="grayscale",target_size=(28,28,1))
print(np.array(testIMG).shape)
testIMG = testIMG.resize([28,28])

#converting to np array for prediction
img = np.array(testIMG)
print(img.shape)

#reshaping to 1D array
img = img.reshape(1,784)

result = model.predict(img)
print(result[0])
result = result[0]

index=0
for i in np.nditer(result):
	if(i):
		print("Prediction : "+str(index))
		break
	index += 1





