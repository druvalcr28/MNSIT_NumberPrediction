#!/usr/bin/env python
# coding: utf-8
import os,sys
import keras
import PIL
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

model = load_model('model_keras_mnist.h5')

dir = raw_input("enter the directory name : ")
unique = np.zeros((10,),dtype=np.int)

for f in os.listdir(dir):
	#print(f)
	imgPath = os.path.join(dir,f)
	testIMG = image.load_img(path=imgPath,color_mode="grayscale",target_size=(28,28,1))
	testIMG = testIMG.resize([28,28])
	
	#converting to np array for prediction
	img = np.array(testIMG)
	#reshaping to 1D array
	img = img.reshape(1,784)

	result = model.predict(img)
	result = result[0]

	#extract the non-zero position
	pos = np.nonzero(result)[0][0]
	unique[pos] = 1

print("Unique Numbers Found : "+str(np.nonzero(unique)[0]))





