#!/usr/bin/env python
# coding: utf-8
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


(x_train,y_train),(x_test,y_test)=mnist.load_data()

print("x_train.shape : ",x_train.shape)

#changing to input vector
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

#normalize
x_train /= 255
x_test /= 255

print("x_train.shape : ",x_train.shape)

n_classes=10

Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

print("y_train.shape : ",y_train.shape)
print("Y_train.shape : ",Y_train.shape)

#defining model
model=Sequential()
model.add(Dense(256, input_shape=(784,)))
model.add(Activation('relu'))                            
model.add(Dropout(0.2))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

#compiling sequential model
model.compile(loss=keras.losses.categorical_crossentropy,
	metrics=['accuracy'],
	optimizer=keras.optimizers.Adadelta())

model.fit(x_train,Y_train,
	batch_size=128,
	epochs=2,
	verbose=1,
	validation_data=(x_test,Y_test))
model.save('model_keras_mnist.h5')

#find accuracy and loss
model_NumImgReg = load_model('model_keras_mnist.h5')

result = model_NumImgReg.evaluate(x_test,Y_test,verbose=1)

print("Accuracy : ",result[1])
print("Loss : ",result[0])

print("Accuracy Percentage : ",(result[1]*100))


