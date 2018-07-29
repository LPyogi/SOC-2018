import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential 
from keras.models import Model
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools 
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(28,28,3)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(1,1)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(1,1)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(1,1)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(1,1)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(1,1)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

train_path = 'C:/Users/dell/Desktop/digit/train'
valid_path = 'C:/Users/dell/Desktop/digit/validate'
test_path = 'C:/Users/dell/Desktop/digit/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(28,28), classes = ['0','1','2','3','4','5','6','7','8','9'], batch_size=100)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(28,28), classes = ['0','1','2','3','4','5','6','7','8','9'], batch_size=10)
#model = Sequential([Conv2D(64, (3,3), activation='relu', input_shape=(28,28,3)), Flatten(), Dense(10,activation='softmax')])
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=100, validation_data=valid_batches, validation_steps=10, epochs=1, verbose=1)
#vgg16_model = keras.applications.vgg16.vgg16.VGG16()
#vgg16_model.layers.pop()
##model=Sequential([Conv2D(64, (3,3), activation='relu', input_shape=(28,28,3))])
#model=Model(InputLayer(None,input_shape=(28,3,3)))
#mode=Sequential()
#count=0
#for layer in model.layers:
#    mode.add(layer)
#for layer in vgg16_model.layers:
#    if count==0:
#        count=count+1
#    mode.add(layer)
#for layer in model.layers:
#    layer.trainable=False
#mode.add(Dense(10,activation='softmax'))
#mode.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#mode.fit_generator(train_batches, steps_per_epoch=100, validation_data=valid_batches, validation_steps=10, epochs=1, verbose=1)
