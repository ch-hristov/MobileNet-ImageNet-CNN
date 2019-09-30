from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras.applications import MobileNet 
from keras import backend as K
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
import time
import os
import pdf2image
from PIL import Image

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

base_model = MobileNet(weights='imagenet',include_top=False) #imports the VGG16yyyyyyy model and discards the last 1000 neuron layer.
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and
									 #classify for better results.

x = Dense(1024,activation='relu')(x) #dense layer 2
x = Dense(512,activation='relu')(x) #dense layer 3

preds = Dense(10,activation='softmax')(x) #final layer with softmax activation
model = Model(inputs=base_model.input,outputs=preds)

for layer in model.layers:
	layer.trainable = False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
	layer.trainable = False
for layer in model.layers[20:]:
	layer.trainable = True

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
train_generator = train_datagen.flow_from_directory('C:\images',
												 target_size=(224,224),
												 color_mode='rgb',
												 batch_size=10,
												 class_mode='categorical',
												 shuffle=True)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train = train_generator.n // train_generator.batch_size
model.fit_generator(generator=train_generator,
				   steps_per_epoch=step_size_train,
				   epochs=10)


from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')