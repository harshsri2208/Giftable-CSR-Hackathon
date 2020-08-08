

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAvgPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

data_dir = '/kaggle/input/indian-coins-dataset/indian_coins_dataset'
num_classes = 4

batch_size = 32
epochs = 2
img_height = 240
img_width = 320

image_generator = ImageDataGenerator(rescale=1./255,
    zoom_range=0.3,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2)

train_generator = image_generator.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = image_generator.flow_from_directory(
    data_dir, 
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

Found 527 images belonging to 4 classes.
Found 129 images belonging to 4 classes.

total_train = 527
total_val = 129

model = Sequential()

# CNN network
model.add( Conv2D(16, 3, activation='relu', padding='same', input_shape=(img_height, img_width, 3)) )
model.add( MaxPooling2D(2) )

model.add( Conv2D(32, 3, activation='relu', padding='same') )
model.add( MaxPooling2D(2) )

model.add( Conv2D(64, 3, activation='relu', padding='same') )
model.add( MaxPooling2D(2) )

model.add( Conv2D(128, 3, activation='relu', padding='same') )
model.add( MaxPooling2D(2) )

model.add( Conv2D(256, 3, activation='relu', padding='same') )

# Transition between CNN and MLP
model.add( GlobalAvgPool2D() )

# MLP network
model.add( Dense(256, activation='relu') )

model.add( Dense(5, activation='softmax') )

model.summary()

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 240, 320, 16)      448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 120, 160, 16)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 120, 160, 32)      4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 60, 80, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 60, 80, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 30, 40, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 30, 40, 128)       73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 15, 20, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 15, 20, 256)       295168    
_________________________________________________________________
global_average_pooling2d (Gl (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               65792     
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 1285      
=================================================================
Total params: 459,685
Trainable params: 459,685
Non-trainable params: 0
_________________________________________________________________

#model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

model.load_weights('/kaggle/input/best-model/best.model')
for i in range(10):
    model.layers[i].trainable = False
    
ll = model.layers[10].output
ll = Dense(128, activation='relu')(ll)
ll = Dense(num_classes,activation="softmax")(ll)

new_model = tf.keras.Model(inputs=model.input,outputs=ll)

new_model.summary()

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_input (InputLayer)    [(None, 240, 320, 3)]     0         
_________________________________________________________________
conv2d (Conv2D)              (None, 240, 320, 16)      448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 120, 160, 16)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 120, 160, 32)      4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 60, 80, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 60, 80, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 30, 40, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 30, 40, 128)       73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 15, 20, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 15, 20, 256)       295168    
_________________________________________________________________
global_average_pooling2d (Gl (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               65792     
_________________________________________________________________
dense_2 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 516       
=================================================================
Total params: 491,812
Trainable params: 99,204
Non-trainable params: 392,608
_________________________________________________________________

new_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(patience=5, factor=0.1, verbose=True),
    ModelCheckpoint('best.model', save_best_only=True),
    EarlyStopping(patience=12)
]

history = new_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    batch_size=batch_size,
    callbacks=callbacks
)

Epoch 1/2
17/17 [==============================] - 253s 15s/step - loss: 2.0664 - accuracy: 0.2429 - val_loss: 1.5898 - val_accuracy: 0.2946 - lr: 0.0010
Epoch 2/2
17/17 [==============================] - 237s 14s/step - loss: 1.5759 - accuracy: 0.2713 - val_loss: 1.4642 - val_accuracy: 0.2946 - lr: 0.0010

#!mkdir -p saved_model
#new_model.save('saved_model/my_model') 

converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

open("coin_classifier.tflite", "wb").write(tflite_model)

1971800

 

