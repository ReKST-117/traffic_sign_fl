# -*- coding: utf-8 -*-
"""
Created  on Mon  May  8 15:53:59 2023
Modified on Tue June 17 17:04:34 2025

@author: XXCH
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import cv2

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

data_path = "/home/nvidia/spd/js11"

tf.config.set_visible_devices([], 'GPU')

def show_list(list):
    print("Classes To Be Detected:")
    for i in range(len(list)):
        print(i+1,":",list[i])
    print("Total Number of Classes: ", len(list))
    return

images = []
labels = []

class_list = os.listdir(data_path)
show_list(class_list)

for i in range(len(class_list)):
    nth_class_list = os.listdir(data_path+'/'+class_list[i])
    for j in nth_class_list:
        image_data = cv2.imread(data_path+'/'+class_list[i]+'/'+j)
        images.append(image_data)
        labels.append(i)
        
images = np.array(images)
labels = np.array(labels)

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, train_size=0.75)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2)

def prepos(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img
 
X_train=np.array(list(map(prepos,X_train)))  
X_test=np.array(list(map(prepos,X_test)))
X_validation=np.array(list(map(prepos,X_validation)))

dim = 1

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],dim)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],dim)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],dim)

# Convert to categorical data
Y_train = to_categorical(Y_train,len(class_list))
Y_test = to_categorical(Y_test,len(class_list))
Y_validation = to_categorical(Y_validation,len(class_list))

# LeNet-5
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(6, [3,3], activation='relu', input_shape=(32,32,dim)),
#     tf.keras.layers.MaxPooling2D(2,2),
    
#     tf.keras.layers.Conv2D(16, [5,5], activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
    
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(120, activation='relu'),
#     tf.keras.layers.Dense(84, activation='relu'),
#     tf.keras.layers.Dense(len(class_list), activation='softmax')
#     ])
# model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
# model.summary()

# MiniVGGNet
model = tf.keras.models.Sequential([
     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.MaxPool2D(2,2),
     tf.keras.layers.Dropout(0.5),
    
     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.MaxPool2D(2,2),
     tf.keras.layers.Dropout(0.5),
    
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512, activation='relu'),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dropout(0.5)
     ])
# model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
# model.summary()

# OtherNet
# model = tf.keras.models.Sequential([
#    tf.keras.layers.Conv2D(60, (5,5), activation='relu', input_shape=(32,32,1)),
#    tf.keras.layers.Conv2D(60, (5,5), activation='relu'),
#    tf.keras.layers.MaxPool2D(2,2),
#    
#    tf.keras.layers.Conv2D(30, (3,3), activation='relu'),
#    tf.keras.layers.Conv2D(30, (3,3), activation='relu'),
#    tf.keras.layers.MaxPool2D(2,2),
#    
#    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dense(500, activation='relu'),
#    tf.keras.layers.Dropout(0.5),
#    tf.keras.layers.Dense(len(class_list), activation='softmax')
#    ])

model.load_weights("/home/nvidia/git/tsfl/pre_vgg.h5", by_name=True, skip_mismatch=True)

output = tf.keras.layers.Dense(len(class_list), activation='softmax')(model.output)
model = tf.keras.models.Model(inputs=model.input, outputs=output)


model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

dataGen = ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    )

TrainG = dataGen.flow(
    X_train, Y_train,
    batch_size = 32
    )

model_pred = model.fit(TrainG,
    steps_per_epoch=len(X_train)//32,
    epochs=20,
    validation_data=(X_validation, Y_validation),
    batch_size=32
    )

model.save('/home/nvidia/git/tsfl/model.h5')

if os.path.exists('/home/nvidia/git/tsfl/model.h5'):
    print("Model successfully saved.")
else:
    print("Model save failed.")

exit(0)
