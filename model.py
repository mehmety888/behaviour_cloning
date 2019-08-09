# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 00:09:11 2019

@author: Mehmet
"""
import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
lines = lines [1:]
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = "data/IMG/"+filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
    
augmented_images,augmented_measurements = [],[]
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Conv2D

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Conv2D(filters=24, kernel_size = (5,5), activation = "relu", subsample=(2, 2)))
model.add(Conv2D(filters=36, kernel_size = (5,5), activation = "relu", subsample=(2, 2)))
model.add(Conv2D(filters=48, kernel_size = (5,5), activation = "relu", subsample=(2, 2)))
model.add(Conv2D(filters=64, kernel_size = (3,3), activation = "relu"))
model.add(Conv2D(filters=64, kernel_size = (3,3), activation = "relu"))

model.add(Flatten())
model.add(Dense(100,activation = "relu"))
model.add(Dropout(rate = 0.3))
model.add(Dense(50,activation = "relu"))
model.add(Dense(10,activation = "relu"))
model.add(Dense(1))

model.compile(loss = "mse", optimizer = "adam")
model.fit(X_train,y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)

model.save("model.h5")