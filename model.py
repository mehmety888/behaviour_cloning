# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 00:09:11 2019

@author: Mehmet
"""
import csv
import cv2
import numpy as np

# read and store the lines from driving log
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
def load_image(row):
    filename = row.split('\\')[-1]
    current_path = "data/IMG/"+filename
    #load image
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image
    
images = []
measurements = []
for line in lines: 
    # line[0]: center image path
    # line[1]: left image path
    # line[2]: right image path
    # line[3]: steering angle
    center_image = load_image(line[0])
    left_image   = load_image(line[1])
    right_image  = load_image(line[2]) 
    
    images.append(center_image)
    images.append(left_image)
    images.append(right_image)
    
    # correction factor
    # http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    correction = 0.2 
    center_measurement = float(line[3])
    left_measurement   = float(line[3]) + correction
    right_measurement  = float(line[3]) - correction
    
    measurements.append(center_measurement)
    measurements.append(left_measurement)
    measurements.append(right_measurement)
    
# flipping images and steering measurements
augmented_images,augmented_measurements = [],[]
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

# convert measurements to numpy arrays as keras requires 
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Conv2D

model = Sequential()
# image normalization and mean centering
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

#remove irrelevant parts of the image
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