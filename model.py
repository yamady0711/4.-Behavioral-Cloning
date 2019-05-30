import os
import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

image_path = '/opt/carnd_p3/data/IMG/'
angle_adjustment = 0.2

samples = []
with open('/opt/carnd_p3/data/driving_log.csv', newline='') as csvfile:
    next(csvfile)  # to skip first row in csvfile
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print()
print('Train Sample Size : ', len(train_samples))
print('Validation Sample Size : ', len(validation_samples))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                #Center camera image = batch_sample[0]
                name = image_path + batch_sample[0].split('/')[-1]
                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)
                #flipped image
                images.append(cv2.flip(image, 1))
                angles.append(-angle)

                #Left camera image = batch_sample[1]
                name = image_path + batch_sample[1].split('/')[-1]
                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[3]) + angle_adjustment
                images.append(image)
                angles.append(angle)
                #flipped image
                images.append(cv2.flip(image, 1))
                angles.append(-angle)

                #Right camera image = batch_sample[2]
                name = image_path + batch_sample[2].split('/')[-1]
                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[3]) - angle_adjustment
                images.append(image)
                angles.append(angle)
                #flipped image
                images.append(cv2.flip(image, 1))
                angles.append(-angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))  # addtional layer for avoid overfitting
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print('Model Summary')
model.summary()

print('Model Compile')
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

print('Model Fit with Generator')
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples)/32, validation_data=validation_generator, nb_val_samples = len(validation_samples), nb_epoch=5, verbose=1)

print('Model saving')
model.save('model.h5')