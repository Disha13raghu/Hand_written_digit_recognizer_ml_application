import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



mnist= tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test)  =mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model= tf.keras.models.Sequential()


model.add(tf.keras.layers.Conv2D(32, (3,3), activation='sigmoid', input_shape=(28, 28, 1))) #
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(48, (3,3), activation='sigmoid'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(500, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))


model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3)

model.save('digits.keras')


#