import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.preprocessing import LabelEncoder


"""
How to set up:
---
Run `pip3 install tensorflow, sklearn` to install the required libraries.

How to run:
---
Run `python3 animal_model.py` to run the script.
The model has 3 Conv2D layers with 32, 64 and 64 filters.
It aims to classify images from the CIFAR-10 dataset where the dataset was filtered to contain only animals.
The script will print the accuracy of the model.

Authors: Adam Łuszcz, Anna Rogala
"""

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

animal_classes = [2, 3, 4, 5, 6, 7]

filtered_train_indexes = np.where(np.isin(train_labels, animal_classes))[0]
filtered_test_indexes = np.where(np.isin(test_labels, animal_classes))[0]

train_images = train_images[filtered_train_indexes]
train_labels = train_labels[filtered_train_indexes]
test_images = test_images[filtered_test_indexes]
test_labels = test_labels[filtered_test_indexes]

le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
test_labels = le.transform(test_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6))

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, 
          validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)