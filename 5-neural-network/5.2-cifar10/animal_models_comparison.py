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
Run `python3 animal_models_comparison.py` to run the script.
The script contains two models.
First model has 2 Conv2D layers with 64 and 128 filters and the second has 3 Conv2D layers and 32, 32 and 64 filters.
Both models aim to classify images from the CIFAR-10 dataset where the dataset was filtered to contain only animals.
The script will print the accuracy of both models.

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

model_1 = models.Sequential()
model_1.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_1.add(layers.MaxPooling2D((2, 2)))
model_1.add(layers.Conv2D(128, (3, 3), activation='relu'))

model_1.add(layers.Flatten())
model_1.add(layers.Dense(128, activation='relu'))
model_1.add(layers.Dense(6))

model_2 = models.Sequential()
model_2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_2.add(layers.MaxPooling2D((2, 2)))
model_2.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_2.add(layers.MaxPooling2D((2, 2)))
model_2.add(layers.Conv2D(64, (3, 3), activation='relu'))

model_2.add(layers.Flatten())
model_2.add(layers.Dense(64, activation='relu'))
model_2.add(layers.Dense(6))

model_1.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

model_1.fit(train_images, train_labels, epochs=10, 
            validation_data=(test_images, test_labels))

test_loss, test_acc = model_1.evaluate(test_images,  test_labels, verbose=2)
print('\nModel 1 Test accuracy:', test_acc)

model_2.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

model_2.fit(train_images, train_labels, epochs=10, 
          validation_data=(test_images, test_labels))

test_loss, test_acc = model_2.evaluate(test_images,  test_labels, verbose=2)
print('\nModel 2 Test accuracy:', test_acc)