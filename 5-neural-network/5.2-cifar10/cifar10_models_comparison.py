import tensorflow as tf
from tensorflow.keras import datasets, layers, models


"""
How to set up:
---
Run `pip3 install tensorflow` to install the required libraries.

How to run:
---
Run `python3 cifar10_models_comparison.py` to run the script.
The script contains two models.
The first model has 2 Conv2D layers with 64 and 128 filters and the second model has 3 Conv2D layers and 32, 32 and 64 filters.
Both models aim to classify images from the CIFAR-10 dataset.
The script will print the accuracy of both models.

Authors: Adam Łuszcz, Anna Rogala
"""

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

model1 = models.Sequential()
model1.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(128, (3, 3), activation='relu'))

model1.add(layers.Flatten())
model1.add(layers.Dense(128, activation='relu'))
model1.add(layers.Dense(10))

model1.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])
model1.fit(train_images, train_labels, epochs=10, 
          validation_data=(test_images, test_labels))

test_loss1, test_acc1 = model1.evaluate(test_images,  test_labels, verbose=2)
print('\nModel1 Test accuracy:', test_acc1)

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(32, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))

model2.add(layers.Flatten())
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(10))

model2.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])
model2.fit(train_images, train_labels, epochs=10, 
          validation_data=(test_images, test_labels))

test_loss2, test_acc2 = model2.evaluate(test_images,  test_labels, verbose=2)
print('\nModel2 Test accuracy:', test_acc2)
