import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd


"""
How to set up:
---
Run `pip3 install tensorflow, sklearn, pandas` to install the required libraries.

How to run:
---
Run `python3 phones_price_model.py` to run the script.
The model has 3 Conv2D layers with 32, 64 and 64 filters.
It aims to classify phones price ranges from the phones_price.csv dataset.
The script will print the accuracy of the model.

Authors: Adam Łuszcz, Anna Rogala
"""

data = pd.read_csv('phones_price.csv')

features = data.drop('price_range', axis=1)
labels = data['price_range']

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

input_size = len(data.columns) - 1

model = Sequential()
model.add(Dense(32, input_dim=input_size, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_features, train_labels, epochs=250, batch_size=32)

loss, accuracy = model.evaluate(test_features, test_labels)

print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")