import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


"""
How to set up:
---
Run `pip3 install tensorflow, pandas, sklearn, seaborn, matplotlib` to install the required libraries.

How to run:
---
Run `python3 sonar_model.py` to run the script.
The model will be trained on the sonar dataset.
Its aim is to classify sonar signals as either rocks or mines.
The script will print the accuracy of the model and show the confusion matrix.

Authors: Adam Łuszcz, Anna Rogala
"""

data = pd.read_csv('sonar.all-data.txt', header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]
output_dim = 1

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, input_dim=input_dim, activation='relu'),
    tf.keras.layers.Dense(output_dim, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()