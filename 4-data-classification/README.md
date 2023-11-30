# Data classificator

The program is designed to analyze datasets using decision tree and SVM (Support Vector Machine) models.  
It uses the sklearn library to load the datasets and train the models.

The program is designed to load, preprocess and split datasets into training and test sets.  
It also fits the provided model to the training data, evaluates it on the test data  
and computes the model's accuracy and confusion matrix.

It supports datasets loaded from CSV files or directly from sklearn's dataset loading functions.

It uses the following datasets:

- Sonar Dataset: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks) (read from a [txt file](sonar.all-data.txt) )
- Wine Recognition Dataset: https://archive.ics.uci.edu/ml/datasets/Wine

The program prints the results to the console.

## How to set up:

Install the packages from the requirements.txt with the following command `pip3 install -r requirements.txt`

## How to run:

Run the program with the following command `python3 data_classificator.py`  
The program will print the results to the console.

## Demo:
https://github.com/annarogala/ai-projects/assets/13242654/f592c84a-373d-46d6-9833-c2a946733861

## Authors:

Adam Łuszcz s22994  
Anna Rogala s21487

## Sources:

- https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
- https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
- https://archive.ics.uci.edu/ml/datasets/Wine
