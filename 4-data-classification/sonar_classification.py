from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

INPUT_FILE = 'sonar.all-data.txt'

df = pd.read_csv(INPUT_FILE, delimiter=',')
data = df.to_numpy()

X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

params = {'random_state': 0, 'max_depth': 8}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")