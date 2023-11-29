import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_wine

class DataAnalyzer:
    def __init__(self, dataset_name, data_source, skip_first_column=False, is_sklearn_dataset=False):
        self.dataset_name = dataset_name
        if is_sklearn_dataset:
            self.features, self.labels = self.load_sklearn_dataset(data_source)
        else:
            self.features, self.labels = self.load_csv_dataset(data_source, skip_first_column)
        if self.features is not None:
            self.split_dataset()

    @staticmethod
    def load_csv_dataset(file_path, skip_first_column):
        """Load a dataset from a CSV file."""
        try:
            dataset = pd.read_csv(file_path)
            features = dataset.iloc[:, 1:] if skip_first_column else dataset.iloc[:, :-1]
            labels = dataset.iloc[:, -1]
            return features, labels
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None, None

    @staticmethod
    def load_sklearn_dataset(dataset_loader):
        """Load a dataset from sklearn.datasets."""
        dataset = dataset_loader()
        return dataset.data, dataset.target

    def split_dataset(self):
        """Split the dataset into training and test sets."""
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(
            self.features, self.labels, test_size=0.25, random_state=0, stratify=self.labels)


class ModelEvaluation:
    def __init__(self, model_name, model, data_analyzer):
        self.model_name = model_name
        self.accuracy, self.matrix = self.evaluate_model(model, data_analyzer)

    @staticmethod
    def evaluate_model(model, data_analyzer):
        """Fit and evaluate the model."""
        model.fit(data_analyzer.train_features, data_analyzer.train_labels)
        accuracy = model.score(data_analyzer.test_features, data_analyzer.test_labels)
        predictions = model.predict(data_analyzer.test_features)
        matrix = confusion_matrix(data_analyzer.test_labels, predictions)
        return accuracy, matrix

    def __str__(self):
        return (f"Model: {self.model_name}\nAccuracy: {self.accuracy}\n"
                f"Confusion Matrix:\n{self.matrix}\n")

if __name__ == "__main__":
    decision_tree_model = DecisionTreeClassifier(max_depth=4, random_state=1, class_weight="balanced")
    support_vector_machine = SVC(random_state=1, class_weight="balanced")

    sonar_dataset = DataAnalyzer("Sonar Dataset", 'sonar.all-data.txt')
    wine_dataset = DataAnalyzer("Wine Recognition Dataset", load_wine, is_sklearn_dataset=True)

    models = {"Decision Tree": decision_tree_model, "SVM": support_vector_machine}
    analyzed_datasets = [sonar_dataset, wine_dataset]

    for dataset in analyzed_datasets:
        if dataset.features is not None:
            print(f"Analyzing {dataset.dataset_name}\n")
            for model_name, model in models.items():
                evaluation = ModelEvaluation(model_name, model, dataset)
                print(evaluation)
