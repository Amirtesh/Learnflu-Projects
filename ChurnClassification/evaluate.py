import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

class Evaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, pred)
        precision = precision_score(self.y_test, pred)
        recall = recall_score(self.y_test, pred)
        f1 = f1_score(self.y_test, pred)
        y_scores = self.model.predict_proba(self.X_test)[:, 1]
        roc_auc = auc(*roc_curve(self.y_test, y_scores)[:2])
        model_name = self.model.__class__.__name__
        print(f'{"=" * 40}')
        print(f'Evaluating Model: {model_name}')
        print(f'{"=" * 40}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')
        print(f'{"=" * 40}')

    def plot_confusion_matrix(self):
        pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

    def plot_roc_curve(self):
        y_scores = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC Curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
        
    def predict_single(self, row_idx):
        """Predict for a single test data row."""
        # Check if X_test is a NumPy array or a pandas DataFrame
        if isinstance(self.X_test, np.ndarray):
            row = self.X_test[row_idx].reshape(1, -1)  # Use normal indexing for NumPy arrays
        else:
            row = self.X_test.iloc[row_idx].values.reshape(1, -1)  # Use .iloc for pandas DataFrame

        # Similarly, check for y_test
        if isinstance(self.y_test, np.ndarray):
            actual = self.y_test[row_idx]  # Normal indexing for NumPy array
        else:
            actual = self.y_test.iloc[row_idx]  # .iloc for pandas Series

        predicted = self.model.predict(row)[0]
        predicted_prob = self.model.predict_proba(row)[0, 1]

        print(f'Predicted: {predicted} (Probability: {predicted_prob:.4f}), Actual: {actual}')


    def predict_range(self, start_idx, end_idx):
        """Predict for a range of test data rows without probabilities."""
        # Create lists to store actual and predicted values
        actual_values = []
        predicted_values = []
        
        # Loop over the specified range
        for i in range(start_idx, end_idx + 1):
            # Check if X_test is a NumPy array or a pandas DataFrame
            if isinstance(self.X_test, np.ndarray):
                row = self.X_test[i].reshape(1, -1)  # Use normal indexing for NumPy arrays
            else:
                row = self.X_test.iloc[i].values.reshape(1, -1)  # Use .iloc for pandas DataFrame

            # Similarly, check for y_test
            if isinstance(self.y_test, np.ndarray):
                actual = self.y_test[i]  # Normal indexing for NumPy array
            else:
                actual = self.y_test.iloc[i]  # .iloc for pandas Series
            
            predicted = self.model.predict(row)[0]

            # Store actual and predicted values
            actual_values.append(actual)
            predicted_values.append(predicted)

        # Create a pandas DataFrame to display the results in tabular format
        results_df = pd.DataFrame({
            'Index': range(start_idx, end_idx + 1),
            'Actual': actual_values,
            'Predicted': predicted_values
        })
        
        # Print the DataFrame
        print(results_df.to_string(index=False))