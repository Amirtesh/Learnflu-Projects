import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt

class Evaluator:

    def __init__(self,model,X_test,y_test,y_scaler):
        self.model=model
        self.X_test=X_test
        self.y_test=y_test
        self.y_scaler=y_scaler

    def evaluate(self):
        pred=self.model.predict(self.X_test)
        mae=mean_absolute_error(self.y_test,pred)
        mse=mean_squared_error(self.y_test,pred)
        rmse=np.sqrt(mse)
        r2=r2_score(self.y_test,pred)
        print('\n')

        print(f'Mean absolute error: {mae:.8f}')
        print(f'Mean square error: {mse:.8f}')
        print(f'Root mean squared error: {rmse:.8f}')
        print(f'R2 score: {r2:.8f}')

    def plot_predictions(self):
        predictions=self.model.predict(self.X_test)
        
        y_test_reshaped=self.y_test.reshape(-1, 1)
        y_actual_inv=self.y_scaler.inverse_transform(y_test_reshaped).ravel()
        predictions_reshaped=predictions.reshape(-1, 1)
        predictions_inv=self.y_scaler.inverse_transform(predictions_reshaped).ravel()

        jitter_strength=0.1

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_actual_inv)) + np.random.uniform(-jitter_strength, jitter_strength, size=len(y_actual_inv)),
                y_actual_inv, color="blue", label="Actual Values", alpha=0.5)
        plt.scatter(range(len(predictions_inv)) + np.random.uniform(-jitter_strength, jitter_strength, size=len(predictions_inv)),
                predictions_inv, color="red", label="Predicted Values", alpha=0.5)
        plt.title("Actual vs Predicted Values")
        plt.xlabel("Sample")
        plt.ylabel("Expense Value")
        plt.legend()
        plt.grid(True)
        plt.show()
    