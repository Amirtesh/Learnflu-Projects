import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:

    def __init__(self,df:pd.DataFrame):
        self.df=df
    
    def display_basic_info(self):
        print("DataFrame Shape:", self.df.shape)
        print("\nFirst 5 Rows:")
        print(self.df.head())
        print("\nData Types:")
        print(self.df.dtypes)
        print("\nMissing Values:")
        print(self.df.isnull().sum())
    
    def display_statistics(self):
        print("\nStatistical Summary:")
        print(self.df.describe())
    
    def plot_distribution(self, column):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()
    
    def plot_correlation_heatmap(self):
        plt.figure(figsize=(12, 8))
        correlation = self.df.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap')
        plt.show()

    def plot_churn_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Churn', data=self.df)
        plt.title('Churn Distribution')
        plt.xlabel('Churn')
        plt.ylabel('Count')
        plt.grid()
        plt.show()