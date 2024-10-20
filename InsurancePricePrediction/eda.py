import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:

    def __init__(self,df:pd.DataFrame):
        self.df=df
    
    def summary(self):
        print('Dataset summary: ')
        print(self.df.describe())
        print('\nMissing values: ')
        print(self.df.isnull().sum())
        print('\nData types: ')
        print(self.df.dtypes)

    def correlation_matrix(self):
        plt.figure(figsize=(10,6))
        sns.heatmap(self.df.corr(),annot=True,cmap='coolwarm',fmt='.2f')
        plt.title('Correlation matrix')
        plt.show()

    def plot_distributions(self):
        self.df.hist(bins=20,figsize=(20,15))
        plt.suptitle('Feature distribution')
        plt.show()
    
    def plot_boxplots(self):
        numerical_cols=self.df.select_dtypes(include=['int64','float64']).columns
        for col in numerical_cols:
            plt.figure(figsize=(8,4))
            sns.boxplot(data=self.df,x=col)
            plt.title(f'Boxplot for {col}')
            plt.show()
        