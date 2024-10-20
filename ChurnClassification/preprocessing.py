import pandas as pd
from sklearn.utils import shuffle

class Preprocessor:

    def __init__(self,file_path):
        self.df=pd.read_csv(file_path)
        print(f"Original dataset shape: {self.df.shape}")

    def shuffle_data(self):
        self.df=shuffle(self.df)

    def get_processed_data(self):
        return self.df