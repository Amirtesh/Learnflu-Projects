import pandas as pd
from sklearn.utils import shuffle

class Preprocessor:

    def __init__(self,file_path):
        self.df=pd.read_csv(file_path)

    def augment_data(self):
        self.df=pd.concat([self.df]*4,ignore_index=True)

    def shuffle_data(self):
        self.df=shuffle(self.df)
    
    def encode_data(self):
        encoded_features=pd.get_dummies(self.df.drop('expenses', axis=1), drop_first=True, dtype=int)
        self.df_encoded=pd.concat([encoded_features, self.df['expenses']], axis=1)

    def get_processed_data(self):
        return self.df_encoded