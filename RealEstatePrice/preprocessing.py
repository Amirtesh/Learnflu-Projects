import pandas as pd
from sklearn.utils import shuffle

class Preprocessor:

    def __init__(self,file_path):
        self.df=pd.read_csv(file_path)

    def augment_data(self):
        self.df=pd.concat([self.df]*4,ignore_index=True)
    
    def shuffle_data(self):
        self.df=shuffle(self.df)

    def get_processed_data(self):
        self.df=self.df.drop(['No','X1 transaction date'],axis=1)
        return self.df
    