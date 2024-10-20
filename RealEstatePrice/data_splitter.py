import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Splitter:

    def __init__(self,df:pd.DataFrame,target,test_size=0.2,random_state=101):
        self.df=df
        self.target=target
        self.test_size=test_size
        self.random_state=random_state
        self.scaler_x=StandardScaler()
        self.scaler_y=StandardScaler()
    
    def split_data(self):
        X=self.df.drop(self.target,axis=1)
        y=self.df[self.target]
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X,y,test_size=self.test_size,random_state=self.random_state)

    def get_splits(self,scale=False):
        if scale:
            self.X_train=self.scaler_x.fit_transform(self.X_train)
            self.X_test=self.scaler_x.transform(self.X_test)
            self.y_train=self.scaler_y.fit_transform(self.y_train.values.reshape(-1,1))
            self.y_test=self.scaler_y.transform(self.y_test.values.reshape(-1,1))

        return self.X_train,self.X_test,self.y_train,self.y_test