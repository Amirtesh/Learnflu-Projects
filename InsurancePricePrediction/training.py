import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,VotingRegressor
from xgboost import XGBRegressor

class Trainer:

    def __init__(self,X_train,y_train,X_test,y_test,y_scaler):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.y_scaler=y_scaler

    def evaluate(self,model,X,y):
        pred=model.predict(X)
        mae=mean_absolute_error(y,pred)
        mse=mean_squared_error(y,pred)
        rmse=np.sqrt(mse)
        r2=r2_score(y,pred)

        print(f'Mean absolute error: {mae:.8f}')
        print(f'Mean square error: {mse:.8f}')
        print(f'Root mean squared error: {rmse:.8f}')
        print(f'R2 score: {r2:.8f}')

    def make_df(self,model, X, y):
        pred = model.predict(X)
        y = y.ravel()
        y_reshaped = y.reshape(-1, 1)
        y_actual_inv = self.y_scaler.inverse_transform(y_reshaped).ravel()
        pred = pred.ravel()
        pred_inv=self.y_scaler.inverse_transform(pred.reshape(-1,1)).ravel()
        return pd.DataFrame({'Actual': y_actual_inv, 'Predicted': pred_inv})        
    
    def train_linear_regression(self):
        lin_model=LinearRegression(n_jobs=-1)
        lin_model.fit(self.X_train,self.y_train)
        print(f'Linear regression results on training data: ')
        self.evaluate(lin_model,self.X_train,self.y_train)
        print(f'\nLinear regression results on testing data: ')
        self.evaluate(lin_model,self.X_test,self.y_test)
        return lin_model
    
    def train_knn(self,n_neighbors=2):
        knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
        knn_model.fit(self.X_train, self.y_train)
        print(f'\nKNN results on training data :')
        self.evaluate(knn_model, self.X_train, self.y_train)
        print(f'\nKNN results on testing data: ')
        self.evaluate(knn_model, self.X_test, self.y_test)
        return knn_model

    def train_random_forest(self, n_estimators=300, max_depth=10):
        rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_depth=max_depth)
        rf_model.fit(self.X_train, self.y_train)
        print(f'\nRandom forest results on training data: ')
        self.evaluate(rf_model, self.X_train, self.y_train)
        print(f'\nRandom forest results on testing data: ')
        self.evaluate(rf_model, self.X_test, self.y_test)
        return rf_model

    def train_xgboost(self, n_estimators=1000, learning_rate=0.01, max_depth=10, gamma=0.1):
        xgb_model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, gamma=gamma, n_jobs=-1)
        xgb_model.fit(self.X_train, self.y_train)
        print(f'\nXGBoost results on training data: ')
        self.evaluate(xgb_model, self.X_train, self.y_train)
        print(f'\nXGBoost results on testing data: ')
        self.evaluate(xgb_model, self.X_test, self.y_test)
        return xgb_model

    def train_voting(self, knn_model, rf_model, xgb_model):
        vote_model = VotingRegressor(estimators=[('knn', knn_model), ('rf', rf_model), ('xgb', xgb_model)], n_jobs=-1,weights=[1,1,1])
        vote_model.fit(self.X_train, self.y_train)
        print(f'\nVoting regressor results on training data: ')
        self.evaluate(vote_model, self.X_train, self.y_train)
        print(f'\nVoting regressor results on testing data: ')
        self.evaluate(vote_model, self.X_test, self.y_test)
        return vote_model
    
