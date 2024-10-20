import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

class Trainer:

    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test

    def evaluate(self,model,X,y):
        pred=model.predict(X)
        y_scores=model.predict_proba(X)[:,1]

        accuracy=accuracy_score(y,pred)
        precision=precision_score(y,pred)
        recall=recall_score(y,pred)
        f1=f1_score(y,pred)

        fpr,tpr,thresholds=roc_curve(y,y_scores)
        roc_auc=auc(fpr,tpr)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {roc_auc:.4f}")

    def train_logistic_regression(self):
        lr_model=LogisticRegression(n_jobs=-1)
        lr_model.fit(self.X_train,self.y_train)
        print(f'Logistic regression results on training data: ')
        self.evaluate(lr_model,self.X_train,self.y_train)
        print(f'Logistic regression results on testing data: ')
        self.evaluate(lr_model,self.X_test,self.y_test)
        return lr_model

    def train_random_forest(self):
        rf_model=RandomForestClassifier(n_jobs=-1,n_estimators=300,max_depth=10,bootstrap=True)
        rf_model.fit(self.X_train,self.y_train)
        print(f'Random forest results on training data: ')
        self.evaluate(rf_model,self.X_train,self.y_train)
        print(f'Random forest results on testing data: ')
        self.evaluate(rf_model,self.X_test,self.y_test)
        return rf_model

    def train_xgboost(self):
        xgb_model=XGBClassifier(n_jobs=-1,learning_rate=0.01,n_estimators=1000,gamma=0.01)
        xgb_model.fit(self.X_train,self.y_train)
        print(f'XGBoost results on training data: ')
        self.evaluate(xgb_model,self.X_train,self.y_train)
        print(f'XGBoost results on testing data: ')
        self.evaluate(xgb_model,self.X_test,self.y_test)
        return xgb_model

    def train_voting_classifier(self,rf_model,xgb_model):
        vc_model=VotingClassifier(n_jobs=-1,estimators=[('rf',rf_model),('xgb',xgb_model)],weights=[1,1],voting='soft')
        vc_model.fit(self.X_train,self.y_train)
        print(f'Voting classifier results on training data: ')
        self.evaluate(vc_model,self.X_train,self.y_train)
        print(f'Voting classifier results on testing data: ')
        self.evaluate(vc_model,self.X_test,self.y_test)
        return vc_model
    

    def train_mlp_classifier(self):
        mlp_model=MLPClassifier(hidden_layer_sizes=[256,128,64],
                                batch_size=32,
                                learning_rate_init=0.001,
                                shuffle=True,
                                random_state=101)
        mlp_model.fit(self.X_train,self.y_train)
        print(f'MLP classifier results on training data: ')
        self.evaluate(mlp_model,self.X_train,self.y_train)
        print(f'MLP classifier results on testing data: ')
        self.evaluate(mlp_model,self.X_test,self.y_test)
        return mlp_model