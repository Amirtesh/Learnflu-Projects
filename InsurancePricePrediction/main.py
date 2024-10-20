import warnings
warnings.filterwarnings('ignore')

from preprocessing import Preprocessor
path='insurance.csv'

preprocessor=Preprocessor(file_path=path)
preprocessor.augment_data()
preprocessor.shuffle_data()
preprocessor.encode_data()

df=preprocessor.get_processed_data()
print(df.head())


from eda import EDA

eda=EDA(df=df)
eda.summary()
eda.correlation_matrix()
eda.plot_distributions()
eda.plot_boxplots()



from data_splitter import Splitter

data_splitter=Splitter(df=df,target='expenses')
data_splitter.split_data()
X_train,X_test,y_train,y_test=data_splitter.get_splits(scale=True)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)


from training import Trainer
from evaluate import Evaluator


trainer=Trainer(X_train,y_train,X_test,y_test,data_splitter.scaler_y)

lin_model=trainer.train_linear_regression()
lin_evaluator=Evaluator(lin_model,X_test,y_test,data_splitter.scaler_y)
lin_evaluator.evaluate()
lin_evaluator.plot_predictions()

knn_model=trainer.train_knn()
knn_evaluator=Evaluator(knn_model,X_test,y_test,data_splitter.scaler_y)
knn_evaluator.evaluate()
knn_evaluator.plot_predictions()

rf_model=trainer.train_random_forest()
rf_evaluator=Evaluator(rf_model,X_test,y_test,data_splitter.scaler_y)
rf_evaluator.evaluate()
rf_evaluator.plot_predictions()

xgb_model=trainer.train_xgboost()
xgb_evaluator=Evaluator(xgb_model,X_test,y_test,data_splitter.scaler_y)
xgb_evaluator.evaluate()
xgb_evaluator.plot_predictions()

vote_model=trainer.train_voting(knn_model,rf_model,xgb_model)
vote_evaluator=Evaluator(vote_model,X_test,y_test,data_splitter.scaler_y)
vote_evaluator.evaluate()
vote_evaluator.plot_predictions()


knn_df=trainer.make_df(knn_model,X_test,y_test)
print(knn_df.head(20))
