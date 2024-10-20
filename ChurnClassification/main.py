import warnings
warnings.filterwarnings('ignore')

from preprocessing import Preprocessor

file_path='CustomerChurn.csv'
preprocessor=Preprocessor(file_path)
preprocessor.shuffle_data()
df=preprocessor.get_processed_data()
print(df.head())
print(df.shape)


from eda import EDA

eda=EDA(df)
eda.display_basic_info()
eda.display_statistics()
eda.plot_correlation_heatmap()
eda.plot_churn_distribution()


from data_splitter import Splitter

data_splitter=Splitter(df,target='Churn')
data_splitter.split_data()
X_train,X_test,y_train,y_test=data_splitter.get_splits(scale=True)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


from training import Trainer
from evaluate import Evaluator

trainer=Trainer(X_train,y_train,X_test,y_test)

log_model=trainer.train_logistic_regression()
log_evaluator=Evaluator(log_model,X_test,y_test)
log_evaluator.evaluate()
log_evaluator.plot_confusion_matrix()
log_evaluator.plot_roc_curve()

rf_model=trainer.train_random_forest()
rf_evaluator=Evaluator(rf_model,X_test,y_test)
rf_evaluator.evaluate()
rf_evaluator.plot_confusion_matrix()
rf_evaluator.plot_roc_curve()

xgb_model=trainer.train_xgboost()
xgb_evaluator=Evaluator(xgb_model,X_test,y_test)
xgb_evaluator.evaluate()
xgb_evaluator.plot_confusion_matrix()
xgb_evaluator.plot_roc_curve()


vc_model=trainer.train_voting_classifier(rf_model,xgb_model)
vc_evaluator=Evaluator(vc_model,X_test,y_test)
vc_evaluator.evaluate()
vc_evaluator.plot_confusion_matrix()
vc_evaluator.plot_roc_curve()

mlp_model=trainer.train_mlp_classifier()
mlp_evaluator=Evaluator(mlp_model,X_test,y_test)
mlp_evaluator.evaluate()
mlp_evaluator.plot_confusion_matrix()
mlp_evaluator.plot_roc_curve()


mlp_evaluator.predict_single(100)
mlp_evaluator.predict_range(1,20)