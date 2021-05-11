# Multi-class Classification for Self-admitted Technical Debt Based on XGBoost

The dataset used for paper "Multi-class Classification for Self-admitted Technical Debt Based on XGBoost" is in xgboost/data/technical_debt_dataset

data_helper.py contains the preprocess of the raw data.

fs.py contains the work of feature selection part.

generate_sample.py contains the work of data augmentation.

xgboost_satd.py trains the model used in the paper and performs the prediction.
 
# The process to use the code 

a. run data_helper.py to preprocess the data

b. run generate_sample.py to generate the data of augmentation

c. run xgboost_satd to train xgboost classifier and predict the results.
