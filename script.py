#!/usr/bin/env python3

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve, GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Ignore useless warnings
warnings.filterwarnings("ignore")

# Step 1: Remove unwanted columns from the dataset
def usefulDF(filename):
# Read data
  data = pd.read_csv(filename, low_memory=False)
# print(data.shape)
  drop_list = ['sub_grade',
  'emp_title',
  'issue_d',
  'desc',
  'title',
  'zip_code',
  'addr_state',
  'earliest_cr_line',
  'last_pymnt_d',
  'next_pymnt_d',
  'last_credit_pull_d',
  'sec_app_earliest_cr_line',
  'hardship_start_date',
  'hardship_end_date',
  'payment_plan_start_date',
  'hardship_loan_status',
  'debt_settlement_flag_date',
  'settlement_date',
  'id',
  'member_id',
  'url',
  'id',
  'member_id',
  'url',
  'desc',
  'mths_since_last_delinq',
  'mths_since_last_record',
  'next_pymnt_d',
  'mths_since_last_major_derog',
  'annual_inc_joint',
  'dti_joint',
  'verification_status_joint',
  'open_acc_6m',
  'open_act_il',
  'open_il_12m',
  'open_il_24m',
  'mths_since_rcnt_il',
  'total_bal_il',
  'il_util',
  'open_rv_12m',
  'open_rv_24m',
  'max_bal_bc',
  'all_util',
  'inq_fi',
  'total_cu_tl',
  'inq_last_12m',
  'mths_since_recent_bc_dlq',
  'mths_since_recent_revol_delinq',
  'revol_bal_joint',
  'sec_app_earliest_cr_line',
  'sec_app_inq_last_6mths',
  'sec_app_mort_acc',
  'sec_app_open_acc',
  'sec_app_revol_util',
  'sec_app_open_act_il',
  'sec_app_num_rev_accts',
  'sec_app_chargeoff_within_12_mths',
  'sec_app_collections_12_mths_ex_med',
  'sec_app_mths_since_last_major_derog',
  'hardship_type',
  'hardship_reason',
  'hardship_status',
  'deferral_term',
  'hardship_amount',
  'hardship_start_date',
  'hardship_end_date',
  'payment_plan_start_date',
  'hardship_length',
  'hardship_dpd',
  'hardship_loan_status',
  'orig_projected_additional_accrued_interest',
  'hardship_payoff_balance_amount',
  'hardship_last_payment_amount',
  'debt_settlement_flag_date',
  'settlement_status',
  'settlement_date',
  'settlement_amount',
  'settlement_percentage',
  'settlement_term'
  ]
  data = data.drop(drop_list, axis=1)
  # save the new dataset as a CSV so that you don't have to run this function again
  data.to_csv('useful.csv')

# Process the new dataset 
def preprocessing(filename):
  data = pd.read_csv('useful.csv')
  data.dropna(subset=['acc_now_delinq'], inplace=True)
  df = pd.DataFrame({
    'Count': data.isnull().sum(),
    'Percent': 100*data.isnull().sum()/len(data)
  })
  for i in list(data.columns.values):
    if (data[i].isnull().sum() / len(data)) > 0.05:
      data.dropna(subset=[i], how="all", inplace=True)
  data.dti.fillna(data.dti.mean(), inplace=True)
  data.inq_last_6mths.fillna(data.inq_last_6mths.mode()[0], inplace=True)
  data.revol_util.fillna(data.revol_util.mean(),inplace=True)
  data.avg_cur_bal.fillna(-100000000,inplace=True)
  data.bc_open_to_buy.fillna(data.bc_open_to_buy.mode()[0],inplace=True)
  data.bc_util.fillna(data.bc_util.mean(),inplace=True)
  data.mths_since_recent_bc.fillna(-100000000,inplace=True)
  data.num_rev_accts.fillna(-100000000,inplace=True)
  data.num_tl_120dpd_2m.fillna(0,inplace=True)
  data.pct_tl_nvr_dlq.fillna(data.pct_tl_nvr_dlq.mean(),inplace=True)
  data.percent_bc_gt_75.fillna(data.percent_bc_gt_75.mode()[0],inplace=True)

  # Taking Loan status out as y and join it again at the end of the dataset
  y = data["loan_status"]
  newdata = data.drop(columns=["loan_status"])
  newdata = newdata.join(y) 

  # Categorical Encoding
  categorical_vars = list(newdata.columns[newdata.dtypes == object].values)
  for i in categorical_vars:
    print(newdata.columns.get_loc(i))  
  transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [4, 7, 8, 9, 11, 12, 13, 22, 35, 75, 76, 77])], remainder="passthrough")
  newdata = transformer.fit_transform(newdata)

  labelencoder = LabelEncoder()
  newdata[:, -1] = labelencoder.fit_transform(newdata[:, -1])
  return newdata

def classify():
  processed_data = preprocessing('useful.csv')
  X = processed_data[:, :-1]
  y = processed_data[:, -1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  def build_classifier():  
    classifier = Sequential()
    classifier.add(Dense(units=5, kernel_initializer='glorot_uniform', activation='relu'))
    classifier.add(Dense(units=5, kernel_initializer='glorot_uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
  
  classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=3)
  accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1)

  print(accuracies.mean())
  print(accuracies.std())

  def another_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer="glorot_uniform", activation="relu"))
    classifier.add(Dense(units=6, kernel_initializer="glorot_uniform", activation="relu"))
    classifier.add(Dense(units=1, kernel_initializer="glorot_uniform", activation="sigmoid"))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

  classifier = KerasClassifier(build_fn=build_classifier)
  parameters = {
    'batch_size': [25,32], 
    'nb_epoch':[1,2], 
    'optimizer': ['adam', 'rmsprop']
  }
  grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
  grid_search = grid_search.fit(X_train, y_train)

  best_parameters = grid_search.best_params_
  best_accuracy = grid_search.best_score_

  print(best_parameters)
  print(best_accuracy)

classify()