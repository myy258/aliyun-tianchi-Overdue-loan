# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:22:15 2020

@author: myy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import datetime
np.random.seed(2020) 

train_file_name = 'train.csv'
test_file_name = 'testA.csv'

df_train = pd.read_csv(train_file_name)
df_test = pd.read_csv(test_file_name)

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
df_train['grade'] = labelEncoder.fit_transform(df_train['grade'])
df_train['grade'] = labelEncoder.fit_transform(df_train['grade'])

employmentLength = ['< 1 year','1 year','2 years',
                    '3 years',  '5 years', '4 years', 
                    '6 years', '8 years', '7 years','9 years','10+ years']
j = 0
for i in employmentLength:
    df_train['employmentLength'] = df_train['employmentLength'].replace(i, j)
    j += 1

_ = pd.crosstab(df_train.subGrade, df_train.isDefault)
_["yp"] = _[1]/(_[0]+_[1])
_.reset_index(inplace=True)
_.sort_values(by="yp", inplace=True)

df_train = pd.merge(df_train, _[["subGrade", "yp"]], on="subGrade", how="left")
df_train['subGrade'] = labelEncoder.fit_transform(df_train['subGrade'])
df_train['issueDate'] = pd.to_datetime(df_train['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
df_train['issueDateDT'] = df_train['issueDate'].apply(lambda x: x-startdate).dt.days
df_train['earliesCreditLine'] = df_train['earliesCreditLine'].apply(lambda s: int(s[-4:]))

tags = ['loanAmnt', 'term', 'interestRate', 'installment', 'grade',
       'subGrade', 'employmentTitle', 'employmentLength', 'homeOwnership',
       'annualIncome', 'verificationStatus', 'issueDateDT', 'earliesCreditLine',
       'purpose', 'postCode', 'regionCode', 'dti', 'delinquency_2years',
       'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'pubRec',
       'pubRecBankruptcies', 'revolBal', 'revolUtil', 'totalAcc',
       'initialListStatus', 'applicationType', 'title',
       'n0', 'n1', 'n2', 'n4', 'n5', 'n6', 'n7', 'n8',
       'n9', 'n10', 'n11', 'n12', 'n13', 'n14','yp']

df_test['grade'] = labelEncoder.fit_transform(df_test['grade'])

employmentLength = ['< 1 year','1 year','2 years',
                    '3 years',  '5 years', '4 years', 
                    '6 years', '8 years', '7 years','9 years','10+ years']
j = 0
for i in employmentLength:
    df_test['employmentLength'] = df_test['employmentLength'].replace(i, j)
    j += 1

df_test = pd.merge(df_test, _[["subGrade", "yp"]], on="subGrade", how="left")
df_test['subGrade'] = labelEncoder.fit_transform(df_test['subGrade'])
df_test['issueDate'] = pd.to_datetime(df_test['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
df_test['issueDateDT'] = df_test['issueDate'].apply(lambda x: x-startdate).dt.days
df_test['earliesCreditLine'] = df_test['earliesCreditLine'].apply(lambda s: int(s[-4:]))


Standard_scaler = StandardScaler()
Standard_scaler.fit(df_train[tags].values)
x = Standard_scaler.transform(df_train[tags].values)
x_ = Standard_scaler.transform(df_test[tags].values)
y = df_train['isDefault'].values


lgbr = LGBMRegressor(num_leaves=30
                        ,max_depth=10
                        ,learning_rate=0.01
                        ,n_estimators=13000
                        ,subsample_for_bin=5000
                        ,min_child_samples=200
                        ,colsample_bytree=.2
                        ,reg_alpha=.1
                        ,reg_lambda=.1
                        ,seed=2020                       
                        )


cat = CatBoostRegressor(depth=9, 
                            l2_leaf_reg=1, 
                            learning_rate=0.01, 
                            eval_metric = 'AUC' ,
                            border_count = 128, 
                            bagging_temperature = 0.9 , 
                            n_estimators=16000,
                            early_stopping_rounds=500, 
                            subsample = 0.9,
                            random_seed=1,
                            verbose = 0)


from sklearn.ensemble import VotingRegressor

rg_model = VotingRegressor([('lgb', lgbr), ('catboost', cat)],n_jobs=12)

# kf = KFold(n_splits=10, shuffle=True, random_state=100)
# devscore = []
# for tidx, didx in kf.split(train.index):
#     tf = train.iloc[tidx]
#     df = train.iloc[didx]
#     tt = y.iloc[tidx]
#     dt = y.iloc[didx]
#     rg_model.fit(tf, tt)
#     pre = lgbr.predict(df)
#     fpr, tpr, thresholds = roc_curve(dt, pre)
#     score = auc(fpr, tpr)
#     devscore.append(score)
# print(np.mean(devscore))

# rg_model.fit(x,y)
# y_1 = rg_model.predict(x_test)
# from sklearn.metrics import roc_auc_score
# print(roc_auc_score(y_test, y_1))


rg_model.fit(x,y)
pre = pd.DataFrame(rg_model.predict(x_),columns=['isDefault'])
results = pd.concat([df_test['id'],pre],axis = 1)
results.to_csv('lgb&cat.csv', index=False)