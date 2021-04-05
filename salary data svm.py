# Importing Dataset
import pandas as pd
import numpy as np 
train = pd.read_csv("../17.Support Vector Machines/SalaryData_Train.csv")
test = pd.read_csv("../17.Support Vector Machines/SalaryData_Test.csv")
del train['educationno'], test['educationno']

# checking data type of all variables
col=pd.DataFrame(train.columns)
col_dtypes = pd.DataFrame(train.dtypes)

train = train.iloc[:,[12,11,7,6,5,4,3,2,1,10,9,8,0]]
test = test.iloc[:,[12,11,7,6,5,4,3,2,1,10,9,8,0]]

# For summary
train.describe()

# creating Dummy variable for categorical variables on train data
train_dummy_native= pd.get_dummies(train.native,prefix='native',prefix_sep='_')
train_dummy_sex = pd.get_dummies(train.sex,prefix='sex',prefix_sep='_')
train_dummy_race = pd.get_dummies(train.race,prefix='race',prefix_sep='_')
train_dummy_relationship = pd.get_dummies(train.relationship,prefix='relationship',prefix_sep='_')
train_dummy_occupation = pd.get_dummies(train.occupation,prefix='occupation',prefix_sep='_')
train_dummy_maritalstatus = pd.get_dummies(train.maritalstatus,prefix='maritalstatus',prefix_sep='_')
train_dummy_education = pd.get_dummies(train.education,prefix='education',prefix_sep='_')
train_dummy_workclass = pd.get_dummies(train.workclass,prefix='workclass',prefix_sep='_')

# joining the dummy variables with train data
train = train.join([train_dummy_native,train_dummy_sex,train_dummy_race
              ,train_dummy_relationship,train_dummy_occupation
              ,train_dummy_maritalstatus,train_dummy_education,train_dummy_workclass])

# deleting the categorical variable from train data
train.dtypes
del train["native"],train["sex"],train["race"],train["relationship"],train["occupation"],train["maritalstatus"],train["education"],train["workclass"]

# creating Dummy variable for categorical variables on test data
test_dummy_native= pd.get_dummies(test.native,prefix='native',prefix_sep='_')
test_dummy_sex = pd.get_dummies(test.sex,prefix='sex',prefix_sep='_')
test_dummy_race = pd.get_dummies(test.race,prefix='race',prefix_sep='_')
test_dummy_relationship = pd.get_dummies(test.relationship,prefix='relationship',prefix_sep='_')
test_dummy_occupation = pd.get_dummies(test.occupation,prefix='occupation',prefix_sep='_')
test_dummy_maritalstatus = pd.get_dummies(test.maritalstatus,prefix='maritalstatus',prefix_sep='_')
test_dummy_education = pd.get_dummies(test.education,prefix='education',prefix_sep='_')
test_dummy_workclass = pd.get_dummies(test.workclass,prefix='workclass',prefix_sep='_')

# joining the dummy variables with test data
test = test.join([test_dummy_native,test_dummy_sex,test_dummy_race
              ,test_dummy_relationship,test_dummy_occupation
              ,test_dummy_maritalstatus,test_dummy_education,test_dummy_workclass])

# deleting the categorical variable from test data
test.dtypes
del test["native"],test["sex"],test["race"],test["relationship"],test["occupation"],test["maritalstatus"],test["education"],test["workclass"]

# checking output variable; Salary
train.Salary.unique()
train.Salary.value_counts() # Imbalanced data

# save CSV file to cwd
import os
os.getcwd()
train.to_csv("trainn.csv",encoding = "utf-8")
test.to_csv("testt.csv", encoding="utf-8")

# EDA Part : Boxplot representation
import seaborn as sns
train.columns
sns.boxplot(x="age",y="Salary",data=train,palette = "hls")
sns.boxplot(x="hoursperweek",y="Salary",data=train,palette = "hls")
sns.boxplot(x="capitalloss",y="Salary",data=train,palette = "hls")
sns.boxplot(x="capitalgain",y="Salary",data=train,palette = "hls")

# train_X,train_y and test_X,test_y split
train_X = train.iloc[:,1:]
train_y = train.iloc[:,0]
test_X  = test.iloc[:,1:]
test_y  = test.iloc[:,0]

from sklearn.svm import SVC
# help(SVC)
# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid'

# kernel = linear
model_linear = SVC(kernel = "linear", verbose = True)
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

import numpy as np
np.mean(pred_test_linear==test_y)

# Kernel = poly
model_poly = SVC(kernel = "poly",verbose = True)
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) 

# kernel = rbf
model_rbf = SVC(kernel = "rbf", verbose = True)
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)

# kernel = sigmoid
model_sig = SVC(kernel = "sigmoid", verbose = True)
model_sig.fit(train_X,train_y)
pred_test_sig = model_sig.predict(test_X)

np.mean(pred_test_sig==test_y) 

