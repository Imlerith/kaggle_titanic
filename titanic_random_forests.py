#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 21:19:35 2016

@author: nasekins
"""

import os
import csv
import numpy as np
import pandas as pd
#from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression
from scipy.stats import randint as sp_randint
#from sklearn.pipeline import Pipeline
import seaborn as sns
sns.set(style="ticks")


os.chdir("/Users/nasekins/Documents/kaggle_titanic")


#DATA READING AND PREPARATION
train_df = pd.read_csv('train.csv',header=0)

train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int) #encode the "Gender" variable

#Fill in the missing Embarked values 
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

ports             = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
ports_dict        = { name : i for i, name in ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: ports_dict[x]).astype(int)     # Convert all Embarked strings to int
train_df          = train_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

###############################################################################
""" DO THE SAME FOR THE TEST SET AND THEN MERGE THE TWO TOGETHER
"""
test_df           = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values

# Again convert all Embarked strings to int
test_df.Embarked  = test_df.Embarked.map( lambda x: ports_dict[x]).astype(int)
ids               = test_df['PassengerId'].values
test_df           = test_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

age_pclass_test  = test_df.groupby('Pclass').mean().loc[:,'Age'].to_dict()                      
fare_pclass_test = test_df.groupby('Pclass').mean().loc[:,'Fare'].to_dict()
for f in range(0,3):
    test_df.loc[(test_df.Fare.isnull()) & (test_df.Pclass == f+1),'Fare'] = fare_pclass_test[f+1]

for f in range(0,3):
    test_df.loc[(test_df.Age.isnull()) & (test_df.Pclass == f+1),'Age'] = age_pclass_test[f+1]

#test_df["Pclass"].hist(bins = 10)
#sns.factorplot('Pclass', 'Survived', data = train_df)
#sns.violinplot(x='Pclass', y='Age', hue='Survived', data = train_df, split=True)
test_df.info()                            


#OBTAIN THE X AND Y MATRICES FOR MODEL FITTING

#median_age  = train_df['Age'].dropna().median()    
age_pclass  = train_df.groupby('Pclass').mean().loc[:,'Age'].to_dict() 
for f in range(0,3):
    train_df.loc[(train_df.Age.isnull()) & (train_df.Pclass == f+1),'Age'] = age_pclass[f+1]                    

#encode dummy variables
embark_dummies_full         = pd.get_dummies(train_df['Embarked']).astype(int)
embark_dummies_full.columns = ports_dict.keys()
embark_dummies_full         = embark_dummies_full.drop(['S'], axis=1)
pclass_dummies_full         = pd.get_dummies(train_df['Pclass']).astype(int)
pclass_dummies_full.columns = ['fclass','sclass','thclass']
pclass_dummies_full         = pclass_dummies_full.drop(['thclass'], axis=1)
train_df                    = pd.concat([train_df,embark_dummies_full,pclass_dummies_full], axis=1)

train_df = train_df.drop(['Pclass', 'Embarked'], axis=1)
train_df.info() 

embark_dummies_full_test         = pd.get_dummies(test_df['Embarked']).astype(int)
embark_dummies_full_test.columns = ports_dict.keys()
embark_dummies_full_test         = embark_dummies_full_test.drop(['S'], axis=1)
pclass_dummies_full_test         = pd.get_dummies(test_df['Pclass']).astype(int)
pclass_dummies_full_test.columns = ['fclass','sclass','thclass']
pclass_dummies_full_test         = pclass_dummies_full_test.drop(['thclass'], axis=1)
test_df                          = pd.concat([test_df,embark_dummies_full_test,pclass_dummies_full_test], axis=1)

test_df = test_df.drop(['Pclass', 'Embarked'], axis=1)
test_df.info() 

#TRANSFORM FAMILY VARIABLES
train_df['Family'] =  train_df["Parch"] + train_df["SibSp"]
test_df['Family']  =  test_df["Parch"] + test_df["SibSp"]
sns.barplot(x='Family', y='Survived', data=train_df)

fam_survived  = train_df.groupby('Family').mean().loc[:,'Survived'].to_dict() 
def get_surv(col,surv_dict=fam_survived):
    if surv_dict[col] > 0.5:
        return 1
    else:
        return 0
        

train_df['fam_help'] = train_df['Family'].apply(get_surv)
#train_df['SibS_help'] = train_df['SibSp'].apply(get_surv_sibsp)

test_df['fam_help'] = train_df['Family'].apply(get_surv)
#test_df['SibS_help'] = train_df['SibSp'].apply(get_surv_sibsp)


# Drop Parch & SibSp
train_df = train_df.drop(['SibSp','Parch','Family'], axis=1)
test_df  = test_df.drop(['SibSp','Parch','Family'], axis=1)

def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 10 else sex
    
train_df['Person']   = train_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

train_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

person_dummies_train         = pd.get_dummies(train_df['Person'])
person_dummies_train.columns = ['Child','Female','Male']
person_dummies_train.drop(['Male'], axis=1, inplace=True)

person_dummies_test         = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

train_df = train_df.join(person_dummies_train)
test_df  = test_df.join(person_dummies_test)

person_perc = train_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
#sns.barplot(x='Person', y='Survived', data=person_perc, order=['male','female','child'])

train_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)

train_df.info() 
test_df.info() 

Y_train = train_df['Survived'].values
X_train = train_df.iloc[:,1:len(train_df.columns.tolist())]
X_test = test_df

X_train.info()
X_test.info()


#TRAIN A RANDOM FOREST CLASSIFIER WITH RANDOMIZED CROSS-VALIDATION SEARCH

# Specify parameters and distributions to sample from
param_dist = {"max_depth": [3, 4, 5, 6, 7, 8, None],
              "max_features": sp_randint(1, 10),
              "min_samples_split": sp_randint(1, 12),
              "min_samples_leaf": sp_randint(1, 12),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# Run randomized search with jobs running in parallel
n_iter_search = 1000
clf = RandomForestClassifier(n_estimators=100)
random_search = RandomizedSearchCV(clf, param_distributions = param_dist,
                                   n_iter = n_iter_search, n_jobs = -1)
random_search.fit(X_train, Y_train)

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
report(random_search.cv_results_)
best_forest = random_search.best_estimator_



#WRITE THE OUTPUT FILE
test_pred = best_forest.predict(X_test).astype(int)
predictions_file = open("svm_predict.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids,test_pred))
predictions_file.close()


