# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:21:17 2019

@author: ARJUN MENON
"""
# importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
#importing datasets
df_features = pd.read_csv('dengue_features_train.csv')
df_lables = pd.read_csv('dengue_labels_train.csv')
df_test = pd.read_csv('dengue_features_test.csv')

# preparing feature data
def features_data(a):
    dummies = pd.get_dummies(a['city'])
    merged = pd.concat([dummies,a], axis = 'columns')
    x = merged.drop('city', axis = 'columns' )
    
    x['week_start_date'] = x['week_start_date'].str.replace('-','').astype(int)
    
    x = x.values
    
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(x[:,4:])
    x[:,4:] = imputer.transform(x[:,4:])
    return (x)

X_train = features_data(df_features)
y_train = df_lables.iloc[ : , -1].values
X_test = features_data(df_test)
#RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
test_prediction = np.rint(regressor.predict(X_test))
#Preparing output
label = pd.DataFrame({'total_cases':test_prediction}) 
features = pd.DataFrame(df_test.iloc[:,:3])
output = pd.concat([features,label], axis = 'columns')
output['total_cases'] = output['total_cases'].astype(int)
#getting output file as comma seprated file
submission = pd.DataFrame(output).to_csv('DengAI_submission.csv', index = False)
