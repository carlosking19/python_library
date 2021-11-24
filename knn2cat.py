#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, minmax_scale, QuantileTransformer , Normalizer, PowerTransformer
from sklearn.model_selection import train_test_split
import os

import pickle

import urllib.request
code = "https://github.com/carlosking19/python_library/blob/c0210e1fa2c385a851f847faab299bc47e581983/knn_5n_model.sav"
response = urllib.request.urlopen(code)
data = response.read()
exec(data)

filename = (os.path.join(os.path.dirname(__file__),data))

loaded_model = pickle.load(open(filename, 'rb'))

#print('KNN training score {:.3f}'.format(grid_search.score(X_train, y_train)))
# print('KNN testing score {:.3f}'.format(grid_search.score(X_test, y_test)))
import pip

import mysql.connector
mydb = mysql.connector.connect(host = "localhost", user = "root", passwd = "", database = "grading_db")

mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM student_descriptive")

myresult = pd.DataFrame(mycursor,columns = ['section id', 'section sis id', 'section name', 'course id',
       'course sis id', 'course name', 'term id', 'term sis id', 'term name',
       'enrollment type', 'user id', 'user sis id', 'user name',
       'content type', 'content', 'times viewed', 'times participated',
       'last viewed'])
myresult["user id"] = myresult["user id"].astype("float")
myresult["times viewed"] = myresult["times viewed"].astype("float")
myresult

pivot_df = myresult.pivot_table(values='times viewed', index=['user id'], columns='content type', aggfunc=pd.Series.sum)
pivot_df = pd.DataFrame(pivot_df)
pivot_df.fillna(0)
pivot_df.insert(4, 'assignment sum',pivot_df['assignment']+pivot_df['assignments'])
pivot_df.drop(['assignment','assignments'], axis = 1, inplace=True)

pivot_df.insert(6, 'collaboration',pivot_df['collaborations'])
pivot_df.drop(['collaborations'], axis = 1, inplace=True)
# pivot_df.columns
pivot_df = pivot_df.fillna(0)
pivot_df['user id'] = pivot_df.index
pivot_df.reset_index(drop=True, inplace=True)
pivot_df = pivot_df[['user id', 'announcements', 'assignment sum', 'discussion', 'files', 'grades', 'modules', 'quizzes', 'quizzesquiz', 'topics', 'wiki']]
pivot_df
#Convert Index User ID into column
x_no_out = pivot_df.loc[:,['announcements', 'assignment sum', 'discussion', 'files', 'grades', 'modules', 'quizzes', 'quizzesquiz', 'topics', 'wiki']].values
scaler = StandardScaler()
x_no_out_scaled = scaler.fit_transform(x_no_out)
# print(X_no_out.shape)
x_no_out = pivot_df.loc[:,['announcements', 'assignment sum', 'discussion', 'files', 'grades', 'modules', 'quizzes', 'quizzesquiz', 'topics', 'wiki']].values
y_test_hat=loaded_model.predict(x_no_out_scaled)



y_test_hat
# df = pd.DataFrame(mlp_predict, columns = ['Predicted_Grade'])
# print(df)
pivot_df['PREDICTED_STATUS'] = y_test_hat
pivot_df[["PREDICTED_STATUS"]].value_counts()
pivot_df = pivot_df.replace({'PREDICTED_STATUS': {'Pass': 'PASS', 'Fail': 'FAIL'}})
#pivot_df.drop (['Predicted_Grade'], axis=1, inplace=True)

predict_val = pivot_df[['user id', 'PREDICTED_STATUS']].to_dict()

import pymysql
import pandas as pd
import mysql.connector

mydb = mysql.connector.connect(host = "localhost", user = "root", passwd = "", database = "grading_db")
# create cursor
mycursor = mydb.cursor()
#sql = f" UPDATE student_predictive SET PREDICTED_STATUS = 'try' "
for i in range(len(predict_val['user id'])):
    sql = "UPDATE student_predictive SET PREDICTED_STATUS = %s WHERE USER_ID = %s"
    mycursor.execute(sql, [predict_val['PREDICTED_STATUS'][i], predict_val['user id'][i]])

# predict_val['PREDICTED_STATUS'][i]
# {predict_val['user id'][i]
mydb.commit()
print("KNN 2cat")
