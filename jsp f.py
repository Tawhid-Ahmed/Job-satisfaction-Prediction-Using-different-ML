# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:57:43 2022

@author: ASUS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder#for train test splitting
from sklearn.model_selection import train_test_split#for decision tree object
from sklearn.tree import DecisionTreeClassifier#for checking testing results
from sklearn.metrics import classification_report, confusion_matrix#for visualizing tree 
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from IPython.display import Image 
from io import StringIO
import pydotplus
from sklearn import tree
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler

#reading the data
df2 = pd.read_excel (r'job selection problem 1.xlsx')
df = pd.DataFrame(df2)
df.head()

target = df['Are you satisfied with your current job?']
df1 = df.copy()
df1 = df1.drop('Are you satisfied with your current job?', axis =1)

# Defining the attributes
X = df1

target

#label encoding
le = LabelEncoder()
target = le.fit_transform(target)
target
for i in range(66):
    X.iloc[:,i] = le.fit_transform(X.iloc[:,i].astype(str))

y = target

# Splitting the data - 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.20, random_state =0)
print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
lr_pred=logreg.predict(X_test)
print(confusion_matrix(y_test,lr_pred))
print(classification_report(y_test,lr_pred))

lcm = confusion_matrix(y_test, lr_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=lcm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(logreg.score(X_test, y_test))
plt.title(all_sample_title, size = 15)


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
gnb_pred = gnb.predict(X_test)

print(confusion_matrix(y_test,gnb_pred))
print(classification_report(y_test,gnb_pred))

gcm = confusion_matrix(y_test, gnb_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=gcm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {:.4f}'.format(gnb.score(X_test, y_test))
plt.title(all_sample_title, size = 15)


from sklearn.svm import SVC
svclassifier = SVC(kernel='poly')
svclassifier.fit(X_train, y_train)
svm_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,svm_pred))
print(classification_report(y_test,svm_pred))

scm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=scm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {:.4f}'.format(svclassifier.score(X_test, y_test))
plt.title(all_sample_title, size = 15)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=125,criterion='gini')
rf.fit(X_train,y_train)

rf_pred=rf.predict(X_test)
print(confusion_matrix(y_test,rf_pred))
print(classification_report(y_test,rf_pred))

fcm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=fcm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {:.4f}'.format(rf.score(X_test, y_test))
plt.title(all_sample_title, size = 15)


from sklearn.neural_network import MLPClassifier


mlp = MLPClassifier(hidden_layer_sizes=(20),random_state =0, activation='relu', solver='adam', max_iter=1500)

mlp.fit(X_train,y_train)


predict_train = mlp.predict(X_train)

predict_test = mlp.predict(X_test)

print(confusion_matrix(y_train,predict_train))

print(classification_report(y_test,predict_test))

mcm = confusion_matrix(y_test, predict_test)
plt.figure(figsize=(5,5))
sns.heatmap(data=mcm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {:.4f}'.format(mlp.score(X_test, y_test))
plt.title(all_sample_title, size = 15)
plt.title('Confussion matrix of MLP')


# Defining the decision tree algorithm
dtree=DecisionTreeClassifier(criterion='gini')
dtree.fit(X_train,y_train)
print('Decision Tree Classifier Created')

y_pred = dtree.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.title('Confussion matrix of Decision Tree')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {:.4f}'.format(dtree.score(X_test, y_test))
plt.title(all_sample_title, size = 15)
#plt.title('Confussion matrix of Decision Tree')





















































