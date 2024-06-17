import time
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import tensorflow as tf
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from Preprocessing import *
from MyLabelEncoder import *
from Load_Preprocessing import *
import csv

import pickle

df = pd.read_csv('movies-classification-dataset.csv')

# Split the data to training and testing sets
# =============================================
target_map = {'High': 2, 'Intermediate': 1, 'Low': 0}
df['Rate'] = df['Rate'].map(target_map)
X = df.iloc[:, 0:19]  # Features
Y = df['Rate']  # Label
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=False, random_state=275)
# =========================================================================================================================================
X_train,y_train = Preprocessing_Train_Test(X_train,y_train,'train')
# ============================================================================================================================

X_test,y_test = Preprocessing_Train_Test(X_test,y_test,'test')

# Select the k best features using f_classif statistical test
k = 6  # number of top features to select
selector = SelectKBest(f_classif, k=k)
selector.fit(X_train, y_train)

# Get the selected features and their scores
selected_features = X_train.columns[selector.get_support()]
scores = selector.scores_[selector.get_support()]

X_train = X_train[selected_features]
X_test = X_test[selected_features]


# # Print the selected features and their scores
# for feature, score in zip(selected_features, scores):
#     print(feature, score)
# sns.heatmap(scores, annot=True)
# plt.show()

# sns.pairplot(df[X_train + [y_train]], diag_kind='hist')
# plt.show()
# print(y_train)
# print(y_test)

# Plot the scores of the features
plt.bar(range(len(X_train.columns)), scores)
plt.xticks(range(len(X_train.columns)), X_train.columns, rotation=90)
plt.xlabel('Features')
plt.ylabel('Scores')
plt.show()

# =======================================================================================================
Model_Names = []
Classification_Accuracy = []
Training_Time_Models = []
Testing_Time_Models = []
# ==================================================================================================
# logistic regression
log_reg = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=200)
Model_Names.append('LR_Model')
start_time_LR = time.time()
log_reg.fit(X_train, y_train)
end_time_LR = time.time()
training_time_LR_model = end_time_LR - start_time_LR
Training_Time_Models.append(training_time_LR_model)
start_time_LR = time.time()
y_pred = log_reg.predict(X_test)
end_time_LR = time.time()
testing_time_LR_model = end_time_LR - start_time_LR
Testing_Time_Models.append(testing_time_LR_model)
print('Accuracy of logistic regression model = ', log_reg.score(X_test, y_test) * 100, "%")
Classification_Accuracy.append(log_reg.score(X_test, y_test) * 100)
# **********************************************************************************
# **********************************************************************************
filename = 'LogisticRegression_model.pkl'

with open(filename, 'wb') as f:
    pickle.dump(log_reg, f)
# **********************************************************************************
# **********************************************************************************

# =======================================================================================================
# apply RandomForestClassifier model
clf = RandomForestClassifier()
Model_Names.append('RFC_Model')
start_time_RFC = time.time()
clf.fit(X_train, y_train)
end_time_RFC = time.time()
training_time_RFC_model = end_time_RFC - start_time_RFC
Training_Time_Models.append(training_time_RFC_model)
start_time_RFC = time.time()
# predict the class labels for the test data
y_pred = clf.predict(X_test)
end_time_RFC = time.time()
testing_time_RFC_model = end_time_RFC - start_time_RFC
Testing_Time_Models.append(testing_time_RFC_model)
# calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of randomforestclassifier model = ", accuracy * 100, "%")
Classification_Accuracy.append(accuracy * 100)
# **********************************************************************************
# **********************************************************************************
filename = 'RandomForestClassifier_model.pkl'

with open(filename, 'wb') as f:
    pickle.dump(clf, f)
# **********************************************************************************
# **********************************************************************************

# ===============================================================================================================
# Train an SVM Classifier
# we create an instance of SVM and fit out data.
Model_Names.append('SVM linear kernel')
Model_Names.append('SVM rbf kernel')
Model_Names.append('SVM poly kernel')
C = [1, 2, 3]  # SVM regularization parameter
best_accuracy = 0
start_time_SVM_linear_C3 = 0
end_time_SVM_linear_C3 = 0
start_time_SVM_rbf_C3 = 0
end_time_SVM_rbf_C3 = 0
start_time_SVM_poly_C3 = 0
end_time_SVM_poly_C3 = 0
start_time_SVM_linear_C3_test = 0
end_time_SVM_linear_C3_test = 0
start_time_SVM_rbf_C3_test = 0
end_time_SVM_rbf_C3_test = 0
start_time_SVM_poly_C3_test = 0
end_time_SVM_poly_C3_test = 0
accuracy_linear = 0
accuracy_rbf = 0
accuracy_poly = 0
for i in C:
    print('when C = ', i)
    start_time_SVM_linear_C3 = time.time()
    lin_svc = SVC(kernel='linear', C=i).fit(X_train, y_train)
    end_time_SVM_linear_C3 = time.time()
    start_time_SVM_rbf_C3 = time.time()
    rbf_svc = SVC(kernel='rbf', gamma=0.8, C=i).fit(X_train, y_train)
    end_time_SVM_rbf_C3 = time.time()
    start_time_SVM_poly_C3 = time.time()
    poly_svc = SVC(kernel='poly', degree=3, C=i).fit(X_train, y_train)
    end_time_SVM_poly_C3 = time.time()
    # title for the plots
    titles = ['linear kernel',
              'RBF kernel',
              'polynomial (degree 3) kernel']

    y = 0
    for j, clf in enumerate((lin_svc, rbf_svc, poly_svc)):
        start_time_SVM_linear_C3_test = time.time()
        predictions = clf.predict(X_test)
        end_time_SVM_linear_C3_test = time.time()
        if j == 0:
            start_time_SVM_linear_C3_test = start_time_SVM_linear_C3_test
            end_time_SVM_linear_C3_test = end_time_SVM_linear_C3_test
        if j == 1:
            start_time_SVM_rbf_C3_test = start_time_SVM_linear_C3_test
            end_time_SVM_rbf_C3_test = end_time_SVM_linear_C3_test
        if j == 2:
            start_time_SVM_poly_C3_test = start_time_SVM_linear_C3_test
            end_time_SVM_poly_C3_test = end_time_SVM_linear_C3_test
        accuracy = np.mean(predictions == y_test)
        print('Accuracy of SVM with ', titles[y], 'model = ', accuracy * 100, "%")
        if j == 0:
            accuracy_linear = accuracy*100
        if j == 1:
            accuracy_rbf = accuracy*100
        if j == 2:
            accuracy_poly = accuracy*100
        y += 1
        # Store the best model with highest accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
            best_model_title = titles[y - 1]
            best_model_C = i
    print('=============================================================')
Classification_Accuracy.append(accuracy_linear)
Classification_Accuracy.append(accuracy_rbf)
Classification_Accuracy.append(accuracy_poly)
training_time_SVM_linear_C3_model = end_time_SVM_linear_C3 - start_time_SVM_linear_C3
Training_Time_Models.append(training_time_SVM_linear_C3_model)
training_time_SVM_rbf_C3_model = end_time_SVM_rbf_C3 - start_time_SVM_rbf_C3
Training_Time_Models.append(training_time_SVM_rbf_C3_model)
training_time_SVM_poly_C3_model = end_time_SVM_poly_C3 - start_time_SVM_poly_C3
Training_Time_Models.append(training_time_SVM_poly_C3_model)
testing_time_SVM_linear_model = end_time_SVM_linear_C3_test - start_time_SVM_linear_C3_test
Testing_Time_Models.append(testing_time_SVM_linear_model)
testing_time_SVM_rbf_model = end_time_SVM_rbf_C3_test - start_time_SVM_rbf_C3_test
Testing_Time_Models.append(testing_time_SVM_rbf_model)
testing_time_SVM_poly_model = end_time_SVM_poly_C3_test - start_time_SVM_poly_C3_test
Testing_Time_Models.append(testing_time_SVM_poly_model)
# **********************************************************
# **********************************************************
# Save the best model in a pickle file
filename = 'best_svm_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)
# **********************************************************
# **********************************************************


# ======================================================================================
# svm_onevsrest = OneVsRestClassifier(SVC(kernel='linear',C=1).fit(X_train,y_train))
# # model accuracy for X_test
# acc = svm_onevsrest.score(X_test,y_test)
# print('accuracy of onevsrestclassifier model = ',str(acc))
# ========================================================================================


Model_Names.append('KNN n_neighbors = 8')
# create an instance of the KNN classifier
N_Neighbors = [6, 7, 8]
best_acc = 0
best_model = None
start_time_KNN = 0
end_time_KNN = 0
start_time_KNN_test = 0
end_time_KNN_test = 0
accuracy = 0
for i in N_Neighbors:
    print('when n_neighbors = ', i)
    clf = KNeighborsClassifier(n_neighbors=i)
    start_time_KNN = time.time()
    # train the model on the training data
    clf.fit(X_train, y_train)
    end_time_KNN = time.time()
    start_time_KNN_test = time.time()
    # predict the class labels for the test data
    y_pred = clf.predict(X_test)
    end_time_KNN_test = time.time()
    # calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of KNN model = ", accuracy * 100, "%")
    if accuracy > best_acc:
        best_acc = accuracy
        best_model = clf
Classification_Accuracy.append(accuracy * 100)
training_time_KNN_model = end_time_KNN - start_time_KNN
Training_Time_Models.append(training_time_KNN_model)
testing_time_KNN_model = end_time_KNN_test - start_time_KNN_test
Testing_Time_Models.append(testing_time_KNN_model)
# **********************************************************
# **********************************************************
# Save the best model to a file using pickle
filename = 'best_knn_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)
# **********************************************************
# **********************************************************


# ================================================================================================
Model_Names.append('GBC_Model')
# Train a Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
start_time_GBC = time.time()
gbc.fit(X_train, y_train)
end_time_GBC = time.time()
training_time_GBC_model = end_time_GBC - start_time_GBC
Training_Time_Models.append(training_time_GBC_model)
start_time_GBC = time.time()
# Make predictions on the test set
y_pred = gbc.predict(X_test)
end_time_GBC = time.time()
testing_time_GBC_model = end_time_GBC - start_time_GBC
Testing_Time_Models.append(testing_time_GBC_model)
# Evaluate the model
print("Accuracy of GradientBoostingClassifier model = ", accuracy_score(y_test, y_pred) * 100, "%")
Classification_Accuracy.append(accuracy_score(y_test, y_pred) * 100)
# **********************************************************
# **********************************************************
# Save the model to a file
filename = 'gbc_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(gbc, file)
# **********************************************************
# **********************************************************
# Creating the bar graph for classification accuracy
plt.figure(figsize=(12, 6))
plt.bar(Model_Names, Classification_Accuracy)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy')
plt.show()

# Creating the bar graph for total training time
plt.figure(figsize=(12, 6))
plt.bar(Model_Names, Training_Time_Models)
plt.xlabel('Models')
plt.ylabel('Time (seconds)')
plt.title('Total Training Time')
plt.show()

# Creating the bar graph for total test time
plt.figure(figsize=(12, 6))
plt.bar(Model_Names, Testing_Time_Models)
plt.xlabel('Models')
plt.ylabel('Time (seconds)')
plt.title('Total Test Time')
plt.show()

# ===============================================================================================
print('=====================================================================================')
# ===================================================================================================