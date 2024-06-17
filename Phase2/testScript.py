import pickle

from sklearn.metrics import accuracy_score

from Load_Preprocessing import *
from main import selected_features
print('TestScript Starting...')
data = pd.read_csv('movies-tas-test.csv')
target_map = {'High': 2, 'Intermediate': 1, 'Low': 0}
data['Rate'] = data['Rate'].map(target_map)
X_test = data.iloc[:, 0:19]  # Features
y_test = data['Rate']  # Label
# print(len(X_test))
X_test,y_test = Preprocessing_Train_Test(X_test,y_test,'testscript')
# print(len(X_test))
X_test = X_test[selected_features]
# Load the saved model from file
filename = 'LogisticRegression_model.pkl'
with open(filename, 'rb') as file:
    LogisticRegression_model = pickle.load(file)

y_pred = LogisticRegression_model.predict(X_test)

print('Accuracy of logistic regression model on test script = ', LogisticRegression_model.score(X_test, y_test) * 100, "%")
    
# Load the saved model from file
filename = 'RandomForestClassifier_model.pkl'
with open(filename, 'rb') as file:
    RandomForestClassifier_model = pickle.load(file)

y_pred = RandomForestClassifier_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of randomforestclassifier model on test script = ", accuracy * 100, "%")
    
    
# Load the saved model from file
filename = 'best_svm_model.pkl'
with open(filename, 'rb') as file:
    best_svm_model = pickle.load(file)
    

predictions = best_svm_model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print('Accuracy of best SVM with ', 'model on test script = ', accuracy * 100, "%")

# Load the saved model from file
filename = 'best_knn_model.pkl'
with open(filename, 'rb') as file:
    best_knn_model = pickle.load(file)

y_pred = best_knn_model.predict(X_test)
# calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of best KNN model = on test script ", accuracy * 100, "%")

# Load the saved model from file
filename = 'gbc_model.pkl'
with open(filename, 'rb') as file:
    gbc_model = pickle.load(file)

y_pred = gbc_model.predict(X_test)

# Evaluate the model
print("Accuracy of GradientBoostingClassifier model = on test script ", accuracy_score(y_test, y_pred) * 100, "%")