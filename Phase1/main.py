import time
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from preprocessing import *
from MyLabelEncoder import *
from Load_Preprocessing import *



df = pd.read_csv('movies-regression-dataset.csv')

num_nulls_homepage = df["homepage"].isnull().sum()
most_frequent_value_id = df["id"].value_counts().idxmax()
num_values_id = df["id"].value_counts()[most_frequent_value_id]
num_values_status = df["status"].value_counts()['Released']
# print(num_nulls)
# print(num_values)
if (num_nulls_homepage > df.shape[0] / 3 and num_values_id == 1 and num_values_status > len(df['status']) / 2):
    df.drop(['homepage', 'id', 'status'], axis=1, inplace=True)
    #Drop rows
df.dropna(how='any', inplace=True)

# Split the data to training and testing sets
#=============================================
X = df.iloc[:, 0:16]  # Features
Y = df['vote_average']  # Label
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True , random_state=100)
#=========================================================================================================================================
#Processing on Training Data
#============================
X_train_columns = X_train.iloc[:, 0:16]  # Features
# print(X_train_columns)
# print(X_train)
# print(X_test)
scaler_budget = MinMaxScaler()
X_train['budget'] = scaler_budget.fit_transform(X_train[['budget']])
X_train["budget"].replace(0, X_train["budget"].median(), inplace=True)  # any row in budget = 0 replace it by median
median = np.median(X_train["budget"])
std = np.std(X_train["budget"])
outliers = (X_train["budget"] - median).abs() > 3 * std
X_train.loc[outliers, "budget"] = median
print(X_train['budget'])

##############################################################################3
scaler_runtime = MinMaxScaler()
X_train['runtime'] = replace_zeros__scale(X_train, 'runtime', scaler_runtime)
YY = np.expand_dims(X_train["runtime"], axis=1)
X_train["genres"] = Dict_Column_genres(X_train["genres"], YY)
#print(X_train["genres"])

X_train["keywords"] = X_train["keywords"].apply(lambda x: [i["name"] for i in eval(x)])
lbl_keyword = MyLabelEncoder(-1)
X_train["keywords"] = Feature_Encoder(X_train_columns, 'keywords', lbl_keyword)
scaler_keywords = MinMaxScaler()
X_train['keywords'] = scaler_keywords.fit_transform(X_train[['keywords']])
X_train["keywords"].replace(0, X_train["keywords"].median(), inplace=True)
# print(X_train['keywords'])

lbl_orig_lang = MyLabelEncoder(-1)
X_train["original_language"] = Feature_Encoder(X_train_columns, 'original_language', lbl_orig_lang)
# print(X_train["original_language"])

X_train['original_title'] = string_column(X_train['original_title'])
lbl_orig_title = MyLabelEncoder(-1)
X_train["original_title"] = Feature_Encoder(X_train_columns, 'original_title', lbl_orig_title)
scaler_orig_title = MinMaxScaler()
X_train['original_title'] = scaler_orig_title.fit_transform(X_train[['original_title']])
# print(X_train['original_title'])

X_train['tagline'] = string_column(X_train['tagline'])
lbl_tagline = MyLabelEncoder(-1)
X_train["tagline"] = Feature_Encoder(X_train_columns, 'tagline', lbl_tagline)
scaler_tagline = MinMaxScaler()
X_train['tagline'] = scaler_tagline.fit_transform(X_train[['tagline']])
# print(X_train['tagline'])

X_train['overview'] = string_column(X_train['overview'])
lbl_overview = MyLabelEncoder(-1)
X_train["overview"] = Feature_Encoder(X_train_columns, 'overview', lbl_overview)
scaler_overview = MinMaxScaler()
X_train['overview'] = scaler_overview.fit_transform(X_train[['overview']])
# print(X_train['overview'])

X_train["production_countries"] = Dict_Column_countries(X_train["production_countries"], YY)
# print(X_train['production_countries'])

X_train["production_companies"] = Dict_Column_companies(X_train["production_companies"], YY)
# print(X_train['production_companies'])

date1 = datetime.strptime('2/1/1970', "%m/%d/%Y")
ll = []
for i in X_train['release_date']:
    ii = datetime.strptime(i, "%m/%d/%Y")
    date_object = max(ii, date1)  # example date string in the format "%m/%d/%Y
    ll.append(time.mktime(date_object.timetuple()))
ll = (ll - np.min(ll)) / (np.max(ll) - np.min(ll))
X_train["release_date"] = ll.copy()
# print(X_train['release_date'])

scaler_viewercount = MinMaxScaler()
X_train['viewercount'] = scaler_viewercount.fit_transform(X_train[['viewercount']])
median = np.median(X_train["viewercount"])
std = np.std(X_train["viewercount"])
outliers = (X_train["viewercount"] - median).abs() > 3 * std
X_train.loc[outliers, "viewercount"] = median
# print(X_train['viewercount'])

scaler_revenue = MinMaxScaler()
X_train["revenue"] = scaler_revenue.fit_transform(X_train[["revenue"]])
X_train["revenue"].replace(0, X_train["revenue"].median(), inplace=True)
std = np.std(X_train["revenue"])
outliers = (X_train["revenue"] - median).abs() > 3 * std
X_train.loc[outliers, "revenue"] = median
# print(X_train["revenue"])

X_train["spoken_languages"] = X_train["spoken_languages"].apply(lambda x: [i["iso_639_1"] for i in eval(x)])
lbl_spoken_lang = MyLabelEncoder(-1)
X_train["spoken_languages"] = Feature_Encoder(X_train_columns, 'spoken_languages', lbl_spoken_lang)
scaler_spoken_lang = MinMaxScaler()
X_train['spoken_languages'] = scaler_spoken_lang.fit_transform(X_train[['spoken_languages']])
X_train["spoken_languages"].replace(0, X_train["spoken_languages"].median(), inplace=True)
median = np.median(X_train["spoken_languages"])
std = np.std(X_train["spoken_languages"])
outliers = (X_train["spoken_languages"] - median).abs() > 3 * std
X_train.loc[outliers, "spoken_languages"] = median
# print(X_train["spoken_languages"])

scaler_vote_count = MinMaxScaler()
X_train['vote_count'] = replace_zeros__scale(X_train, 'vote_count', scaler_vote_count)
median = np.median(X_train["vote_count"])
std = np.std(X_train["vote_count"])
outliers = (X_train["vote_count"] - median).abs() > 3 * std
X_train.loc[outliers, "vote_count"] = median
# print(X_train['vote_count'])

X_train['title'] = string_column(X_train['title'])
lbl_title = MyLabelEncoder(-1)
X_train["title"] = Feature_Encoder(X_train_columns, 'title', lbl_title)
scaler_title = MinMaxScaler()
X_train['title'] = scaler_title.fit_transform(X_train[['title']])
# print(X_train['title'])

# ============================================================================================================================
#Processing on Testing Data
#============================
X_test_columns = X_test.iloc[:, 0:16]  # Features

X_test['budget'] = scaler_budget.transform(X_test[['budget']])
X_test["budget"].replace(0, X_test["budget"].median(), inplace=True)
median = np.median(X_test["budget"])
std = np.std(X_test["budget"])
outliers = (X_test["budget"] - median).abs() > 3 * std
X_test.loc[outliers, "budget"] = median
# print(X_test['budget'])

X_test['runtime'] = replace_zeros__scale_testing(X_test, 'runtime', scaler_runtime)
YY = np.expand_dims(X_test["runtime"], axis=1)
X_test["genres"] = Dict_Column_genres(X_test["genres"], YY)
# print(X_test["genres"])

X_test["keywords"] = X_test["keywords"].apply(lambda x: [i["name"] for i in eval(x)])
X_test['keywords'] = Feature_Encoder_testing(X_test_columns, 'keywords', lbl_keyword)
X_test['keywords'] = scaler_keywords.transform(X_test[['keywords']])
X_test["keywords"].replace(0, X_test["keywords"].median(), inplace=True)
# print(X_test['keywords'])

X_test["original_language"] = Feature_Encoder_testing(X_test_columns, 'original_language', lbl_orig_lang)
# print(X_test["original_language"])

X_test['original_title'] = string_column(X_test['original_title'])
X_test["original_title"] = Feature_Encoder_testing(X_test_columns, 'original_title', lbl_orig_title)
X_test['original_title'] = scaler_orig_title.transform(X_test[['original_title']])
# print(X_test['original_title'])

X_test['tagline'] = string_column(X_test['tagline'])
X_test["tagline"] = Feature_Encoder_testing(X_test_columns, 'tagline', lbl_tagline)
X_test['tagline'] = scaler_tagline.transform(X_test[['tagline']])
# print(X_test['tagline'])

X_test['overview'] = string_column(X_test['overview'])
X_test["overview"] = Feature_Encoder_testing(X_test_columns, 'overview', lbl_overview)
X_test['overview'] = scaler_overview.transform(X_test[['overview']])
# print(X_test['overview'])

X_test["production_countries"] = Dict_Column_countries(X_test["production_countries"], YY)
# print(X_test['production_countries'])

X_test["production_companies"] = Dict_Column_companies(X_test["production_companies"], YY)
# print(X_test['production_companies'])

date1 = datetime.strptime('2/1/1970', "%m/%d/%Y")
ll = []
for i in X_test['release_date']:
    ii = datetime.strptime(i, "%m/%d/%Y")
    date_object = max(ii, date1)  # example date string in the format "%m/%d/%Y
    ll.append(time.mktime(date_object.timetuple()))
ll = (ll - np.min(ll)) / (np.max(ll) - np.min(ll))
X_test["release_date"] = ll.copy()
# print(X_test['release_date'])

X_test['viewercount'] = scaler_viewercount.transform(X_test[['viewercount']])
median = np.median(X_test["viewercount"])
std = np.std(X_test["viewercount"])
outliers = (X_test["viewercount"] - median).abs() > 3 * std
X_test.loc[outliers, "viewercount"] = median
# print(X_test['viewercount'])

X_test["revenue"] = scaler_revenue.transform(X_test[["revenue"]])
X_test["revenue"].replace(0, X_test["revenue"].median(), inplace=True)  # any row in budget = 0 replace it by median
median = np.median(X_test["revenue"])
std = np.std(X_test["revenue"])
outliers = (X_test["revenue"] - median).abs() > 3 * std
X_test.loc[outliers, "revenue"] = median
# print(X_test["revenue"])

X_test["spoken_languages"] = X_test["spoken_languages"].apply(lambda x: [i["iso_639_1"] for i in eval(x)])
X_test["spoken_languages"] = Feature_Encoder_testing(X_test_columns, 'spoken_languages', lbl_spoken_lang)
X_test['spoken_languages'] = scaler_spoken_lang.transform(X_test[['spoken_languages']])
X_test["spoken_languages"].replace(0, X_test["spoken_languages"].median(), inplace=True)
median = np.median(X_test["spoken_languages"])
std = np.std(X_test["spoken_languages"])
outliers = (X_test["spoken_languages"] - median).abs() > 3 * std
X_test.loc[outliers, "spoken_languages"] = median
# print(X_test["spoken_languages"])

X_test['vote_count'] = replace_zeros__scale_testing(X_test, 'vote_count', scaler_vote_count)
median = np.median(X_test["vote_count"])
std = np.std(X_test["vote_count"])
outliers = (X_test["vote_count"] - median).abs() > 3 * std
X_test.loc[outliers, "vote_count"] = median
# print(X_test['vote_count'])

X_test['title'] = string_column(X_test['title'])
X_test["title"] = Feature_Encoder_testing(X_test_columns, 'title', lbl_title)
X_test['title'] = scaler_title.transform(X_test[['title']])
# print(X_test['title'])

# ================================================================================================
# apply the Correlation ---------------------------------------------------
data = pd.concat([X_train, y_train], axis=1)
corr = data.corr()
topFeatures = corr.index[abs(corr['vote_average'] > 0.1)]
topcorr = data[topFeatures].corr()
# print(topcorr)
sns.heatmap(topcorr, annot=True)
plt.show()
topFeatures = topFeatures.delete(-1)
X_train = X_train[topFeatures]
X_test = X_test[topFeatures]
#=======================================================================================================
# apply the linearRegression ---------------------------------------------------
linearRegression = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=1)

linearRegression.fit(X_train, y_train)
y_pred = linearRegression.predict(X_test)
# y_pred_train = linearRegression.predict(X_train)
# calculate MSE
MSEValue = mean_squared_error(y_test, y_pred, multioutput="uniform_average")
print("Mean Squared Error for Linear Regression Model : ", MSEValue)

# -------------------------------------------------------
print("r2_score for linear regression model: ")
print(r2_score(y_test, y_pred) * 100)

for col in X_test.columns:
    plt.scatter(X_test[col], y_test)
    # set the title and labels
    plt.title("Linear Regression Model for " + col)
    plt.xlabel(col)
    plt.ylabel("Vote Average")
    # plot the regression line
    plt.plot(X_test[col], y_pred, color="red", linewidth=2)
    # show the plot
    plt.show()
# -------------------------------------------------------

# apply polynomialRegression ---------------------------------------------------
poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

# calculate MSE
print('Mean Square Error for polynomial regression model', mean_squared_error(y_test, prediction))
print("r2_score polynomial regression model: ")
print(r2_score(y_test, prediction) * 100)
for col in X_test.columns:
    plt.scatter(X_test[col], y_test)
    # set the title and labels
    plt.title("Polynomial Regression Model for " + col)
    plt.xlabel(col)
    plt.ylabel("Vote Average")
    # plot the regression line
    plt.plot(X_test[col], y_pred, color="red", linewidth=2)
    # show the plot
    plt.show()
# --------------------------------------------------------------------------------------------------
# Create Ridge Regression model
alpha = 0.1
ridge = Ridge(alpha=alpha)

# Fit the model to the training data
ridge.fit(X_train, y_train)

# Make predictions on the test data
y_pred = ridge.predict(X_test)

# Calculate the Mean Squared Error (MSE) of the predictions
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate the R-squared (R2) score of the model
r2_score = ridge.score(X_test, y_test)
print("R-squared Score for ridge regression:", r2_score * 100)

for col in X_test.columns:
    plt.scatter(X_test[col], y_test)
    # set the title and labels
    plt.title("Ridge Regression Model for " + col)
    plt.xlabel(col)
    plt.ylabel("Vote Average")
    # plot the regression line
    plt.plot(X_test[col], y_pred, color="red", linewidth=2)
    # show the plot
    plt.show()
# ========================================================================

# this part is for the testscripting phase
# ================================================================
print('TestScript Starting...')
data_test_script = pd.read_csv('movies-tas-test.csv')
X_test_script = data_test_script.iloc[:, 0:19]  # Features
y_test_script = data_test_script['vote_average']  # Label
# print(len(X_test))
X_test_script,y_test_script = Preprocessing_Train_Test(X_test_script,y_test_script,'testscript')
X_test_script = X_test_script[topFeatures]
# linearregression
y_pred_test_script_LR = linearRegression.predict(X_test_script)
MSEValue = mean_squared_error(y_test_script, y_pred_test_script_LR, multioutput="uniform_average")
print("Test Script for Linear Regression Model : ", MSEValue)
print("r2_score for linear regression model: ")
print(r2_score(y_test, y_pred))

# polynomialregression
y_pred_test_script_PR = poly_model.predict(poly_features.transform(X_test_script))
MSEValue = mean_squared_error(y_test_script, y_pred_test_script_PR, multioutput="uniform_average")
print("Test Script for Polynomial Regression Model : ", MSEValue)
print("r2_score polynomial regression model: ")
print(r2_score(y_test, prediction))
# ridgeregression
y_pred_test_script_Ridge = ridge.predict(X_test_script)
MSEValue = mean_squared_error(y_test_script, y_pred_test_script_Ridge, multioutput="uniform_average")
print("Test Script for Ridge Regression Model : ", MSEValue)
r2_score = ridge.score(X_test, y_test)
print("R-squared Score for ridge regression:", r2_score)