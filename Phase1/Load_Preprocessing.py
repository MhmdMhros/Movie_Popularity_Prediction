import pandas as pd

from preprocessing import *
import time
from datetime import datetime
def Preprocessing_Train_Test(X_train,y_train,condition):
    num_nulls_homepage = X_train["homepage"].isnull().sum()
    most_frequent_value_id = X_train["id"].value_counts().idxmax()
    num_values_id = X_train["id"].value_counts()[most_frequent_value_id]
    num_values_status = X_train["status"].value_counts()['Released']
    # print(num_nulls)
    # print(num_values)
    if (num_nulls_homepage > X_train.shape[0] / 3 and num_values_id == 1 and num_values_status > len(X_train['status']) / 2):
        X_train.drop(['homepage', 'id', 'status'], axis=1, inplace=True)
    df = pd.concat([X_train,y_train],axis=1)
    if condition == 'testscript':
        df['budget'].fillna(df['budget'][0], inplace=True)
        df['genres'].fillna(df['genres'][0], inplace=True)
        df['keywords'].fillna(df['keywords'][0], inplace=True)
        df['original_language'].fillna(df['original_language'][0], inplace=True)
        df['original_title'].fillna(df['original_title'][0], inplace=True)
        df['overview'].fillna(df['overview'][0], inplace=True)
        df['viewercount'].fillna(df['viewercount'][0], inplace=True)
        df['production_companies'].fillna(df['production_companies'][0], inplace=True)
        df['production_countries'].fillna(df['production_countries'][0], inplace=True)
        df['release_date'].fillna(df['release_date'][0], inplace=True)
        df['revenue'].fillna(df['revenue'][0], inplace=True)
        df['runtime'].fillna(df['runtime'][0], inplace=True)
        df['spoken_languages'].fillna(df['spoken_languages'][0], inplace=True)
        df['tagline'].fillna(df['tagline'][0], inplace=True)
        df['title'].fillna(df['title'][0], inplace=True)
        df['vote_count'].fillna(df['vote_count'][0], inplace=True)
    else:
        df.dropna(how='any', inplace=True)
    X_train = df.iloc[:, 0:16]
    y_train = df['vote_average']
    # df.fillna(df.mean(), inplace=True)
    # Processing on Training Data
    # ============================
    X_train_columns = X_train.iloc[:, 0:16]  # Features
    # print(X_train)
    # print(X_test)
    scaler_budget = MinMaxScaler()
    X_train['budget'] = scaler_budget.fit_transform(X_train[['budget']])
    X_train["budget"].replace(0, X_train["budget"].median(), inplace=True)  # any row in budget = 0 replace it by median
    median = np.median(X_train["budget"])
    std = np.std(X_train["budget"])
    outliers = (X_train["budget"] - median).abs() > 3 * std
    X_train.loc[outliers, "budget"] = median
    # print(X_train['budget'])

    scaler_vote_count = MinMaxScaler()
    X_train['vote_count'] = replace_zeros__scale(X_train, 'vote_count', scaler_vote_count)
    median = np.median(X_train["vote_count"])
    std = np.std(X_train["vote_count"])
    outliers = (X_train["vote_count"] - median).abs() > 3 * std
    X_train.loc[outliers, "vote_count"] = median
    # print(X_train['vote_count'])
    scaler_runtime = MinMaxScaler()
    X_train['runtime'] = replace_zeros__scale(X_train, 'runtime', scaler_runtime)
    YY = np.expand_dims(X_train["vote_count"], axis=1)
    X_train["genres"] = Dict_Column_genres(X_train["genres"], YY)
    # print(X_train["genres"])

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


    X_train['title'] = string_column(X_train['title'])
    lbl_title = MyLabelEncoder(-1)
    X_train["title"] = Feature_Encoder(X_train_columns, 'title', lbl_title)
    scaler_title = MinMaxScaler()
    X_train['title'] = scaler_title.fit_transform(X_train[['title']])
    # print(X_train['title'])
    return X_train,y_train