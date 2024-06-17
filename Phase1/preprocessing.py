import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import re

import nltk
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.datasets import load_files

from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn, stopwords
from nltk.corpus import stopwords
import string
from MyLabelEncoder import *

def Feature_Encoder(df, X, lbl):
    #lbl.fit(list(df[X]))
    df[X] = lbl.fit_transform(list(df[X]))
    return df[X]


def Feature_Encoder_testing(df, X, lbl):
    df[X] = lbl.transform(list(df[X]))
    return df[X]


def replace_zeros__scale(df, X, scaler):
    # df[X] = scaler.fit_transform(df[[X]])
    scaler.fit(df[[X]])
    df[X] = scaler.transform(df[[X]])
    df[X] = df[X].replace(0, df[X].median())
    return df[X]


def replace_zeros__scale_testing(df, X, scaler):  # not sure about having another on for testing
    df[X] = scaler.transform(df[[X]])
    df[X] = df[X].replace(0, df[X].median())
    return df[X]


def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


def Dict_Column_genres(X, Y):
    # Split the genres column into separate lists of genre IDs
    X = X.apply(lambda x: [int(i["id"]) for i in eval(x)])
    unique_list = X.explode().unique()
    dict = {}
    for i in unique_list:
        dict[i] = [0.0, 0.0]
    index = -1
    for i in X:  # loops on geners rows
        index += 1
        for j in unique_list:  # loops on unique_ids
            if j in i:
                dict[j][0] += 1
                dict[j][1] += Y[index] / len(i)
    dic = {}
    for dd in dict:
        dic[dd] = dict[dd][1] / dict[dd][0]

    s = 0
    ll = []
    i = 0
    for g in X:
        if len(g) != 0:
            for j in g:
                s += dic[j]

            s = float(s / len(g))
            ll.insert(i, s)
            i += 1
            s = 0
        else:
            ll.insert(i, 0)
            i += 1
    return ll


def Dict_Column_countries(X, Y):
    X = X.apply(
        lambda x: [i["iso_3166_1"] for i in eval(x)])
    unique_list = X.explode().unique()
    # print(unique_list)

    dict = {}
    for i in unique_list:
        dict[i] = [0.0, 0.0]

    index = -1
    for i in X:  # loops on geners rows
        index += 1
        for j in unique_list:  # loops on unique_ids
            if j in i:
                dict[j][0] += 1
                dict[j][1] += Y[index] / len(i)

    # print(dict)

    dic = {}
    for dd in dict:
        if dict[dd][0] != 0:
            dic[dd] = dict[dd][1] / dict[dd][0]
        else:
            dic[dd] = 0
    # print(dic)

    s = 0
    ll = []
    for g in X:
        if len(g) != 0:
            for j in g:
                s += dic[j]

            s = float(s / len(g))
            ll.append(s)
            s = 0
        else:
            ll.append(0)
    X = ll.copy()
    return X


def Dict_Column_companies(X, Y):
    X = X.apply(lambda x: [i["id"] for i in eval(x)])
    unique_list = X.explode().unique()

    dict = {}
    for i in unique_list:
        dict[i] = [0.0, 0.0]

    index = -1
    for i in X:  # loops on geners rows
        index += 1
        for j in unique_list:  # loops on unique_ids
            if j in i:
                dict[j][0] += 1
                dict[j][1] += Y[index] / len(i)

    dic = {}
    for dd in dict:
        if dict[dd][0] != 0:
            dic[dd] = dict[dd][1] / dict[dd][0]
        else:
            dic[dd] = 0

    s = 0
    ll = []
    for g in X:
        if len(g) != 0:
            for j in g:
                s += dic[j]

            s = float(s / len(g))
            ll.append(s)
            s = 0
        else:
            ll.append(0)

    X = ll.copy()
    return X

def string_column(X):
    y=0
    ll = []
    for x in X:
        # x = re.sub(r'\S+@\S+', '', x)
        x = re.sub(r'[^a-zA-Z\s]+', '', x)
        x = re.sub(r'[0-9]*', '', x)
        stop_words = set(stopwords.words('english'))
        words = nltk.word_tokenize(x)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_text = " ".join(filtered_words)
        stemmer = PorterStemmer()
        filtered_text = filtered_text.translate(str.maketrans("", "", string.punctuation)).lower()
        words = filtered_text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        filtered_text = " ".join(stemmed_words)
        ll.append(filtered_text)
    X = ll.copy()
    return X

