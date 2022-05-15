from razdel import tokenize
import re
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from string import punctuation
import pickle
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("data_train.csv", sep='\t', encoding='utf8', nrows=300000, header=None,
                   usecols=[1, 3], names=['gender', 'text'])

test_data = pd.read_csv("data_test.csv", sep='\t', encoding='utf8', nrows=60000, header=None,
                   usecols=[1, 3], names=['gender', 'text'])


def clear_text(dataframe):
    dataframe['text'].replace(to_replace=r'[0-9]|"|[.,?!%+—-]|(None)|#(\w|\d)*', value='', regex=True, inplace=True)
    dataframe.drop_duplicates(subset=['text'], inplace=True)
    dataframe.reset_index(inplace=True, drop=True)


clear_text(train_data)
clear_text(test_data)

Xtrain = train_data['text']
ytrain = train_data['gender']
Xtest = test_data['text']
ytest = test_data['gender']

del train_data
del test_data


def tokenizer(raw_text):
    return [token.text.lower() for token in tokenize(raw_text) if len(token.text) > 1]

extra_stopwords = ['noun', 'punct', 'verb', 'adp', 'adj', 'pron', 'cconj', 'adv', 'part', 'num', 'nan',
        'person', 'det', 'sconj', 'aux', 'sym', 'intj', 'gt', 'lt', 'amp',
        'мочь', 'очень', 'просто', 'весь', 'это', 'который', 'наш', 'еще', 'ещё',
        'год', 'свой', 'человек', 'твой', 'самый']
noise = stopwords.words('russian') + list(punctuation) + extra_stopwords


vec = TfidfVectorizer(ngram_range=(1, 1), tokenizer=tokenizer, stop_words=noise)
Xtrain = vec.fit_transform(Xtrain)


classifiers = {
    'LogisticRegression'
    'LinearSVC': LinearSVC(),
    'RandomForestClassifier': RandomForestClassifier()
}

for name, clf in classifiers.items():
        clf.fit(Xtrain, ytrain)
        pred = clf.predict(vec.transform(Xtest))
        print(f"Classification report for {name}: {classification_report(pred, ytest)}")
        print(f"f1score: {f1_score(ytest, pred, average='weighted')}")


# Target data classification
df_love = pd.read_csv('love_dataset_copy.csv', sep='\t', usecols=[0,1], names=['gender', 'text'], encoding='utf8', header=None)
df_politics = pd.read_csv('politics_dataset_copy.csv', sep='\t', usecols=[0, 1], names=['gender', 'text'], encoding='utf8', header=None)
df_family = pd.read_csv('family_dataset_copy.csv', sep='\t', usecols=[0, 1], names=['gender', 'text'], encoding='utf8', header=None)


def classify_target_data(df):
    Xtest = df['text']
    ytest = df['gender']
    for name, clf in classifiers.items():
        pred = clf.predict(vec.transform(Xtest))
        print(classification_report(pred, ytest))
        print(f"f1score: {f1_score(ytest, pred, average='weighted')}")
