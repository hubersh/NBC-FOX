#!/usr/bin/python3

"""@authors Hunter Hubers and Bryce Hutton
   @Created 2019-03-24
"""

import numpy as np
import os
import pandas as pd
import pyprind

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

import scrape


# Read the data
basepath = "./training_data"
labels = {'left': 1, 'right': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('left', 'right'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[1]]], ignore_index=True)
            pbar.update()
# Change names of columns
df.columns = ['Article', 'Directional-Bias']


# Shuffle the data
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./shuffled_data.csv', index=False)


# Read shuffled data
df = pd.read_csv('./shuffled_data.csv')
print(df.head(10))

porter = PorterStemmer()


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# Create a training and test set.
# This is a simplified approach.
X_train = df.loc[:25, 'Article'].values
y_train = df.loc[:25, 'Directional-Bias'].values
X_test = df.loc[25:, 'Article'].values
y_test = df.loc[25:, 'Directional-Bias'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

stop = stopwords.words('english')

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# This fitting process takes a long time!
gs_lr_tfidf.fit(X_train, y_train)

# print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

# TODO Read https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# TODO Call scrape.external_call() this return will be the article text string that needs to be predicted
X = None  # TODO X should be the data from the new url
gs_lr_tfidf.predict(X)  # TODO Figure out how to pass new data to the prediction function
