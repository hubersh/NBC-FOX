#!/usr/bin/python3

"""@authors Hunter Hubers and Bryce Hutton
   @Created 2019-03-24
"""

import numpy as np
import os
import pandas as pd
import pyprind
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

import scrape


class SentimentAnalysis:
    def __init__(self):
        self.porter = PorterStemmer()
        self.df = None

    def load_in_data(self):

        # Read the data
        basepath = "./training_data"
        labels = {'left': 1, 'right': 0}
        pbar = pyprind.ProgBar(500)

        self.df = pd.DataFrame()
        for s in ('test', 'train'):
            for l in ('left', 'right'):
                path = os.path.join(basepath, s, l)
                for file in os.listdir(path):
                    with open(os.path.join(path, file), 'r', encoding="ISO-8859-1") as infile:
                        txt = infile.read()
                    self.df = self.df.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()
        # Change names of columns
        self.df.columns = ['Article', 'Directional-Bias']

        # Shuffle the data
        np.random.seed(0)
        self.df = self.df.reindex(np.random.permutation(self.df.index))
        # Used to save and reload shuffled data
        self.df.to_csv('./shuffled_data.csv',  encoding= "ISO-8859-1", index=False)

    def read_in_data(self):

        # Read shuffled data
        self.df = pd.read_csv('./shuffled_data.csv')
        # print(self. df.head(10))

    @staticmethod
    def tokenizer(text):
        return text.split()

    def tokenizer_porter(self, text):
        return [self.porter.stem(word) for word in text.split()]

    def train_data(self):
        # bug: Pickle only works if you put __name__ == '__main__' here
        if __name__ == '__main__':
            # Create a training and test set.
            # This is a simplified approach.
            X_train = self.df.loc[:350, 'Article'].values
            y_train = self.df.loc[:350, 'Directional-Bias'].values
            X_test = self.df.loc[350:, 'Article'].values
            y_test = self.df.loc[350:, 'Directional-Bias'].values

            tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

            stop = stopwords.words('english')

            param_grid = [{'vect__ngram_range': [(1, 1)],
                           'vect__stop_words': [stop, None],
                           'vect__tokenizer': [self.tokenizer, self.tokenizer_porter],
                           'clf__penalty': ['l1', 'l2'],
                           'clf__C': [1.0, 10.0, 100.0]},
                          {'vect__ngram_range': [(1, 1)],
                           'vect__stop_words': [stop, None],
                           'vect__tokenizer': [self.tokenizer, self.tokenizer_porter],
                           'vect__use_idf': [False],
                           'vect__norm': [None],
                           'clf__penalty': ['l1', 'l2'],
                           'clf__C': [1.0, 10.0, 100.0]},
                          ]
            lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])

            gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=2, verbose=1, n_jobs=1)

            # This fitting process takes a long time!
            gs_lr_tfidf.fit(X_train, y_train)
            with open('pickled_model_gs.pkl', 'wb+') as f:
                pickle.dump(gs_lr_tfidf, f)
            pickle.dump(gs_lr_tfidf, open('pickled_model_gs.sav', 'wb'))

            # print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
            print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

            clf = gs_lr_tfidf.best_estimator_
            print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
            with open('pickled_model_clf.sav', 'wb+') as f:
                pickle.dump(gs_lr_tfidf, f)


if __name__ == '__main__':
    sa = SentimentAnalysis()
    sa.read_in_data("./training_data")
    sa.train_data()
