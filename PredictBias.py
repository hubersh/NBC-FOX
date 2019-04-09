import numpy as np
import os
import pandas as pd
import pyprind
import pickle
import SentimentAnalysis


from SentimentAnalysis import tokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

import scrape

class PredictBias:
    def __init__(self, url, modelfile="pickled_model.sav"):
        self.url = url
        self.model = pickle.load(open(modelfile, 'rb'))

    def predict_url(self):
        X = scrape.external_call(self.url)

        # TODO create temporary file to send to predict, as it needs a file not a string. Or work with streams for it.
        print(self.model.gs_lr_tfidf.predict())  # TODO Figure out how to pass new data to the prediction function

pb = PredictBias("https://www.reuters.com/article/us-usa-immigration-asylum/u-s-judge-halts-trump-policy-of-returning-asylum-seekers-to-mexico-idUSKCN1RK2E6")
pb.predict_url()