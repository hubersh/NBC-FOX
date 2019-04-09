import numpy as np
import os
import pandas as pd
import pyprind
import pickle


from SentimentAnalysis import SentimentAnalysis

import scrape

class PredictBias:
    def __init__(self, url, modelfile="pickled_model.sav"):
        self.url = url
        self.model = pickle.load(open(modelfile, 'rb'))

    def predict_url(self):
        X = scrape.external_call(self.url)
        X = [X]

        print(self.model.predict(X))
        print(self.model.predict_proba(X))

pb = PredictBias("https://www.cnn.com/2019/04/09/politics/donald-trump-mcgahn-tillerson-nielsen/index.html")
pb.predict_url()
