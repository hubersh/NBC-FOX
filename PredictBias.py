import numpy as np
import os
import pandas as pd
import pyprind
import pickle
import sys


import scrape


from SentimentAnalysis import SentimentAnalysis


class PredictBias:
    def __init__(self, url, modelfile="pickled_model.pkl"):
        self.url = url
        with open(modelfile, "rb") as f:
            self.model = pickle.load(f)
       # self.model = pickle.load(open(modelfile, 'rb'))

    def predict_url(self):
        X = scrape.external_call(self.url)
        X = [X]

        print(self.model.predict(X))
        print("Left: {}%. Right: {}%".format(int(100 * self.model.predict_proba(X)[0][1]),
                                             int(100 * self.model.predict_proba(X)[0][1])))


pb = PredictBias(sys.argv[1])
pb.predict_url()