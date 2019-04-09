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
        print(self.model.decision_function(X))

pb = PredictBias("https://www.foxnews.com/us/arrests-announced-in-murder-for-hire-plot-that-killed-the-wrong-person")
pb.predict_url()