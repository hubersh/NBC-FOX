import numpy as np
import os
import pandas as pd
import pyprind
import pickle

import scrape

from SentimentAnalysis import SentimentAnalysis

import warnings
warnings.filterwarnings("ignore")


class PredictBias:
    def __init__(self, url, modelfile="pickled_model.sav"):
        self.url = url
        with open(modelfile, "rb") as f:
            self.model = pickle.load(f)
        # self.model = pickle.load(open(modelfile, 'rb'))

    def predict_url(self):
        X = scrape.external_call(self.url)
        X = [X]
    
        value_dict = {0: 'Right', 1: "Left"}
        print("We predict that this article is",value_dict[int(self.model.predict(X)[0])],"leaning.")
        print("Left: {}%. Right: {}%".format(int(100 * self.model.predict_proba(X)[0][1]),
                                             int(100 * self.model.predict_proba(X)[0][0])))


if __name__ == "__main__":

    user_input = input("Please enter a url: ")
    pb = PredictBias(user_input)
    pb.predict_url()

