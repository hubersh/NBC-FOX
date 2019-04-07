#!/usr/bin/python3

"""@author Hunter Hubers
   @Created 2019-04-06
"""

import scrape

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelPropagation


# class PredictContent:
#
#     def __init__(self, url):
#
#         scrape.external_call(url)  # Create the temp.txt file
#         self.data = self.load_data("./temp.txt")

#    @staticmethod

def load_data(filename):
    dataset = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(line.strip())
    return dataset


if __name__ == "__main__":

    current_page = load_data("temp.txt")

    left_data = load_data("./training_data/left.txt")
    right_data = load_data("./training_data/right.txt")

    vect = TfidfVectorizer(stop_words="english", max_df=0.8, min_df=10)

    vData = vect.fit_transform(current_page)
    vKeep = vect.transform(left_data)
    vDelete = vect.transform(right_data)

    lpm = LabelPropagation()

    yVals = []
    tagged = 0
    for row in current_page:
        if row in left_data:
            yVals.append(1)
            tagged += 1
        elif row in right_data:
            yVals.append(0)
            tagged += 1
        else:
            yVals.append(-1)

    print("Fitting...")
    lpm.fit(vData.toarray(), yVals)

    for row, vRow in zip(current_page[:30], vData[:30]):
        print(row)
        print("Left: {}%, Right: {}%".format(int(100 * lpm.predict_proba(vRow)[0][1]), int(100 * lpm.predict_proba(vRow)[0][0])))
        print("------------")
