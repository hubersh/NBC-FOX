#necessary to shuffle csv file so when you seperate training and testing data there won't be a clear split of sites.
import pandas as pd
import numpy as np

df = pd.read_csv('left.csv', encoding= "ISO-8859-1",header=None)
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('shuffled_left.csv', index=False)