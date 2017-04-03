import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs


from pandas import read_csv, DataFrame, Series
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
import seaborn as sns

data = read_csv('drivers_5000.csv')

data.at[data['Accidents'] > 0, 'AccidentsBin'] = 1
data.at[data['Accidents'] == 0, 'AccidentsBin'] = 0
 
X = data[['HomeLat', 'HomeLng']].values

#Compute clustering with Means
batch_size = 45
n_clusters=50
k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0
#Compute clustering with MiniBatchKMeans
mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)
t0 = time.time()
mbk.fit(X)
t_mini_batch = time.time() - t0
#Plot result
pal = sns.diverging_palette(240, 10, n=n_clusters, as_cmap=True)

plt.scatter(X[:,0], X[:,1], c=k_means.labels_, cmap=pal)
plt.colorbar()
plt.show()

plt.scatter(X[:,0], X[:,1], c=mbk.labels_, cmap=pal)
plt.colorbar()
plt.show()
