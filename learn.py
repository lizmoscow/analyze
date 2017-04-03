from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import ensemble
from  sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def PrintTest(classifier):
    y_pred = classifier.predict(X_test)
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(X_test)
        y_proba = proba[:,1]
        print ('log_loss: ', metrics.log_loss(y_test, proba))
    else:  # use decision function
        prob_pos = classifier.decision_function(X_test)
        y_proba = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())    
        
    print ('brier_score_loss: ', metrics.brier_score_loss(y_test, y_proba))    
    print(metrics.classification_report(y_test, y_pred))

sns.set(color_codes=True)
data = pd.read_csv('drivers_5000.csv')

data.at[data['Accidents'] > 0, 'AccidentsBin'] = 1
data.at[data['Accidents'] == 0, 'AccidentsBin'] = 0
 
XX = data[['Age', 'Experience', 'PreviousAccidents', 'RouteDistance', 'Distance', 'HomeLat', 'HomeLng', 'WorkLat', 'WorkLng']]
y = data['AccidentsBin']

#home = data[['HomeLat', 'HomeLng']].values
#batch_size = 45
#n_clusters=50
#mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size, n_init=10, max_no_improvement=10, verbose=0)
#mbk.fit(XX)

#HomeClusters = pd.get_dummies(mbk.labels_, prefix='Home')
#XX = pd.concat([XX, HomeClusters], axis=1)
#XX['Home'] = mbk.labels_

# standardize the data attributes
X = preprocessing.scale(XX)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 11)

print ("kNN")
#f_neighbors_array = [1, 3, 5, 7, 10, 15, 25]
#knn = KNeighborsClassifier()
#grid = GridSearchCV(knn, param_grid={'n_neighbors': f_neighbors_array})
#grid.fit(X_train, y_train)
#
#best_cv_err = 1 - grid.best_score_
#best_n_neighbors = grid.best_estimator_.n_neighbors
#print (best_cv_err, best_n_neighbors)
#
#knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn = KNeighborsClassifier(25)
knn.fit(X_train, y_train)
PrintTest(knn)

print ("SVC")
#C_array = np.logspace(-2, 2, num=3)
#gamma_array = np.logspace(-3, 2, num=3)
svc = SVC(kernel='rbf')
#grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array})
#grid.fit(X_train, y_train)
#print ('CV error    = ', 1 - grid.best_score_)
#print ('best C      = ', grid.best_estimator_.C)
#print ('best gamma  = ', grid.best_estimator_.gamma)

#svc = SVC(kernel='rbf', C=grid.best_estimator_.C, gamma=grid.best_estimator_.gamma)
svc.fit(X_train, y_train)
PrintTest(svc)

print ("Forest")
rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)
PrintTest(rf)

print ("Logistic Regression")
lr = LogisticRegression()
lr.fit(X_train, y_train)
PrintTest(lr)

print ("Gaussian Process Classifier")
kernel = 1.0 * RBF([1.0])
gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel)
gpc_rbf_isotropic.fit(X_train, y_train)
PrintTest(gpc_rbf_isotropic)

