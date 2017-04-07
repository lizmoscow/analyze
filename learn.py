import os
# Keras backend settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5" # or coment it to use GPU by default
os.environ["KERAS_BACKEND"] = "theano" # or coment it to use tensorflow by default

# Theano settings
os.environ["THEANO_FLAGS"] = "device=cpu" # or "device=cuda"

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne import nonlinearities, updates, objectives
from lasagne.layers import get_all_params

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KNN = 0
SV = 0
RF = 1
GP = 0
MLP = 1
KERAS = 1
LASAGNE = 1

data = pd.read_csv('drivers_250000.csv')

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

def PrintTest(classifier, X_test=X_test, y_test=y_test):
    y_pred = classifier.predict(X_test)
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(X_test)
        if proba.shape[1] == 2:
            y_proba = proba[:,1]
        else:
            y_proba = proba
        print ('log_loss: ', metrics.log_loss(y_test, proba))
    else:  # use decision function
        prob_pos = classifier.decision_function(X_test)
        y_proba = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())    
        
    print ('brier_score_loss: ', metrics.brier_score_loss(y_test, y_proba))    
    print(metrics.classification_report(y_test, y_pred))

print ("Logistic Regression")
lr = LogisticRegression()
lr.fit(X_train, y_train)
PrintTest(lr)

if KNN == 1:
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

if SV == 1:
    print ("SVC")
    #C_array = np.logspace(-2, 2, num=3)
    #gamma_array = np.logspace(-3, 2, num=3)
    svc = SVC(kernel='rbf', random_state = 11)
    #grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array})
    #grid.fit(X_train, y_train)
    #print ('CV error    = ', 1 - grid.best_score_)
    #print ('best C      = ', grid.best_estimator_.C)
    #print ('best gamma  = ', grid.best_estimator_.gamma)
    
    #svc = SVC(kernel='rbf', C=grid.best_estimator_.C, gamma=grid.best_estimator_.gamma)
    svc.fit(X_train, y_train)
    PrintTest(svc)

if RF == 1:
    print ("Forest")
    rf = ensemble.RandomForestClassifier(n_estimators=500, n_jobs = -1, random_state=11, verbose=1)
    rf.fit(X_train, y_train)
    PrintTest(rf)
    
if GP == 1:
    print ("Gaussian Process Classifier")
    kernel = 1.0 * RBF([1.0])
    gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel)
    gpc_rbf_isotropic.fit(X_train, y_train)
    PrintTest(gpc_rbf_isotropic)

if MLP == 1:
    print ("MLP (adam)")
    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(12, 6), random_state=11, verbose=1)
    mlp.fit(X_train, y_train)
    PrintTest(mlp)

    print ("MLP (lbfgs)")
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 6), random_state=11, verbose=1)
    mlp.fit(X_train, y_train)
    PrintTest(mlp)

if KERAS == 1:
    print ("Keras")
    def CreateModel():
    	model = Sequential()
    	model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
    	model.add(Dense(6, activation='relu'))
    	model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001)))
    	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
    	return model
    
    model = KerasClassifier(build_fn=CreateModel, epochs=50, batch_size=100, verbose=1)
    model.fit(X_train.astype(np.float32), y_train.values.astype(np.int32))
    print("\n")
    PrintTest(model, X_test.astype(np.float32), y_test.values.astype(np.int32))

if LASAGNE == 1:
    print ("Lasagne")
    layers0 = [
        (InputLayer, {'shape': (None, X.shape[1])}),
        (DenseLayer, {'num_units': 12}),
        (DenseLayer, {'num_units': 6}),
        (DenseLayer, {'num_units': 1, 'nonlinearity': nonlinearities.sigmoid}),
    ]
    
    ls = NeuralNet(
        layers=layers0,
        max_epochs=50,
    
        update=updates.adam,
    
        objective_l2=0.001,
        objective_loss_function=objectives.binary_crossentropy,
    
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
    )
    
    ls.fit(X_train.astype(np.float32), y_train.values.astype(np.int32))
    print("\n")
    PrintTest(ls, X_test.astype(np.float32), y_test.values.astype(np.int32))
