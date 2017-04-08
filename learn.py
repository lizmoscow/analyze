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
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

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

KNN = 1
SV = 0
BAYES = 1
RF = 1
GP = 0
MLP = 1
KERAS = 1
LASAGNE = 1

data = pd.read_csv('drivers_50000.csv')

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

print ("Dummy")
y_dummy = np.empty_like(y_test)
y_dummy[:] = np.average(y_test.values)
average_probability = np.average(y_dummy)
print ('probability: ', average_probability)    
print ('brier_score_loss: ', metrics.brier_score_loss(y_test, y_dummy))    
print ('profit: ', round(np.sum(y_dummy - y_test) / y_test.shape[0], 4))

def PrintTest(est, calibrate=True):
    """Plot calibration curve for est w/o and with calibration. """
    clfs = [(est, 'Original')]
    if calibrate:
        isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')
        sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')
        clfs = clfs + [(isotonic, 'Isotonic'), (sigmoid, 'Sigmoid')]    

    plt.figure(1, figsize=(15, 15))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.plot([average_probability, average_probability], [0, 1], "k-", label="Average")
    for clf, name in clfs:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:,-1]
            print ('log_loss: ', metrics.log_loss(y_test, prob_pos))
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = metrics.brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.4f" % (clf_score))
        print ("\tProfit: %1.4f" % (np.average(prob_pos - y_test)))
        print(metrics.classification_report(y_test, clf.predict(X_test)))
        Compete(prob_pos, margin1 = 1)
        
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=50)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.4f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=50, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

def Compete(y_proba1, y_proba2 = y_dummy, margin1 = 1, margin2 = 1):
    premium1 = y_proba1 * margin1
    premium2 = y_proba2 * margin2
    selector1 = premium1 < premium2
    selector2 = premium2 < premium1
    profit1 = premium2 - y_test
    profit2 = premium1 - y_test
    print ('1. Profit: ', np.sum(np.select([selector1], [profit1])) / np.sum(selector1), ', deals: ', np.sum(selector1) / selector1.shape[0])
    print ('2: Profit: ', np.sum(np.select([selector2], [profit2])) / np.sum(selector2), ', deals: ', np.sum(selector2) / selector2.shape[0])
    
print ("Logistic Regression")
lr = LogisticRegression()
lr.fit(X_train, y_train)
PrintTest(lr)

if KNN == 1:
    print ("kNN")
    knn = KNeighborsClassifier(100)
    PrintTest(knn)

if SV == 1:
    print ("SVC")
    svc = SVC(kernel='rbf', random_state = 11)
    PrintTest(svc)

if BAYES == 1:
    print ("Naive Bayes")
    nb = GaussianNB()
    PrintTest(nb)
   
if RF == 1:
    print ("Forest")
    rf = ensemble.RandomForestClassifier(n_estimators=500, n_jobs = -1, random_state=11, verbose=0)
    PrintTest(rf)
    
if GP == 1:
    print ("Gaussian Process Classifier")
    kernel = 1.0 * RBF([1.0])
    gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel)
    PrintTest(gpc_rbf_isotropic)

if MLP == 1:
    print ("MLP (adam)")
    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(12, 6), random_state=11, verbose=0)
    PrintTest(mlp)

    print ("MLP (lbfgs)")
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 6), random_state=11, verbose=0)
    PrintTest(mlp)

X_test = X_test.astype(np.float32)
y_test = y_test.values.astype(np.int32)

X_train = X_train.astype(np.float32)
y_train = y_train.values.astype(np.int32)

if KERAS == 1:
    print ("Keras")
    def CreateModel():
    	model = Sequential()
    	model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
    	model.add(Dense(6, activation='relu'))
    	model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001)))
    	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
    	return model
    
    model = KerasClassifier(build_fn=CreateModel, epochs=50, batch_size=100, verbose=0)
    print("\n")
    PrintTest(model)

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
        verbose=0
    )
    
    PrintTest(ls, calibrate=False)
