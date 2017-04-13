import os

# Keras backend settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # or coment it to use GPU by default
os.environ["KERAS_BACKEND"] = "theano"  # or coment it to use tensorflow by default

# Theano settings
os.environ["THEANO_FLAGS"] = "device=cpu"  # or "device=cuda"

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
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
# RF = 0
GBT = 1
# GP = 0
MLP = 0
# KERAS = 0
# LASAGNE = 0

data = pd.read_csv('drivers_50000.csv')

data.at[data['Accidents'] > 0, 'AccidentsBin'] = 1
data.at[data['Accidents'] == 0, 'AccidentsBin'] = 0

XX = data[
    ['Age', 'Experience', 'PreviousAccidents', 'RouteDistance', 'Distance', 'HomeLat', 'HomeLng', 'WorkLat', 'WorkLng']]
y = data['AccidentsBin']

# standardize the data attributes
X = preprocessing.scale(XX)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=11)

print("### Dummy")
y_dummy = np.empty_like(y_test)
y_dummy[:] = np.average(y_test.values)
average_probability = np.average(y_dummy)
print('probability: ', average_probability)
print('brier_score_loss: ', metrics.brier_score_loss(y_test, y_dummy))
print('profit: ', round(np.sum(y_dummy - y_test) / y_test.shape[0], 4))


def PrintTest(clf, X_test=X_test, y_test=y_test):
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, -1]
        print('log_loss: ', metrics.log_loss(y_test, prob_pos))
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

    print('brier_score_loss: ', metrics.brier_score_loss(y_test, prob_pos))
    print(metrics.classification_report(y_test, y_pred))
    Compete(prob_pos)


def Compete(y_proba1, y_proba2=y_dummy, margin1=1, margin2=1):
    premium1 = y_proba1 * margin1
    premium2 = y_proba2 * margin2
    selector1 = premium1 < premium2
    selector2 = premium2 < premium1
    profit1 = premium2 - y_test
    profit2 = premium1 - y_test
    average_profit = np.sum(np.select([selector1], [profit1])) / np.sum(selector1)
    deals = np.sum(selector1) / selector1.shape[0]
    print('1. Profit (average): ', average_profit, ', Profit (total): ', average_profit * deals, ', Deals: ', deals)
    print('2: Profit (average): ', np.sum(np.select([selector2], [profit2])) / np.sum(selector2), ', deals: ',
          np.sum(selector2) / selector2.shape[0])
    return average_profit, deals


def competition_score(estimator, X, y):
    if hasattr(estimator, "predict_proba"):
        prob_pos = estimator.predict_proba(X_test)[:, -1]
        print('log_loss: ', metrics.log_loss(y_test, prob_pos))
    else:  # use decision function
        prob_pos = estimator.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

    return Compete(prob_pos)


# Average profit per selected driver
def average_competition_score(estimator, X, y):
    average_profit, deals = competition_score(estimator, X, y)
    return average_profit


# Total profit for all selected drivers
# value 0.06 means: for 1000 drivers in the set and $100 loss per accident the model gets 0.06 * 1000 * 100 = $6000 profit
def total_competition_score(estimator, X, y):
    average_profit, deals = competition_score(estimator, X, y)
    return average_profit * deals


def grid_scores_to_df(grid_scores):
    """
    Convert a sklearn.grid_search.GridSearchCV.grid_scores_ attribute to a tidy
    pandas DataFrame where each row is a hyperparameter-fold combinatination.
    """
    rows = list()
    for grid_score in grid_scores:
        for fold, score in enumerate(grid_score.cv_validation_scores):
            row = grid_score.parameters.copy()
            row['fold'] = fold
            row['score'] = score
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def grid_results_to_df(grid):
    results = pd.DataFrame.from_dict(grid.cv_results_)
    results = results.drop(
        ['params', 'split0_test_score', 'split0_train_score', 'split1_test_score', 'split1_train_score',
         'split2_test_score',
         'split2_train_score', 'std_fit_time', 'std_score_time', 'std_test_score',
         'std_train_score'], axis=1)
    results = results.sort_values(by='rank_test_score')
    return results

print("### Logistic Regression")
lr = LogisticRegression()
lr.fit(X_train, y_train)
PrintTest(lr)

if KNN == 1:
    print("### kNN")
    f_neighbors_array = [25, 50, 100, 250]
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid={'n_neighbors': f_neighbors_array}, scoring='neg_log_loss')
    grid.fit(X_train, y_train)

    best_cv_err = 1 - grid.best_score_
    best_n_neighbors = grid.best_estimator_.n_neighbors
    print(best_cv_err, best_n_neighbors)

    knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
    knn.fit(X_train, y_train)
    PrintTest(knn)

if SV == 1:
    print("### SVC")
    C_array = np.logspace(-2, 2, num=3)
    gamma_array = np.logspace(-3, 2, num=3)
    svc = SVC(kernel='rbf')
    grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array}, scoring='neg_log_loss')
    grid.fit(X_train, y_train)
    print('CV error    = ', 1 - grid.best_score_)
    print('best C      = ', grid.best_estimator_.C)
    print('best gamma  = ', grid.best_estimator_.gamma)

    svc = SVC(kernel='rbf', C=grid.best_estimator_.C, gamma=grid.best_estimator_.gamma)
    svc.fit(X_train, y_train)
    PrintTest(svc)

if MLP == 1:
    param_grid = {'hidden_layer_sizes': [(12, 6), (24, 6), (24, 12)], 'solver': ['adam', 'lbfgs']}
    mlp = MLPClassifier(alpha=1e-5)
    grid = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring=total_competition_score, verbose=10)
    print("### MLP")
    grid.fit(X_train, y_train)
    print('CV error    = ', 1 - grid.best_score_)
    print('best hidden layers sizes = ', grid.best_estimator_.hidden_layer_sizes)
    print('best solver  = ', grid.best_estimator_.solver)

    mlp = MLPClassifier(solver=grid.best_estimator_.solver, alpha=1e-5,
                        hidden_layer_sizes=grid.best_estimator_.hidden_layer_sizes, random_state=11, verbose=1)
    mlp.fit(X_train, y_train)
    PrintTest(mlp)

if GBT == 1:
    gbt = ensemble.GradientBoostingClassifier()
    param_grid = {'n_estimators': [i for i in range(100, 400, 50)], 'learning_rate': [0.1,0.15,0.2],
                  'loss': ['deviance', 'exponential'],'subsample':[0.2,0.5,0.7]}
    grid = GridSearchCV(estimator=gbt, param_grid=param_grid, scoring=total_competition_score, verbose=1)
    print("### GBT")
    grid.fit(X_train, y_train)

    print(grid_results_to_df(grid))

    best_gbt = grid.best_estimator_
    best_gbt.fit(X_train, y_train)
    PrintTest(best_gbt)
