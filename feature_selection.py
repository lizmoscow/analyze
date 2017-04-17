import pandas as pd
from sklearn import ensemble, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans


data = pd.read_csv('drivers_50000.csv')

data.at[data['Accidents'] > 0, 'AccidentsBin'] = 1
data.at[data['Accidents'] == 0, 'AccidentsBin'] = 0

XX = data[
    ['Age', 'Experience', 'PreviousAccidents', 'RouteDistance', 'Distance', 'HomeLat', 'HomeLng', 'WorkLat', 'WorkLng']]
y = data['AccidentsBin']
X = preprocessing.scale(XX)
feature_names = XX.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=11)

def show_feature_importances(X, y,feature_names):

    rf = ensemble.RandomForestClassifier(random_state=11)
    param_grid = {'n_estimators':[55,75,100],'criterion':["entropy","gini"]}
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_log_loss')
    grid.fit(X,y)
    best=grid.best_estimator_
    importances = best.feature_importances_
    indices = np.argsort(importances)[::-1]


    d_first = len(feature_names)
    plt.figure(figsize=(8, 8))
    plt.title("Feature importances")
    plt.bar(range(d_first), importances[indices[:d_first]], align='center')
    plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
    plt.xlim([-1, d_first]);
    plt.show()

show_feature_importances(X_train,y_train,feature_names)


def cluster_locations(data_input,n_clusters=10):
    homeLoc = data[['HomeLat', 'HomeLng']].values
    workLoc = data[['WorkLat','WorkLng']].values
    k_means_home = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means_home.fit(homeLoc)
    k_means_work = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means_work.fit(workLoc)
    data_input = data_input.assign(work=pd.Series(k_means_work.labels_, dtype="category"))
    data_input = data_input.assign(home=pd.Series(k_means_home.labels_, dtype="category"))
    data_input = data_input.drop(['WorkLat', 'WorkLng', 'HomeLat', 'HomeLng'], axis=1)
    data_input = data_input.dropna()
    home_work_vector = pd.get_dummies(data_input[['home', 'work']])
    data_input = data_input.drop(['home','work'],axis=1)
    data_input= pd.concat([data_input, home_work_vector], axis=1)
    return data_input

data=cluster_locations(data)
print(data.info())
XX = data.drop(['AccidentsBin','Accidents','Skill','RushFactor'],axis=1)
y = data['AccidentsBin']
X = preprocessing.scale(XX)
feature_names = XX.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=11)

show_feature_importances(X_train,y_train,feature_names)