__author__ = 'manabchetia'

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


def read_data(filename):
    # data = np.loadtxt('../data/train.csv')
    data = pd.read_csv(filename)
    id = data['id']
    X = data.ix[:, 1:94]
    if 'train' in filename:
        targets = data['target']
        y = targets.replace(['Class_' + str(i) for i in xrange(1, 10)], range(0, 9))
        return np.asarray(X), np.asarray(y), id.as_matrix
    else:
        return np.asarray(X), id.as_matrix


def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm


def train_NB(X, y):
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X, y)
    return naive_bayes


def train_KNN(X, y):
    knn = KNeighborsClassifier()
    knn.fit(X, y)
    return knn


def train_random_forest(X, y):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    return rf


def main():
    train_file = '../data/train.csv'
    test_file = '../data/test.csv'

    X_train, y_train, _ = read_data(train_file)
    X_test, _ = read_data(test_file)

    rf = train_random_forest(X_train, y_train)
    prob_rf = rf.predict_proba(X_test)

    id = [i for i in xrange(1, len(X_test) + 1)]
    prob_rf = np.insert(prob_rf, 0, id, axis=1)
    np.savetxt("foo.csv", prob_rf, delimiter=",",
               fmt=['%d', '%1.1f', '%1.1f', '%1.1f', '%1.1f', '%1.1f', '%1.1f', '%1.1f', '%1.1f', '%1.1f'],
               header='id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9')
    print(prob_rf)


if __name__ == '__main__': main()
