import numpy as np
import urllib
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.svm import SVC
import csv


def import_csv():
    # Carregando os csvs do teste e treino
    test_y = pd.read_csv('ytest.csv', header=None)
    test_y = test_y.to_numpy()
    test_y = np.ravel(test_y)
    print(test_y.shape)

    train_y = pd.read_csv('ytrain.csv', header=None)
    train_y = train_y.to_numpy()
    train_y = np.ravel(train_y)
    print(train_y.shape)

    # Carregando os testes
    train_x = pd.read_csv('Xtrain.csv', header=None)
    train_x = train_x.to_numpy()
    print(train_x.shape)

    test_x = pd.read_csv('Xtest.csv', header=None)
    test_x = test_x.to_numpy()
    print(test_x.shape)

    return train_x, train_y, test_x, test_y
