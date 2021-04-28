import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeClassifier
from sklearn import preprocessing

from sklearn import svm
from sklearn.svm import SVC

from sklearn import metrics
from reduction import svfs
import matplotlib.pyplot as plt

import collections
import itertools
import math
import scipy
from random import gauss
import statistics

import networkx as nx

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.mlab as mlab

from sklearn.naive_bayes import GaussianNB



# dataset_list = ['GDS1615_full_NoFeature', 'GDS3268_full_NoFeature', 'GDS531_full_NoFeature','GDS1962_full_NoFeature', 'GDS968_full_NoFeature','GDS3929_full_NoFeature', 'GDS2545_full_NoFeature', 'GDS2546_full_NoFeature', 'GDS2547_full_NoFeature']
# dataset_list=['pixraw10P','warpAR10P','warpPIE10P','orlraws10P']
# dataset_list=['TOX_171','SMK_CAN_187','Prostate_GE','lymphoma','leukemia','lung','GLIOMA','GLI_85','CLL_SUB_111','ALLAML','colon','nci9']

dataset_list=['pixraw10P']
path = os.path.abspath(os.getcwd()) + "/Datasets/"

res = pd.DataFrame(columns=['dataset', 'original_features', 'reduced_feature', 'optimizer', 'classifier', 'acc'])
clear = lambda: os.system('cls')
for dataset in dataset_list:
    print("\nDataset: ", dataset)
    data = pd.read_csv(path + dataset + ".csv", header=None)
    dataX = data.copy().iloc[:, :-1]
    dataY = data.copy().iloc[:, data.shape[1] - 1]
    acc_list = []
    acc_list2 = []
    acc_list3 = []
    k_fold = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)

    for reduced_feature in range(5,31,5):
        inter_dim = 64
        encoding_dim = reduced_feature

        allOptim = ['Nadam', 'adamax', 'adagrad', 'adadelta', 'adam', 'SGD', 'RMSprop']
        historyf = []

        for optim in allOptim:

            for train_idx, test_dix in k_fold.split(dataX, dataY):
                train_x, test_x = dataX.iloc[train_idx, :].copy(), dataX.iloc[test_dix, :].copy()
                train_y, test_y = dataY.iloc[train_idx].copy(), dataY.iloc[test_dix].copy()

                min_max_scaler = preprocessing.MinMaxScaler()
                X_train = min_max_scaler.fit_transform(train_x)
                X_test = min_max_scaler.fit_transform(test_x)

                ncol = X_train.shape[1]
                input_dim = Input(shape=(ncol,))


                encoded = Dense(inter_dim, activation='relu', activity_regularizer=regularizers.l1(10e-1))(input_dim)
                encoded = Dense(encoding_dim, activation='relu', name='bottleneck', activity_regularizer=regularizers.l1(10e-1))(encoded)
                decoded = Dense(inter_dim, activation='relu', activity_regularizer=regularizers.l1(10e-1))(encoded)
                decoded = Dense(ncol, activation='relu', activity_regularizer=regularizers.l1(10e-1))(decoded)
                autoencoder = Model(inputs=input_dim, outputs=decoded)
                autoencoder.compile(optimizer=optim, loss='mean_squared_error', metrics=['accuracy'])
                historyf.append( autoencoder.fit(X_train, X_train, epochs=250, batch_size=16, shuffle=True, verbose=0, validation_data=(X_test, X_test)))
#                 print('FITTED with ' + optim)

                encoder = Model(inputs=input_dim, outputs=encoded)

                Zenc = encoder.predict(X_train)
                X = StandardScaler().fit_transform(Zenc)

                ZencT = encoder.predict(X_test)
                X_t = StandardScaler().fit_transform(ZencT)

                clf = RandomForestClassifier()
                clf.fit(X, train_y)
                y_pred = clf.predict(X_t)
                acc = metrics.accuracy_score(test_y, y_pred)
                acc_list.append(acc)

                clf2 = svm.SVC(decision_function_shape='ovo')
                clf2.fit(X, train_y)
                y_pred2 = clf2.predict(X_t)
                acc2 = metrics.accuracy_score(test_y, y_pred2)
                acc_list2.append(acc2)

#                 clf3 = GaussianNB()
#                 clf3.fit(X, train_y)
#                 y_pred3 = clf3.predict(X_t)
#                 acc3 = metrics.accuracy_score(test_y, y_pred3)
#                 acc_list3.append(acc3)

            print(dataset, '\t', ncol, '\t', reduced_feature, '\t', optim, '\t', 'RF', '\t', np.average(acc_list) * 100)
            print(dataset, '\t', ncol, '\t', reduced_feature, '\t', optim, '\t', 'SVM', '\t', np.average(acc_list2) * 100)
            res.loc[len(res.index)] = [dataset, ncol, reduced_feature, optim, 'RF', np.average(acc_list) * 100]
            res.loc[len(res.index)] = [dataset, ncol, reduced_feature, optim, 'SVM', np.average(acc_list2) * 100]
print(res)
