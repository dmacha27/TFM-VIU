import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

import os
from concurrent.futures import ThreadPoolExecutor

from sklearn.semi_supervised import SelfTrainingClassifier

from metodos.GSSL.GSSL import GSSLTransductive


def encontrar_fila_con_palabra(ruta_archivo, palabra):
    with open(ruta_archivo, 'r') as archivo:
        for num_linea, linea in enumerate(archivo, 1):
            if palabra in linea:
                return num_linea
    return -1


def cargar_fold(p_unlabeled, name, k):
    train_data = pd.read_csv(
        f'../../datasets/{p_unlabeled}/{name}-ssl{p_unlabeled}-10-fold/{name}-ssl{p_unlabeled}/{name}-ssl{p_unlabeled}-10-{k}tra.dat',
        header=None,
        skiprows=encontrar_fila_con_palabra(
            f'../../datasets/{p_unlabeled}/{name}-ssl{p_unlabeled}-10-fold/{name}-ssl{p_unlabeled}/{name}-ssl{p_unlabeled}-10-{k}tra.dat',
            '@data'))

    test_data = pd.read_csv(
        f'../../datasets/{p_unlabeled}/{name}-ssl{p_unlabeled}-10-fold/{name}-ssl{p_unlabeled}/{name}-ssl{p_unlabeled}-10-{k}tst.dat',
        header=None,
        skiprows=encontrar_fila_con_palabra(
            f'../../datasets/{p_unlabeled}/{name}-ssl{p_unlabeled}-10-fold/{name}-ssl{p_unlabeled}/{name}-ssl{p_unlabeled}-10-{k}tst.dat',
            '@data'))

    columnas_strings = train_data.iloc[:, :-1].select_dtypes(exclude=['number']).columns.tolist()

    for col in columnas_strings:
        encoder = LabelEncoder()
        train_data.iloc[:, col] = encoder.fit_transform(train_data.iloc[:, col])
        train_data[col] = train_data[col].apply(pd.to_numeric)
        test_data.iloc[:, col] = encoder.transform(test_data.iloc[:, col])
        test_data[col] = test_data[col].apply(pd.to_numeric)

    if pd.api.types.is_numeric_dtype(test_data.iloc[:, -1]):
        train_data.loc[train_data.iloc[:, -1] == ' unlabeled', len(train_data.columns) - 1] = -1
        train_data.iloc[:, -1] = pd.to_numeric(train_data.iloc[:, -1])
    else:
        label_encoder = LabelEncoder()
        # Codificar las etiquetas de clase
        train_data.iloc[:, -1] = label_encoder.fit_transform(train_data.iloc[:, -1])
        train_data.loc[train_data.iloc[:, -1] == label_encoder.transform([' unlabeled'])[0], len(
            train_data.columns) - 1] = -1

        test_data.iloc[:, -1] = label_encoder.transform(test_data.iloc[:, -1])

    train_data[train_data.columns[-1]] = train_data[train_data.columns[-1]].astype(int)
    test_data[test_data.columns[-1]] = test_data[test_data.columns[-1]].astype(int)

    train_data_label = train_data[train_data.iloc[:, -1] != -1]

    return train_data, test_data, train_data_label


def cross_val(name, p_unlabeled, method="knn", graph_method="transductive"):
    accuracy = []

    def process_fold(k):
        train_data, test_data, train_data_label = cargar_fold(p_unlabeled, name, k)

        clf = None
        supervised = False

        if method == 'knn':
            clf = KNeighborsClassifier()
            supervised = True

        if method == "selftraining":
            clf = SelfTrainingClassifier(KNeighborsClassifier())

        if method == "gbili":
            if graph_method == "transductive":
                clf = GSSLTransductive(k_e=11)
            else:
                clf = None

        if method == "rgcli":
            if graph_method == "transductive":
                clf = GSSLTransductive()
            else:
                clf = None

        X = train_data_label.iloc[:, :-1].values if supervised else train_data.iloc[:, :-1].values
        y = train_data_label.iloc[:, -1].values if supervised else train_data.iloc[:, -1].values

        accuracy_k = 0

        if (method == "gbili" or method == "rgcli") and graph_method == "transductive":
            accuracy_k = accuracy_score(test_data.iloc[:, -1].values, clf.fit_predict(
                np.concatenate((train_data.iloc[:, :-1].values, test_data.iloc[:, :-1].values), axis=0)
                , np.concatenate((train_data.iloc[:, -1].values, [-1] * len(test_data.iloc[:, -1].values)))
            , method=method)[len(train_data):])
        else:
            clf.fit(X, y)

            accuracy_k = accuracy_score(test_data.iloc[:, -1].values, clf.predict(test_data.iloc[:, :-1].values))

        print("\t\tFOLD", k, "- Done")

        return accuracy_k

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_fold, range(1, 11)))

    for accuracy_k in results:
        accuracy.append(accuracy_k)

    return np.mean(accuracy)


names = [
    "appendicitis",
    "australian",
    "bupa",
    "cleveland",
    "contraceptive",
    "dermatology",
    "ecoli",
    "flare",
    "german",
    "glass",
    "haberman",
    "heart",
    "hepatitis",
    "iris",
    "led7digit",
    "monk-2",
    #"nursery",
    "saheart",
    "tae",
    "vehicle",
    "wine",
    "wisconsin",
    "yeast",
    "zoo"
]
