import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

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


def estudio_lgc_alpha(name, alpha_values=None, parallel=False, path="../experimentos/lgc_alpha/{}.npy"):
    if alpha_values is None:
        alpha_values = list(np.arange(0.1, 1, 0.1)) + [0.99]

    accuracies = []

    def ejecutar_fold(k, p_unlabeled, name, alpha):
        train_data, test_data, _ = cargar_fold(p_unlabeled, name, k)

        if 'rgcli' in path:
            clf = GSSLTransductive(k_e=50, k_i=2, nt=1, alpha=alpha)
        else:
            clf = GSSLTransductive(k_e=11, alpha=alpha)

        return accuracy_score(test_data.iloc[:, -1].values, clf.fit_predict(
            np.concatenate((train_data.iloc[:, :-1].values, test_data.iloc[:, :-1].values), axis=0)
            , np.concatenate((train_data.iloc[:, -1].values, [-1] * len(test_data.iloc[:, -1].values)))
            , method="rgcli" if 'rgcli' in path else 'gbili')[len(train_data):])

    for i, p_unlabeled in enumerate(["10", "20", "30", "40"]):
        acc = []
        for alpha in alpha_values:
            if parallel:
                with ThreadPoolExecutor(max_workers=min(int(os.cpu_count() * 0.8), 10)) as executor:
                    futures = [executor.submit(ejecutar_fold, k, p_unlabeled, name, alpha) for k in range(1, 11)]
                    run_scores = [future.result() for future in futures]
            else:
                run_scores = [ejecutar_fold(k, p_unlabeled, name, alpha) for k in range(1, 11)]

            acc.append(np.mean(run_scores))

        accuracies.append(acc)
        print("Dataset:", name, "P:", p_unlabeled, "- DONE")

    accuracies = np.array(accuracies)

    np.save(path.format(name), np.flip(accuracies.T, axis=0))
    print(f"{name} saved")
    return accuracies


def alpha_heatmap(matrix, title, more_better=True, w_labels=None):
    if w_labels is None:
        w_labels = ['.99', '.9', '.8', '.7', '.6', '.5', '.4', '.3', '.2', '.1']

    percentage = ["10%", "20%", "30%", "40%"]

    plt.figure(figsize=(4, 5))
    sns.heatmap(matrix, cmap='Blues' if more_better else 'Blues_r', linewidths=0.5, annot=True, annot_kws={"size": 10},
                xticklabels=percentage, yticklabels=w_labels)

    plt.title(title)
    plt.ylabel('Parámetro alpha')
    plt.xlabel('Porcentaje de etiquetados')

    plt.show()

    plt.show()


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
    # "nursery",
    "saheart",
    "tae",
    "vehicle",
    "wine",
    "wisconsin",
    "yeast",
    "zoo"
]
