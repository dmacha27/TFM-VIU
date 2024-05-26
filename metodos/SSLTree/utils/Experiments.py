import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import os
from concurrent.futures import ThreadPoolExecutor

from metodos.SSLTree.SSLTree import SSLTree


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


def cross_val(name, p_unlabeled="20"):
    accuracy_ssl = []
    accuracy_dt = []
    accuracy_st = []

    print("PERCENTAGE:", p_unlabeled, "- DATASET:", name)

    w_values = np.arange(0, 1.01, 0.01)

    best_w = None
    best_score = 0

    for w in w_values:
        run_scores = []

        for k in range(1, 11):
            train_data, test_data, train_data_label = cargar_fold(p_unlabeled, name, k)

            my_tree = SSLTree(w=w)
            my_tree.fit(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)

            run_scores.append(
                accuracy_score(test_data.iloc[:, -1].values, my_tree.predict(test_data.iloc[:, :-1].values)))

        avg_score = np.mean(run_scores)

        if avg_score > best_score:
            best_score = avg_score
            best_w = w

    print(f'Mejor w: {best_w} con accuracy: {best_score}')

    for k in range(1, 11):
        train_data, test_data, train_data_label = cargar_fold(p_unlabeled, name, k)

        my_tree = SSLTree(w=best_w)
        my_tree.fit(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)
        # print(my_tree.export_tree())
        # print(accuracy_score(test_data.iloc[:, -1].values, my_tree.predict(test_data.iloc[:, :-1].values)))

        dt = DecisionTreeClassifier()
        dt.fit(train_data_label.iloc[:, :-1].values, train_data_label.iloc[:, -1].values)
        # print(export_text(dt))
        # print(accuracy_score(test_data.iloc[:, -1].values, dt.predict(test_data.iloc[:, :-1].values)))

        self_training_model = SelfTrainingClassifier(DecisionTreeClassifier())
        self_training_model.fit(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)

        accuracy_ssl.append(
            accuracy_score(test_data.iloc[:, -1].values, my_tree.predict(test_data.iloc[:, :-1].values)))
        accuracy_dt.append(accuracy_score(test_data.iloc[:, -1].values, dt.predict(test_data.iloc[:, :-1].values)))
        accuracy_st.append(accuracy_score(test_data.iloc[:, -1].values,
                                          self_training_model.predict(test_data.iloc[:, :-1].values)))
        print("\tFOLD", k, "- Done")

    return np.median(accuracy_ssl), np.median(accuracy_dt), np.median(accuracy_st)


def estudio_w(name, parallel=False):
    accuracies_ssl = []

    w_values = np.arange(0, 1.1, 0.1)

    def ejecutar_fold(k, p_unlabeled, name, w):
        train_data, test_data, _ = cargar_fold(p_unlabeled, name, k)

        my_tree = SSLTree(w=w)
        my_tree.fit(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)

        return accuracy_score(test_data.iloc[:, -1].values, my_tree.predict(test_data.iloc[:, :-1].values))

    for i, p_unlabeled in enumerate(["10", "20", "30", "40"]):
        acc = []
        for w in w_values:
            if parallel:
                with ThreadPoolExecutor(max_workers=min(int(os.cpu_count() * 0.8), 10)) as executor:
                    futures = [executor.submit(ejecutar_fold, k, p_unlabeled, name, w) for k in range(1, 11)]
                    run_scores = [future.result() for future in futures]
            else:
                run_scores = [ejecutar_fold(k, p_unlabeled, name, w) for k in range(1, 11)]

            acc.append(np.mean(run_scores))

        accuracies_ssl.append(acc)
        print("Dataset:", name, "P:", p_unlabeled, "- DONE")

    accuracies_ssl = np.array(accuracies_ssl)

    np.save(f"../experimentos/w/{name}.npy", np.flip(accuracies_ssl.T, axis=0))
    print(f"{name} saved")
    return accuracies_ssl


def plot_estudio_w_expandido(accuracies, name):
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.xlim(0, 40)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels(['10%', '20%', '30%', '40%'])

    ax.plot(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27,
         28, 29, 30, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        , np.ravel(accuracies), marker='o', markerfacecolor='none', linestyle='-', color='red', markersize=8,
        linewidth=1)

    plt.axvline(x=10, color='g', linestyle='-.')
    plt.axvline(x=20, color='g', linestyle='-.')
    plt.axvline(x=30, color='g', linestyle='-.')

    ax2 = ax.twiny()

    ws = ['.0', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1/0',
          '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1/0',
          '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1/0',
          '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1']

    ax2.set_xticks(np.arange(0, 41, 1))
    ax2.set_xticklabels(ws)

    ax.set_xlabel('Porcentaje de etiquetados')
    ax.set_ylabel('Accuracy (media 10 folds)')
    ax2.set_xlabel('Parámetro peso (w)')

    plt.title("Efecto del parámetro w en el Dataset " + name, pad=15)
    plt.show()


def plot_estudio_w(accuracies, name):
    accuracies = np.ravel(accuracies)
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.xlim(0, 10)
    plt.ylim(min(accuracies) - 0.03, math.ceil((max(accuracies) + 0.01) * 100) / 100)

    ax.plot(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        , accuracies[0:11], marker='o', markerfacecolor='none', linestyle='-', color='green', markersize=8, linewidth=1,
        label="10%")

    ax.plot(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        , accuracies[11:22], marker='o', markerfacecolor='none', linestyle='-', color='blue', markersize=8, linewidth=1,
        label="20%")

    ax.plot(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        , accuracies[22:33], marker='o', markerfacecolor='none', linestyle='-', color='red', markersize=8, linewidth=1,
        label="30%")

    ax.plot(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        , accuracies[33:44], marker='o', markerfacecolor='none', linestyle='-', color='purple', markersize=8,
        linewidth=1, label="40%")

    ws = ['.0', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1']

    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_xticklabels(ws)

    ax.set_xlabel('Parámetro peso (w)')
    ax.set_ylabel('Accuracy (media 10 folds)')
    ax.legend(title='Porcentaje \nde etiquetados', loc='best', fontsize='small')

    plt.title("Efecto del parámetro w en el Dataset " + name, pad=15)
    plt.margins(0)
    plt.show()


def w_heatmap(matrix, less_better=False):
    w_labels = ['1', '.9', '.8', '.7', '.6', '.5', '.4', '.3', '.2', '.1', '.0']
    percentage = ["10%", "20%", "30%", "40%"]

    plt.figure(figsize=(4, 5))  # Tamaño ajustado
    sns.heatmap(matrix, cmap='Blues' if less_better else 'Blues_r', linewidths=0.5, annot=True, annot_kws={"size": 10},
                xticklabels=percentage, yticklabels=w_labels)

    plt.title('Estudio de w')
    plt.ylabel('Parámetro w')
    plt.xlabel('Porcentaje de etiquetados')

    plt.show()

    plt.show()


names = [
    # "abalone",
    "appendicitis",
    "australian",
    # "automobile",
    # "banana", best_left (puede ser porque tiene -1 en las etiquetas)
    # "breast",
    "bupa",
    # "chess",
    "cleveland",
    # "coil2000", tarda demasiado
    "contraceptive",
    # crx",
    "dermatology",
    "ecoli",
    "flare",
    "german",
    "glass",
    "haberman",
    "heart",
    "hepatitis",
    # "housevotes",
    "iris",
    "led7digit",
    # "lymphography",
    # "magic", tarda mucho
    "mammographic",
    "marketing",
    "monk-2",
    "movement_libras",
    # "mushroom",
    "nursery",
    "page-blocks",
    "penbased",
    "phoneme",
    # "pima",
    # "ring", no hay problema, pero tarda como dos milenios (aprox)
    "saheart",
    "satimage",
    "segment",
    "sonar",
    "spambase",
    "spectfheart",
    # "splice",
    "tae",
    # "texture", no hay problema, pero tarda como tres milenios (aprox)
    "thyroid",
    "tic-tac-toe",
    # "titanic", tiene -1 en las etiquetas
    "twonorm",
    "vehicle",
    "vowel",
    "wine",
    "wisconsin",
    "yeast",
    "zoo"
]
