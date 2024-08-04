import io

import numpy as np
from flask import Blueprint, render_template, request, jsonify, send_from_directory
import pandas as pd
from scipy.io import arff

from app.gssl.GraphConstruction import gbili, rgcli
from app.gssl.GraphRegularization import lgc
from app.gssl.utils.datasetloader import DatasetLoader

views = Blueprint('views', __name__)


@views.route('/', methods=['GET'])
def lobby():
    language_code = request.args.get('lang') or request.headers.get('Accept-Language', '').split(',')[0].lower()

    is_spanish = language_code.startswith('es')

    return render_template('gssl.html', is_spanish=is_spanish)


def lgc_dataset_order(X, y):
    """
    Orders the dataset separating labeled and unlabeled instances.

    According to the "Learning with Local and Global Consistency" paper, the first "l" instances correspond to
    labeled points, where x_i for i<l, with l being the number of labeled instances.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The target values. Unlabeled instances are marked with -1.

    Returns
    -------
    X_ordered : array-like or sparse matrix, shape (n_samples, n_features)
        The input samples ordered with labeled instances first followed by unlabeled instances.
    y_ordered : array-like, shape (n_samples,)
        The target values ordered with labeled instances first followed by unlabeled instances.
    n_labeled: int
        The number of labeled instances.
    """

    labeled_indices = y != -1
    X_labeled = X[labeled_indices]
    y_labeled = y[labeled_indices]

    X_unlabeled = X[~labeled_indices]
    y_unlabeled = y[~labeled_indices]

    X_ordered = np.concatenate((X_labeled, X_unlabeled))
    y_ordered = np.hstack((y_labeled, y_unlabeled))

    n_labeled = len(y_labeled)

    return X_ordered, y_ordered, n_labeled


def load_file(file):
    filename = file.filename.lower()
    lines = file.read().decode('UTF-8').splitlines()
    file_stream = io.StringIO("\n".join(lines))
    if filename.endswith('.csv'):
        return pd.read_csv(file_stream)
    elif filename.endswith('.arff'):
        data, _ = arff.loadarff(file_stream)
        return pd.DataFrame(data)


def create_links(neighbors, value):
    links_set = set()
    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            if (i, neighbor) not in links_set and (neighbor, i) not in links_set:
                links_set.add((i, neighbor))
    return [{"source": str(source), "target": str(target), "value": value} for source, target in links_set]


def create_links_from_dict(graph_dict, value):
    links_set = set()
    for source, targets in graph_dict.items():
        for target in targets:
            if (source, target) not in links_set and (target, source) not in links_set:
                links_set.add((source, target))
    return [{"source": str(source), "target": str(target), "value": value} for source, target in links_set]


@views.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        df = load_file(file)
        return jsonify(list(df.columns))


@views.route('/descargar_fichero')
def descargar_fichero():
    """
    Gestiona la descarga de un fichero de prueba (en la carga del conjunto de datos).

    :return: fichero de prueba.
    """

    fichero = request.args.get('nombre')

    directorio = './static/datasets/'
    return send_from_directory(directory=directorio, path=fichero + '.arff')

@views.route('/gbili_data', methods=['POST'])
def gbili_data():
    file = request.files['file']
    target_name = request.form['target_name']
    k = int(request.form['k'])
    alpha = float(request.form['alpha'])
    max_iter = int(request.form['max_iter'])
    threshold = float(request.form['threshold'])

    dataset = load_file(file)
    dl = DatasetLoader(target_name, dataset)

    X, y, mapping, is_unlabelled = dl.get_x_y()

    if not is_unlabelled:
        y[np.random.choice(len(y), int(0.8 * len(y)), replace=False)] = -1
        """y[[23, 56, 112, 47, 3, 77, 89, 120, 130, 99, 12, 38, 141, 58, 11, 25, 105, 135, 79, 108, 8, 20, 64, 5, 6, 9, 75,
           88,
           15, 2, 110, 101, 123, 49, 81, 133, 32, 42, 137, 59]
        ] = -1"""

    X, y, _ = lgc_dataset_order(X, y)

    D, D_argsort, knn, m_knn, semi_graph, graph, unions, component_membership_semi, components_with_labeled, component_membership_graph, W = gbili(
        X, y,
        k)

    F, W, D_diag, D_sqrt_inv, S, F_t_history, pred_history, pred = lgc(X, y, W, alpha, max_iter, threshold)

    nodes = [{"id": str(i), "label": int(label)} for i, label in enumerate(y)]
    nodes_final = [{"id": str(i), "label": int(label)} for i, label in enumerate(pred)]

    links_knn = create_links(knn, value=1)
    links_m_knn = create_links(m_knn, value=8)
    links_semi_graph = create_links_from_dict(semi_graph, value=3)
    links_graph = create_links_from_dict(graph, value=1.5)

    print(m_knn)

    response = {
        "mapping": mapping,

        "gbili": {

            "dataset": {
                "nodes": nodes,
                "links": [],
                "distance": D.tolist()
            },
            "knn": {
                "nodes": nodes,
                "links": links_knn,
                "neighbors": D_argsort.tolist()
            },
            "m_knn": {
                "nodes": nodes,
                "links": links_m_knn,
                "mneighbors": m_knn
            },
            "semi_graph": {
                "nodes": nodes,
                "links": links_semi_graph,
                "components": [component_membership_semi[i] for i in range(len(X))],
                "components_with_labeled": list(components_with_labeled)
            },
            "graph": {
                "nodes": nodes,
                "links": links_graph,
                "unions": list(unions),
                "components": [component_membership_graph[i] for i in range(len(X))]

            }
        },

        "lgc": {
            "affinity": {
                "nodes": nodes,
                "links": links_graph,
                "F": F.tolist(),
                "W": W.tolist(),
            },
            "S": {
                "nodes": nodes,
                "links": links_graph,
                "D": D_diag.tolist(),
                "D_sqrt_inv": D_sqrt_inv.tolist(),
                "S": S.tolist(),
            },
            "iteration": {
                "nodes": nodes,
                "links": links_graph,
                "F_history": F_t_history.tolist(),
                "pred_history": pred_history.tolist()
            },
            "labels": {
                "nodes": nodes_final,
                "links": links_graph,
                "F_final": F_t_history[-1].tolist(),
                "pred_final": pred_history[-1].tolist()
            }
        }
    }

    return jsonify(response)


@views.route('/rgcli_data', methods=['POST'])
def rgcli_data():
    file = request.files['file']
    target_name = request.form['target_name']
    k_e = int(request.form['k_e'])
    k_i = int(request.form['k_i'])
    nt = int(request.form['nt'])
    alpha = float(request.form['alpha'])
    max_iter = int(request.form['max_iter'])
    threshold = float(request.form['threshold'])

    dataset = load_file(file)
    dl = DatasetLoader(target_name, dataset)

    X, y, mapping, is_unlabelled = dl.get_x_y()

    if not is_unlabelled:
        y[np.random.choice(len(y), int(0.4 * len(y)), replace=False)] = -1

    X, y, _ = lgc_dataset_order(X, y)

    D, kNN, L, F_rgcli, graph, W = rgcli(X, y, k_e, k_i, nt)

    F, W, D_diag, D_sqrt_inv, S, F_t_history, pred_history, pred = lgc(X, y, W, alpha, max_iter, threshold)

    nodes = [{"id": str(i), "label": int(label)} for i, label in enumerate(y)]
    nodes_final = [{"id": str(i), "label": int(label)} for i, label in enumerate(pred)]

    links_knn = create_links(kNN, value=1)
    links_graph = create_links_from_dict(graph, value=8)

    response = {
        "mapping": mapping,

        "rgcli": {
            "dataset": {
                "nodes": nodes,
                "links": [],
                "distance": D.tolist()
            },
            "searchknn": {
                "nodes": nodes,
                "links": links_knn,
                "kNN": kNN.tolist(),
                "L": L.tolist(),
                "F": F_rgcli.tolist()
            },
            "graph": {
                "nodes": nodes,
                "links": links_graph
            }
        },
        "lgc": {
            "affinity": {
                "nodes": nodes,
                "links": links_graph,
                "F": F.tolist(),
                "W": W.tolist(),
            },
            "S": {
                "nodes": nodes,
                "links": links_graph,
                "D": D_diag.tolist(),
                "D_sqrt_inv": D_sqrt_inv.tolist(),
                "S": S.tolist(),
            },
            "iteration": {
                "nodes": nodes,
                "links": links_graph,
                "F_history": F_t_history.tolist(),
                "pred_history": pred_history.tolist()
            },
            "labels": {
                "nodes": nodes_final,
                "links": links_graph,
                "F_final": F_t_history[-1].tolist(),
                "pred_final": pred_history[-1].tolist()
            }
        }
    }

    return jsonify(response)
