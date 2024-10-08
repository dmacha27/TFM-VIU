# Autor: David Martínez Acha
# Fecha: 2/08/2024 14:30
# Descripción: Permite cargar datasets
# Version: 1.0

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api import types

from app.gssl.utils.labelencoder import OwnLabelEncoder


class DatasetLoader:

    def __init__(self, target, dataset):
        """
        Cargador para dataframes.

        :param file: ruta del fichero
        """

        self.target = target
        self.dataset = dataset

    def get_allfeatures(self):
        """
        Obtiene las columnas (atributos) de los datos, incluye el target

        :return: listado de las características de los datos.
        """

        return self._get_data().columns.values

    def set_target(self, target):
        """
        Especifica el target de los datos

        :param target: el target o clase para la posterior clasificación
        """
        self.target = target

    def get_only_features(self):
        """
        Obtiene las características de los datos. NO incluye target

        :return: listado de las características de los datos (sin target).
        """
        if self.target is None:
            raise ValueError("La clase o target no ha sido establecida, selecciona primero la característica que "
                             "actúa como target")

        return np.setdiff1d(self._get_data().columns.values, self.target)

    def _get_data(self):
        """
        Obtiene los datos sin procesar.

        :return: datos en forma de dataframe
        """
        return self.dataset

    def _detect_transform_categorical_features(self, x: DataFrame):
        """
        Detecta si existen características categóricas.

        En algunos casos, ciertas características contienen valores
        numéricos discretos (y en realidad, son categóricos).
        Para estas características también se transforma el tipo a
        dato numérico (pues es la misma información que las categorías).

        :param x: instancias
        :return: False si todas son numéricas, True en caso contrario
        """

        for column in x.columns:
            if not pd.api.types.is_numeric_dtype(x[column]):
                try:
                    x[column] = pd.to_numeric(x[column])
                except ValueError:
                    return True

        return False

    def _detect_unlabelled_targets(self, y: DataFrame):
        """
        Detecta si existen datos no etiquetados. Se sigue la convención del "-1"
        para datos no etiquetados.
        Casos considerados: -1, -1.0, "-1", "-1.0"

        :param y: etiquetas
        :return: True si hay datos no etiquetados, False en caso contrario
        """
        values = y[self.target].astype(str).values
        return "-1" in values or "-1.0" in values

    def get_x_y(self):
        """
        Obtiene por separado los datos (las características) y los target o clases

        :return: las instancias (x), las clases o targets (y), el mapeo de las clases codificadas a las
        originales y si el conjunto de datos ya era semi-supervisado
        """

        if self.target is None:
            raise ValueError("La clase o target no ha sido establecida, selecciona primero la característica que "
                             "actúa como target")

        data = self._get_data()

        x = data.drop(columns=[self.target])

        if self._detect_transform_categorical_features(x):
            raise ValueError("Se han detectado características categóricas o indefinidas, "
                             "recuerde que los algoritmos solo soportan características numéricas")

        try:
            y = pd.DataFrame(
                np.array([v.decode("utf-8") if not types.is_numeric_dtype(type(v)) else v for v in
                          data[self.target].values]),
                columns=[self.target])
        except:
            y = pd.DataFrame(data[self.target], columns=[self.target])

        y.replace("?", "-1", inplace=True)

        is_unlabelled = self._detect_unlabelled_targets(y)

        y, mapping = OwnLabelEncoder().transform(y)

        return x.values, y.values.ravel(), mapping, is_unlabelled
