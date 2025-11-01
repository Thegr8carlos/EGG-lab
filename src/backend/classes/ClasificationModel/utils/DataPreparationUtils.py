"""
Utilidades compartidas para preparación de datos en modelos de clasificación.

Este módulo contiene funciones comunes para la carga, validación y procesamiento
de datos que son usadas por múltiples modelos (SVM, RandomForest, SVNN, etc.).
"""

from typing import Optional
import numpy as np

NDArray = np.ndarray


class DataPreparationUtils:
    """
    Utilidades para preparación de datos en modelos de clasificación.

    Funciones compartidas para:
    - Validar dimensionalidad de datos
    - Aplanar arrays 3D a diferentes formatos
    - Calcular moda de etiquetas
    """

    @staticmethod
    def load_and_validate_3d(path: str, allow_pickle: bool = False) -> np.ndarray:
        """
        Carga archivo .npy y valida que sea 3D.

        Args:
            path: Ruta al archivo .npy
            allow_pickle: Si permitir carga de objetos Python serializados

        Returns:
            Array 3D con formato (n_frames, features, n_channels)

        Raises:
            ValueError: Si el array no es 3D, con mensaje explicativo sobre
                       qué transformadas aplicar
        """
        X = np.load(path, allow_pickle=allow_pickle)

        if X.ndim != 3:
            raise ValueError(
                f"Los datos deben ser 3D (n_frames, features, n_channels) después de aplicar transform. "
                f"Recibido {X.ndim}D con shape={X.shape} en {path}. "
                f"Asegúrate de aplicar WindowingTransform, FFTTransform, DCTTransform o WaveletTransform."
            )

        return X

    @staticmethod
    def validate_3d_shape(X: np.ndarray, expected_ndim: int = 3) -> bool:
        """
        Valida que un array tenga la dimensionalidad esperada.

        Args:
            X: Array a validar
            expected_ndim: Número de dimensiones esperadas (default: 3)

        Returns:
            True si la validación pasa

        Raises:
            ValueError: Si la dimensionalidad no coincide
        """
        if X.ndim != expected_ndim:
            raise ValueError(
                f"Se esperaba array {expected_ndim}D, recibido {X.ndim}D con shape={X.shape}"
            )
        return True

    @staticmethod
    def flatten_3d_completely(X: np.ndarray) -> np.ndarray:
        """
        Aplana array 3D completamente a 1D.

        Útil para modelos que necesitan vector de features único por muestra.
        Ejemplo: RandomForest donde cada archivo = 1 muestra

        Args:
            X: Array 3D de forma (T, F, C)

        Returns:
            Array 1D de forma (T*F*C,)

        Example:
            >>> X = np.random.rand(4, 64, 8)  # 4 frames, 64 features, 8 canales
            >>> X_flat = flatten_3d_completely(X)
            >>> X_flat.shape
            (2048,)  # 4 * 64 * 8 = 2048
        """
        if X.ndim != 3:
            raise ValueError(f"Se esperaba 3D, recibido {X.ndim}D con shape={X.shape}")

        return X.reshape(-1)

    @staticmethod
    def flatten_3d_preserve_time(X: np.ndarray) -> np.ndarray:
        """
        Aplana array 3D a 2D preservando la dimensión temporal.

        Útil para modelos que procesan frames como ejemplos independientes.
        Ejemplo: SVM donde cada frame = 1 ejemplo

        Args:
            X: Array 3D de forma (T, F, C)

        Returns:
            Array 2D de forma (T, F*C)

        Example:
            >>> X = np.random.rand(4, 64, 8)  # 4 frames, 64 features, 8 canales
            >>> X_flat = flatten_3d_preserve_time(X)
            >>> X_flat.shape
            (4, 512)  # 4 frames, 512 features (64*8)
        """
        if X.ndim != 3:
            raise ValueError(f"Se esperaba 3D, recibido {X.ndim}D con shape={X.shape}")

        T, F, C = X.shape
        return X.reshape(T, F * C)

    @staticmethod
    def calculate_label_mode(labels: np.ndarray) -> int:
        """
        Calcula la moda (etiqueta más frecuente) de un array de etiquetas.

        Maneja tanto etiquetas int como str, siempre retorna int.
        Usado cuando se tiene un array de etiquetas por frame y se necesita
        una sola etiqueta para toda la muestra.

        Args:
            labels: Array 1D con etiquetas (puede ser int o str)

        Returns:
            Etiqueta más frecuente como entero

        Raises:
            ValueError: Si el array está vacío o la moda no es convertible a int

        Example:
            >>> labels = np.array([0, 0, 0, 1, 0, 0, 0])
            >>> calculate_label_mode(labels)
            0

            >>> labels = np.array(['1', '1', '2', '1'])
            >>> calculate_label_mode(labels)
            1
        """
        from collections import Counter

        labels = np.array(labels).reshape(-1)

        if labels.size == 0:
            raise ValueError("Array de etiquetas vacío")

        # Caso especial: solo una etiqueta
        if labels.size == 1:
            return int(labels[0])

        # Convertir a int si es posible, mantener como str si no
        y_clean = []
        for val in labels:
            try:
                y_clean.append(int(val))
            except (ValueError, TypeError):
                y_clean.append(str(val))

        # Calcular moda
        counts = Counter(y_clean)
        most_common = counts.most_common(1)[0][0]

        # Asegurar que retornamos int
        try:
            return int(most_common)
        except (ValueError, TypeError):
            raise ValueError(
                f"La moda de las etiquetas no es un entero válido: {most_common}"
            )

    @staticmethod
    def load_labels_with_mode(paths: list, allow_pickle: bool = True) -> np.ndarray:
        """
        Carga múltiples archivos de etiquetas y calcula la moda de cada uno.

        Útil para modelos que necesitan una etiqueta escalar por archivo,
        pero los archivos contienen arrays de etiquetas por frame.

        Args:
            paths: Lista de rutas a archivos .npy con etiquetas
            allow_pickle: Si permitir carga de objetos Python serializados

        Returns:
            Array 1D con una etiqueta (moda) por archivo

        Example:
            >>> paths = ['labels1.npy', 'labels2.npy']
            >>> # labels1.npy contiene [0,0,0,1,0]
            >>> # labels2.npy contiene [2,2,2,2]
            >>> labels = load_labels_with_mode(paths)
            >>> labels
            array([0, 2])
        """
        y_list = []

        for path in paths:
            y_sample = np.load(path, allow_pickle=allow_pickle)
            y_sample = np.array(y_sample).reshape(-1)

            # Calcular moda para este archivo
            label_mode = DataPreparationUtils.calculate_label_mode(y_sample)
            y_list.append(label_mode)

        return np.array(y_list, dtype=np.int64)
