"""
Utilidades compartidas para modelos recurrentes (LSTM, GRU).

Este módulo contiene funciones comunes para la carga y preparación de datos
para modelos de secuencia como LSTM y GRU, eliminando código duplicado.
"""

from typing import List, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np

NDArray = np.ndarray


class RecurrentModelDataUtils:
    """
    Utilidades para preparación de datos en modelos recurrentes.

    Métodos compartidos entre LSTM y GRU para:
    - Cargar secuencias desde archivos .npy
    - Cargar etiquetas escalares desde arrays por frame
    - Preparar datasets completos con validación
    """

    @staticmethod
    def load_sequence(path: str, metadata: Optional[dict] = None, model_name: str = "Model") -> NDArray:
        """
        Carga secuencia y la estandariza a formato (T,F) 2D.

        Acepta:
        - 2D: (T,F) o (F,T) → convierte a (T,F)
        - 3D: (T,F,C) → aplana a (T, F*C)

        Args:
            path: Ruta al archivo .npy
            metadata: Diccionario opcional con metadatos de dimensionality_change.
                     Si se proporciona, usa la información semántica para determinar orientación.
                     Si no, usa heurística (T < F => transpone).
            model_name: Nombre del modelo llamante (para mensajes de error)

        Returns:
            Array 2D de forma (T, F) donde:
            - T: número de frames/time steps
            - F: número total de features (features_per_frame * channels si era 3D)
        """
        X = np.load(path, allow_pickle=True)

        # Caso 3D: (frames, features_per_frame, channels)
        # Ejemplo: (4, 64, 8) → (4, 512)
        if X.ndim == 3:
            T, F, C = X.shape
            # Aplanar las dos últimas dimensiones: (T, F, C) → (T, F*C)
            X = X.reshape(T, F * C)

        # Caso 2D: (T, F) o (F, T)
        elif X.ndim == 2:
            # Determinar si necesita transposición
            if metadata and (metadata.get("output_axes_semantics") or metadata.get("output_shape")):
                # Usar metadatos para determinar orientación
                # Interpretación detallada se realiza en el modelo (LSTM/GRU) si la requiere
                pass
            else:
                # Fallback: heurística tradicional, con excepción importante:
                # NO transponer cuando T == 1, porque (1, F) ya es (time, features) con un solo paso.
                T, F = X.shape
                if T > 1 and T < F:  # si viene (F,T), trasponemos para (T,F)
                    X = X.T

        else:
            raise ValueError(
                f"Secuencia inválida en {path}: se esperaba 2D o 3D, "
                f"recibido {X.ndim}D con forma {X.shape}."
            )

        return X.astype(np.float32, copy=False)

    @staticmethod
    def load_label_scalar(path: str) -> int:
        """
        Carga etiqueta para clasificación de secuencia completa.

        Regla de negocio estándar:
        - Las transformadas generan etiquetas por frame: (n_frames,)
        - Para clasificación de secuencia, se toma la MODA (etiqueta más frecuente)

        Args:
            path: Ruta al archivo .npy con etiquetas

        Returns:
            Etiqueta más frecuente (moda) como entero
        """
        y = np.load(path, allow_pickle=True)
        y = np.array(y).reshape(-1)

        if y.size == 0:
            raise ValueError(f"Etiqueta inválida en {path}: array vacío.")

        # Caso 1: Etiqueta escalar (legacy o especial)
        if y.size == 1:
            return int(y[0])

        # Caso 2: Array de etiquetas por frame (estándar)
        # Calcular moda usando Counter para manejar tanto int como str
        from collections import Counter

        # Convertir a int si es posible, mantener como está si no
        y_clean = []
        for val in y:
            try:
                y_clean.append(int(val))
            except (ValueError, TypeError):
                # Si no se puede convertir a int, usar como string
                y_clean.append(str(val))

        # Calcular moda
        counts = Counter(y_clean)
        most_common_label = counts.most_common(1)[0][0]

        # Asegurar que retornamos int
        try:
            return int(most_common_label)
        except (ValueError, TypeError):
            raise ValueError(
                f"La moda de las etiquetas en {path} no es un entero válido: {most_common_label}"
            )

    @staticmethod
    def prepare_sequences_and_labels(
        x_paths: Sequence[str],
        y_paths: Sequence[str],
        pad_value: float = 0.0,
        metadata_list: Optional[Sequence[dict]] = None,
        load_sequence_func=None,
    ) -> Tuple[List[NDArray], NDArray, NDArray]:
        """
        Prepara secuencias y etiquetas para entrenamiento de modelo recurrente.

        Args:
            x_paths: Rutas a archivos .npy con secuencias
            y_paths: Rutas a archivos .npy con etiquetas
            pad_value: Valor de padding para secuencias variables
            metadata_list: Lista opcional de diccionarios con metadatos de dimensionality_change.
                          Si se proporciona, debe tener la misma longitud que x_paths.
            load_sequence_func: Función custom para cargar secuencias (si el modelo necesita
                              interpretar metadatos de forma especial). Si es None, usa
                              RecurrentModelDataUtils.load_sequence

        Returns:
            Tupla con:
            - sequences: lista de arrays (Ti, F) con longitudes variables
            - lengths: array (N,) con longitudes Ti de cada secuencia
            - labels: array (N,) con etiquetas enteras
        """
        if len(x_paths) != len(y_paths):
            raise ValueError(
                "x_paths y y_paths deben tener la misma longitud "
                f"(clasificación por secuencia). Recibido {len(x_paths)} vs {len(y_paths)}"
            )

        # Si no hay metadatos, crear lista vacía
        if metadata_list is None:
            metadata_list = [{} for _ in x_paths]
        elif len(metadata_list) != len(x_paths):
            raise ValueError(
                f"metadata_list debe tener la misma longitud que x_paths. "
                f"Recibido {len(metadata_list)} vs {len(x_paths)}"
            )

        # Usar función de carga por defecto si no se proporciona una custom
        if load_sequence_func is None:
            load_sequence_func = RecurrentModelDataUtils.load_sequence

        sequences: List[NDArray] = []
        lengths: List[int] = []
        labels: List[int] = []
        F_ref: Optional[int] = None
        
        total_items = len(x_paths)

        for idx, (xp, yp, meta) in enumerate(zip(x_paths, y_paths, metadata_list)):
            # Log de progreso cada 10% si hay múltiples archivos
            if total_items > 1 and idx % max(1, total_items // 10) == 0:
                print(f"\r[RecurrentModelDataUtils] Cargando {idx+1}/{total_items}: {Path(xp).name}...", end='', flush=True)
            
            # Cargar secuencia
            X = load_sequence_func(xp, metadata=meta)  # (T, F)

            # Cargar etiqueta escalar
            y = RecurrentModelDataUtils.load_label_scalar(yp)

            # Validar consistencia de número de features
            if F_ref is None:
                F_ref = int(X.shape[1])
            elif int(X.shape[1]) != F_ref:
                raise ValueError(
                    f"Dimensión de características inconsistente: "
                    f"{xp} tiene F={X.shape[1]} vs F_ref={F_ref}"
                )

            sequences.append(X)
            lengths.append(int(X.shape[0]))
            labels.append(int(y))
        
        if total_items > 1:
            print(f"\r[RecurrentModelDataUtils] {total_items} secuencias cargadas.                    ", flush=True)

        return (
            sequences,
            np.array(lengths, dtype=np.int64),
            np.array(labels, dtype=np.int64)
        )
