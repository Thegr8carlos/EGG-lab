from __future__ import annotations

import time
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import Field, field_validator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from backend.classes.ClasificationModel.ClsificationModels import Classifier
from backend.classes.Metrics import EvaluationMetrics
from backend.classes.dataset import _load_and_concat

NDArray = np.ndarray


class RandomForest(Classifier):
    n_estimators: int = Field(100, ge=1, le=1000, description="Número de árboles")
    max_depth: Optional[int] = Field(
        None,
        ge=1,
        description="Profundidad máxima del árbol (None para ilimitado)"
    )
    criterion: Literal["gini", "entropy"] = Field(
        "gini", description="Función de impureza"
    )
    # Opcionales útiles
    random_state: Optional[int] = Field(
        None, description="Semilla para reproducibilidad"
    )
    class_weight: Optional[Literal["balanced", "balanced_subsample"]] = Field(
        None, description="Rebalanceo automático por frecuencia de clases"
    )
    n_jobs: Optional[int] = Field(
        None, description="Núcleos para entrenamiento (None=1, -1=usar todos)"
    )

    # Guardaremos el modelo entrenado para poder hacer query luego
    _model: Optional[RandomForestClassifier] = None  # type: ignore[assignment]

    @staticmethod
    def _prepare_xy(
        x_paths: Sequence[str],
        y_paths: Sequence[str],
    ) -> Tuple[NDArray, NDArray]:
        """
        Carga y prepara datos para RandomForest.

        Regla de negocio estándar:
        - Las transformadas generan datos 3D: (n_samples, time_steps, features, channels)
          o más comúnmente por muestra individual: (time_steps, features, channels)
        - Las etiquetas son por frame: (n_samples, n_frames) o por muestra: (n_frames,)

        Procesamiento:
        1. Datos 3D → Aplanar a 2D (n_samples, time_steps * features * channels)
        2. Etiquetas por frame → Moda por muestra

        Returns:
            X: (n_samples, n_features_totales)
            y: (n_samples,) con etiqueta por muestra (moda de frames)
        """
        from collections import Counter

        # Cargar y procesar datos X
        X_list = []
        for x_path in x_paths:
            X_sample = np.load(x_path, allow_pickle=False)

            # Caso 3D: (time_steps, features_per_step, channels)
            # Aplanar a 1D: time_steps * features_per_step * channels
            if X_sample.ndim == 3:
                X_sample = X_sample.reshape(-1)  # Aplanar completamente

            # Caso 2D: (time_steps, features)
            # Aplanar a 1D
            elif X_sample.ndim == 2:
                X_sample = X_sample.reshape(-1)

            # Caso 1D: ya está aplanado
            elif X_sample.ndim == 1:
                pass

            else:
                raise ValueError(f"Datos con dimensionalidad no soportada: {X_sample.ndim}D en {x_path}")

            X_list.append(X_sample)

        # Concatenar todas las muestras
        X = np.vstack(X_list) if len(X_list) > 1 else X_list[0].reshape(1, -1)

        # Cargar y procesar etiquetas y
        y_list = []
        for y_path in y_paths:
            y_sample = np.load(y_path, allow_pickle=True)
            y_sample = np.array(y_sample).reshape(-1)

            if y_sample.size == 0:
                raise ValueError(f"Etiqueta vacía en {y_path}")

            # Caso 1: Etiqueta escalar
            if y_sample.size == 1:
                y_list.append(int(y_sample[0]))

            # Caso 2: Array de etiquetas por frame (estándar)
            # Calcular moda
            else:
                # Convertir a int si es posible
                y_clean = []
                for val in y_sample:
                    try:
                        y_clean.append(int(val))
                    except (ValueError, TypeError):
                        y_clean.append(str(val))

                # Calcular moda
                counts = Counter(y_clean)
                most_common_label = counts.most_common(1)[0][0]

                try:
                    y_list.append(int(most_common_label))
                except (ValueError, TypeError):
                    raise ValueError(f"La moda de las etiquetas en {y_path} no es un entero válido: {most_common_label}")

        y = np.array(y_list, dtype=np.int64)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y length mismatch: X={X.shape[0]} vs y={y.shape[0]}"
            )

        return X, y

    @classmethod
    def train(
        cls,
        instance: "RandomForest",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
    ) -> EvaluationMetrics:
        """
        Entrena un RandomForest usando las rutas dadas (concatenación de shards).
        Devuelve EvaluationMetrics como en SVM.train.
        """
        # 1) Datos
        X_train, y_train = cls._prepare_xy(xTrain, yTrain)
        X_test, y_test = cls._prepare_xy(xTest, yTest)

        # 2) Modelo
        model = RandomForestClassifier(
            n_estimators=instance.n_estimators,
            max_depth=instance.max_depth,
            criterion=instance.criterion,
            random_state=instance.random_state,
            class_weight=instance.class_weight,
            n_jobs=instance.n_jobs,
        )

        # 3) Entrenamiento + evaluación
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        rec = float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        f1 = float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        cm = confusion_matrix(y_test, y_pred).tolist()

        # AUC-ROC (binario o multiclase con OVR)
        try:
            proba = model.predict_proba(X_test)  # (n_samples, n_classes)
            auc = float(
                roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
            )
        except Exception:
            auc = 0.0

        eval_seconds = time.perf_counter() - t0

        # Guardar modelo en la instancia para inferencia posterior
        instance._model = model

        return EvaluationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            confusion_matrix=cm,
            auc_roc=auc,
            loss=[],  # No hay curva de pérdida en RF de sklearn
            evaluation_time=f"{eval_seconds:.4f}s",
        )

    @classmethod
    def query(
        cls,
        instance: "RandomForest",
        X: Union[NDArray, str, Sequence[str]],
        return_proba: bool = False,
    ) -> Union[List[int], Tuple[List[int], List[List[float]]]]:
        """
        Realiza inferencia con el modelo entrenado.
        - X puede ser: np.ndarray con forma (n_samples, n_features)
          o rutas .npy (str o list[str]) que se concatenarán.
        - return_proba=True devuelve también las probabilidades por clase.
        """
        if instance._model is None:
            raise RuntimeError("El modelo RandomForest no ha sido entrenado aún.")

        if isinstance(X, np.ndarray):
            X_in = X
        else:
            x_paths = [X] if isinstance(X, str) else list(X)
            X_in = _load_and_concat(x_paths)

        y_pred = instance._model.predict(X_in).tolist()
        if return_proba:
            proba = instance._model.predict_proba(X_in).tolist()
            return y_pred, proba
        return y_pred
