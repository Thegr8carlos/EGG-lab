from __future__ import annotations

import time
import pickle
from pathlib import Path
from datetime import datetime
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
from backend.classes.ClasificationModel.utils.TrainResult import TrainResult
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

    # =================================================================================
    # Persistencia: save/load
    # =================================================================================
    def save(self, path: str):
        """
        Guarda la instancia completa (configuración + modelo entrenado) a disco.
        
        Args:
            path: Ruta completa donde guardar el archivo .pkl
                  Ejemplo: "src/backend/models/p300/randomforest_20251109_143022.pkl"
        
        Note:
            Usa pickle para serializar toda la instancia incluyendo _model.
            El directorio padre se crea automáticamente si no existe.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[RandomForest] Modelo guardado en: {path}")
    
    @classmethod
    def load(cls, path: str) -> "RandomForest":
        """
        Carga una instancia completa desde disco.
        
        Args:
            path: Ruta al archivo .pkl guardado previamente
        
        Returns:
            Instancia de RandomForest con modelo entrenado listo para query()
        
        Example:
            rf_model = RandomForest.load("src/backend/models/p300/randomforest_20251109_143022.pkl")
            predictions = RandomForest.query(rf_model, X_new)
        """
        with open(path, 'rb') as f:
            instance = pickle.load(f)
        print(f"[RandomForest] Modelo cargado desde: {path}")
        return instance
    
    @staticmethod
    def _generate_model_path(label: str, base_dir: str = "src/backend/models") -> str:
        """
        Genera ruta única para guardar modelo.
        
        Args:
            label: Etiqueta del experimento (e.g., "p300", "inner")
            base_dir: Directorio base para modelos
        
        Returns:
            Ruta completa: "{base_dir}/{label}/randomforest_{timestamp}.pkl"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"randomforest_{timestamp}.pkl"
        return str(Path(base_dir) / label / filename)

    @staticmethod
    def _prepare_xy(
        x_paths: Sequence[str],
        y_paths: Sequence[str],
        metadata_list: Optional[Sequence[dict]] = None,
        *,
        verbose: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        """
        Carga y prepara datos para RandomForest con la misma convención que SVM:
        - X debe venir como 3D (n_frames, features, n_channels) tras el pipeline.
        - Se aplanan los ejes (features, n_channels) para obtener X 2D con un ejemplo por frame.
        - y se aplana a 1D (una etiqueta por frame) y se valida longitud.

        Soporta tanto listas de archivos por frame como archivos agregados (train_X.npy/test_X.npy).
        """
        t_load = time.perf_counter()
        X = _load_and_concat(x_paths)
        y = _load_and_concat(y_paths)
        load_seconds = time.perf_counter() - t_load

        if verbose:
            print(f"[RF._prepare_xy] Cargados: X={len(x_paths)} y={len(y_paths)} en {load_seconds:.3f}s. Shape X={X.shape} y={y.shape}")

        if X.ndim != 3:
            raise ValueError(
                f"RandomForest espera datos 3D post-transform (n_frames, features, n_channels). Recibido {X.shape}"
            )

        n_samples = X.shape[0]
        flat_features = int(np.prod(X.shape[1:]))
        X = X.reshape(n_samples, -1)
        y = np.ravel(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatch X/y: {X.shape[0]} frames vs {y.shape[0]} labels. Cada frame debe tener una etiqueta."
            )

        if verbose:
            print(f"[RF._prepare_xy] X listo: {n_samples} ejemplos, {flat_features} features planos. y shape={y.shape}")

        return X, y

    @classmethod
    def train(
        cls,
        instance: "RandomForest",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
        metadata_train: Optional[List[dict]] = None,
        metadata_test: Optional[List[dict]] = None,
        model_label: Optional[str] = None,
        *,
        verbose: bool = True,
    ) -> EvaluationMetrics:
        """
        Entrena un RandomForest usando las rutas dadas (wrapper legacy).
        
        Este método mantiene compatibilidad con código existente retornando solo EvaluationMetrics.
        Internamente delega a fit() para evitar duplicación de código.

        Args:
            instance: Instancia configurada de RandomForest
            xTest: Lista de rutas a archivos .npy de test
            yTest: Lista de rutas a archivos .npy con etiquetas de test
            xTrain: Lista de rutas a archivos .npy de entrenamiento
            yTrain: Lista de rutas a archivos .npy con etiquetas de entrenamiento
            metadata_train: Lista opcional de diccionarios con metadatos de dimensionality_change para train
                           (incluido por homogeneidad de API, no usado en RandomForest)
            metadata_test: Lista opcional de diccionarios con metadatos para test
                          (incluido por homogeneidad de API, no usado en RandomForest)
            model_label: Etiqueta opcional para auto-guardar (e.g., "p300", "inner").
                        Si se proporciona, guarda automáticamente en src/backend/models/{label}/

        Returns:
            EvaluationMetrics: Solo las métricas de evaluación (para compatibilidad backward)
            
        Note:
            Si necesitas acceso al modelo entrenado y más información, usa fit() en su lugar.
        """
        # Delegar a fit() y extraer solo métricas (evita duplicación de código)
        result = cls.fit(
            instance=instance,
            xTest=xTest,
            yTest=yTest,
            xTrain=xTrain,
            yTrain=yTrain,
            metadata_train=metadata_train,
            metadata_test=metadata_test,
            model_label=model_label,
            verbose=verbose,
        )
        return result.metrics

    @classmethod
    def fit(
        cls,
        instance: "RandomForest",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
        metadata_train: Optional[List[dict]] = None,
        metadata_test: Optional[List[dict]] = None,
        model_label: Optional[str] = None,
        *,
        verbose: bool = True,
    ) -> TrainResult:
        """Entrena y devuelve paquete TrainResult (modelo + métricas).

        No rompe `train()`: es una API paralela opt-in.
        
        Args:
            model_label: Etiqueta opcional para auto-guardar (e.g., "p300", "inner").
                        Si se proporciona, guarda automáticamente en src/backend/models/{label}/
        """
        if verbose:
            print("[RandomForest.fit] Iniciando entrenamiento RandomForest")
            print(f"[RandomForest.fit] Parámetros: n_estimators={instance.n_estimators} max_depth={instance.max_depth} criterion={instance.criterion} class_weight={instance.class_weight}")
            print(f"[RandomForest.fit] Archivos train: X={len(xTrain)} y={len(yTrain)} | test: X={len(xTest)} y={len(yTest)}")

        # 1) Cargar y preparar datos
        t_prepare = time.perf_counter()
        X_train, y_train = cls._prepare_xy(xTrain, yTrain, metadata_list=metadata_train, verbose=verbose)
        X_test, y_test = cls._prepare_xy(xTest, yTest, metadata_list=metadata_test, verbose=verbose)
        prepare_seconds = time.perf_counter() - t_prepare

        if verbose:
            print(f"[RandomForest.fit] Datos preparados en {prepare_seconds:.3f}s | X_train={X_train.shape} y_train={y_train.shape} | X_test={X_test.shape} y_test={y_test.shape}")

        # Validaciones de shape
        if X_train.ndim != 2:
            raise ValueError(f"X_train debe ser 2D tras aplanar; shape={X_train.shape}")
        if X_test.ndim != 2:
            raise ValueError(f"X_test debe ser 2D tras aplanar; shape={X_test.shape}")
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError("y_train/y_test deben ser 1D")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Número de ejemplos en X_train y y_train no coincide: "
                f"X_train={X_train.shape[0]} vs y_train={y_train.shape[0]}"
            )
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"Número de ejemplos en X_test y y_test no coincide: "
                f"X_test={X_test.shape[0]} vs y_test={y_test.shape[0]}"
            )

        # 2) Instanciar modelo sklearn
        model = RandomForestClassifier(
            n_estimators=instance.n_estimators,
            max_depth=instance.max_depth,
            criterion=instance.criterion,
            random_state=instance.random_state,
            class_weight=instance.class_weight,
            n_jobs=instance.n_jobs,
        )

        # 3) Entrenamiento y evaluación
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        if verbose:
            print(f"[RandomForest.fit] Entrenamiento completado en {train_time:.3f}s")
        
        t_eval = time.perf_counter()
        y_pred = model.predict(X_test)
        eval_seconds = time.perf_counter() - t_eval
        if verbose:
            print(f"[RandomForest.fit] Evaluación completada en {eval_seconds:.3f}s")

        # 4) Métricas
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # AUC-ROC (binario o multiclase con OVR)
        try:
            proba = model.predict_proba(X_test)
            auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
        except Exception:
            auc = 0.0

        # 5) Persistir modelo entrenado para query posterior
        instance._model = model

        metrics = EvaluationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            confusion_matrix=cm,
            auc_roc=auc,
            loss=[],  # RandomForest no produce curva de pérdida
            evaluation_time=f"{eval_seconds:.4f}s",
        )

        if verbose:
            print(f"[RandomForest.fit] Métricas: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")
        
        # Auto-guardar si se proporciona label
        if model_label:
            save_path = cls._generate_model_path(model_label)
            if verbose:
                print(f"[RandomForest.fit] Guardando modelo en {save_path}")
            instance.save(save_path)
        
        return TrainResult(
            metrics=metrics,
            model=model,
            model_name="RandomForest",
            training_seconds=float(train_time),
            history=None,  # RandomForest no tiene historia de entrenamiento por época
            hyperparams={
                "n_estimators": instance.n_estimators,
                "max_depth": instance.max_depth,
                "criterion": instance.criterion,
                "random_state": instance.random_state,
                "class_weight": instance.class_weight,
                "n_jobs": instance.n_jobs,
                "n_features": int(X_train.shape[1]),
                "n_classes": int(len(np.unique(y_train))),
            }
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
