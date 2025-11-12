from __future__ import annotations

import os
import time
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import Field, BaseModel, field_validator

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.svm import SVC



from backend.classes.ClasificationModel.ClsificationModels import Classifier
from backend.classes.Metrics import EvaluationMetrics
from backend.classes.dataset import _load_and_concat


NDArray = np.ndarray







## Class SVM Definition
class SVM(Classifier):
    kernel: Literal["linear", "rbf", "poly", "sigmoid"] = Field(
        "rbf", description="Tipo de kernel para la SVM"
    )
    C: float = Field(1.0, ge=0.0, description="Término de penalización C (regularización)")
    gamma: Optional[str] = Field(
        "scale",
        description="Escala de gamma ('scale', 'auto') o valor numérico (e.g., '0.01')",
    )
    degree: int = Field(
        3, ge=1, description="Grado del polinomio para kernel 'poly' (ignorado en otros kernels)"
    )
    coef0: float = Field(
        0.0, description="Término independiente en kernel 'poly' y 'sigmoid'"
    )
    shrinking: bool = Field(
        True, description="Usar heurística shrinking para acelerar el entrenamiento"
    )
    tol: float = Field(
        1e-3, gt=0.0, description="Tolerancia para criterio de parada"
    )
    max_iter: int = Field(
        -1, description="Máximo de iteraciones (-1 = sin límite)"
    )
    probability: bool = Field(
        True,
        description="Habilita probabilidades para poder calcular AUC-ROC en binario/multiclase",
    )
    class_weight: Optional[Literal["balanced"]] = Field(
        None, description="Pesos de clases: None o 'balanced' para balanceo automático"
    )

    @field_validator("gamma")
    @classmethod
    def _validate_gamma(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return "scale"
        v = str(v).strip().lower()
        if v in {"scale", "auto"}:
            return v
        # allow numeric-as-string (we'll parse later)
        try:
            float(v)
            return v
        except Exception:
            raise ValueError("gamma debe ser 'scale', 'auto' o un valor numérico como string (e.g. '0.01').")

    # =================================================================================
    # Helpers de metadatos
    # =================================================================================
    @staticmethod
    def extract_metadata_from_experiment(experiment_dict: dict, transform_indices: Optional[List[int]] = None) -> List[dict]:
        """
        Extrae metadatos de dimensionality_change desde un diccionario de Experiment.

        Args:
            experiment_dict: Diccionario con estructura de Experiment (output de experiment.dict())
            transform_indices: Índices de las transformadas a extraer. Si None, extrae todas.

        Returns:
            Lista de diccionarios con metadatos de dimensionality_change

        Ejemplo:
            experiment = Experiment._load_latest_experiment()
            metadata = SVM.extract_metadata_from_experiment(experiment.dict(), [0, 1])
        """
        transforms = experiment_dict.get("transform", [])
        if not transforms:
            return []

        if transform_indices is None:
            transform_indices = list(range(len(transforms)))

        metadata_list = []
        for idx in transform_indices:
            if idx < 0 or idx >= len(transforms):
                raise IndexError(f"Índice de transform fuera de rango: {idx}")

            transform_entry = transforms[idx]
            dim_change = transform_entry.get("dimensionality_change", {})

            # Crear diccionario de metadatos con estructura estándar
            metadata = {
                "output_axes_semantics": dim_change.get("output_axes_semantics", {}),
                "output_shape": dim_change.get("output_shape"),
                "input_shape": dim_change.get("input_shape"),
                "standardized_to": dim_change.get("standardized_to"),
                "transposed_from_input": dim_change.get("transposed_from_input"),
                "orig_was_1d": dim_change.get("orig_was_1d"),
            }
            metadata_list.append(metadata)

        return metadata_list


    @staticmethod
    def _prepare_xy(
        x_paths: Sequence[str],
        y_paths: Sequence[str],
        metadata_list: Optional[Sequence[dict]] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Carga features X y etiquetas y desde archivos .npy.

        REGLA DE NEGOCIO: Todas las señales DEBEN pasar por una transformada.
        Por tanto, X SIEMPRE es 3D con formato: (n_frames, features, n_channels)
        donde axis0 = ejemplos (ventanas/frames).

        Args:
            x_paths: Rutas a archivos .npy con datos post-transform
            y_paths: Rutas a archivos .npy con etiquetas
            metadata_list: REQUERIDO - metadatos de transforms para validar

        Returns:
            Tupla (X, y) donde:
                X: (n_samples, n_features_flat) - cada frame es un ejemplo
                y: (n_samples,) - una etiqueta por frame
        """
        X = _load_and_concat(x_paths)
        y = _load_and_concat(y_paths)

        # Validar que X es 3D (regla de negocio)
        if X.ndim != 3:
            raise ValueError(
                f"Los datos deben ser 3D (n_frames, features, n_channels) después de aplicar transform. "
                f"Recibido shape={X.shape}. "
                f"Asegúrate de aplicar WindowingTransform, FFTTransform, DCTTransform o WaveletTransform."
            )

        # Aplanar: (n_frames, features, n_channels) → (n_frames, features*n_channels)
        # axis0 = n_frames (ejemplos/ventanas)
        n_samples = X.shape[0]
        X = X.reshape(n_samples, -1)

        # Asegurar que y es 1D
        y = np.ravel(y)

        # Verificar longitudes
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatch entre ejemplos y etiquetas: X={X.shape[0]} frames vs y={y.shape[0]} labels. "
                f"Cada frame debe tener una etiqueta."
            )

        return X, y

    @classmethod
    def train(
        cls,
        instance: "SVM",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
        metadata_train: Optional[List[dict]] = None,
        metadata_test: Optional[List[dict]] = None,
    ) -> EvaluationMetrics:
        """
        Trains an SVM using ONLY the file paths provided (ignores any Experiment.dataset).
        Each list must contain one or more .npy files. Multiple files will be concatenated.

        Args:
            instance: Instancia de SVM con parámetros configurados
            xTest: Lista de rutas a archivos .npy de test
            yTest: Lista de rutas a archivos .npy con etiquetas de test
            xTrain: Lista de rutas a archivos .npy de entrenamiento
            yTrain: Lista de rutas a archivos .npy con etiquetas de entrenamiento
            metadata_train: Lista de diccionarios con metadatos de dimensionality_change para train.
                           Cada diccionario debe contener:
                           - 'output_axes_semantics': dict con semántica de ejes
                           - 'output_shape': tuple con forma de salida
                           Ejemplo:
                           [{"output_axes_semantics": {"axis0": "sample", "axis1": "channels"},
                             "output_shape": (1000, 64)}]
            metadata_test: Lista de diccionarios con metadatos para test

        Returns:
            EvaluationMetrics object con accuracy, precision, recall, F1, confusion matrix, AUC-ROC

        Notes:
            - X files must be 2D: (n_samples, n_features) o serán reformateados usando metadatos
            - y files must be 1D or (n_samples, 1)
        """
        # Preparar datos train/test con metadatos
        X_train, y_train = cls._prepare_xy(xTrain, yTrain, metadata_list=metadata_train)
        X_test, y_test = cls._prepare_xy(xTest, yTest, metadata_list=metadata_test)

        # Build gamma parameter
        gamma_param: Union[str, float]
        if instance.gamma in {"scale", "auto"}:
            gamma_param = instance.gamma  # type: ignore[assignment]
        else:
            gamma_param = float(instance.gamma)  # type: ignore[arg-type]

        # Create and train model with all parameters
        model = SVC(
            kernel=instance.kernel,
            C=instance.C,
            gamma=gamma_param,
            degree=instance.degree,
            coef0=instance.coef0,
            shrinking=instance.shrinking,
            tol=instance.tol,
            max_iter=instance.max_iter,
            probability=instance.probability,
            class_weight=instance.class_weight,
        )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # AUC-ROC (works for binary and multiclass if probability=True)
        try:
            proba = model.predict_proba(X_test)
            # For multiclass, use OVR with weighted average
            # Note: sklearn expects y_test as integer labels for multiclass AUC with proba matrix
            auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
        except Exception:
            # If anything goes wrong (rare), fall back to 0.0
            auc = 0.0

        eval_seconds = time.perf_counter() - t0

        # SVM does not produce a loss curve; return empty list to comply with schema.
        metrics = EvaluationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            confusion_matrix=cm,
            auc_roc=auc,
            loss=[],  # No training loss available for sklearn SVC
            evaluation_time=f"{eval_seconds:.4f}s",
        )

        print(f"[SVM] Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")
        return metrics


# ======================= Ejemplo de uso =======================
"""
# Ejemplo 1: Construcción básica de modelo SVM con kernel RBF

from backend.classes.ClasificationModel.SVM import SVM

# Configuración básica con RBF kernel (por defecto)
svm_rbf = SVM(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True
)

# Ejemplo 2: SVM con kernel lineal
svm_linear = SVM(
    kernel="linear",
    C=0.5,
    probability=True,
    class_weight="balanced"  # Útil para datasets desbalanceados
)

# Ejemplo 3: SVM con kernel polinomial
svm_poly = SVM(
    kernel="poly",
    degree=3,           # Grado del polinomio
    C=1.0,
    gamma="scale",
    coef0=1.0,         # Término independiente
    probability=True
)

# Ejemplo 4: SVM con kernel sigmoide
svm_sigmoid = SVM(
    kernel="sigmoid",
    C=1.0,
    gamma="auto",
    coef0=0.0,
    probability=True
)

# Ejemplo 5: Configuración avanzada con todos los parámetros
svm_advanced = SVM(
    kernel="rbf",
    C=10.0,              # Regularización más fuerte
    gamma="0.001",       # Gamma específico como string
    shrinking=True,      # Usar shrinking heuristic
    tol=1e-4,           # Tolerancia más estricta
    max_iter=1000,      # Límite de iteraciones
    probability=True,
    class_weight="balanced"
)

# Ejemplo 6: Entrenamiento con metadatos desde Experiment (RECOMENDADO)
from backend.classes.Experiment import Experiment

# Extraer metadatos desde Experiment
experiment = Experiment._load_latest_experiment()
metadata_train = SVM.extract_metadata_from_experiment(experiment.dict(), transform_indices=[0, 1])
metadata_test = SVM.extract_metadata_from_experiment(experiment.dict(), transform_indices=[0])

# Entrenar modelo
metrics = SVM.train(
    svm_rbf,
    xTest=["path/to/test_features.npy"],
    yTest=["path/to/test_labels.npy"],
    xTrain=["path/to/train_features_1.npy", "path/to/train_features_2.npy"],
    yTrain=["path/to/train_labels_1.npy", "path/to/train_labels_2.npy"],
    metadata_train=metadata_train,
    metadata_test=metadata_test
)

print(f"Accuracy: {metrics.accuracy:.3f}")
print(f"F1-Score: {metrics.f1_score:.3f}")

# Ejemplo 7: Metadatos manuales
metadata_train = [
    {
        "output_axes_semantics": {"axis0": "sample", "axis1": "channels"},
        "output_shape": (1000, 64)  # (n_samples, n_features)
    }
]

metadata_test = [
    {
        "output_axes_semantics": {"axis0": "sample", "axis1": "channels"},
        "output_shape": (200, 64)
    }
]

metrics = SVM.train(
    svm_linear,
    xTest=["path/to/test_features.npy"],
    yTest=["path/to/test_labels.npy"],
    xTrain=["path/to/train_features.npy"],
    yTrain=["path/to/train_labels.npy"],
    metadata_train=metadata_train,
    metadata_test=metadata_test
)

# Ejemplo 8: Sin metadatos (fallback - asume formato correcto)
metrics = SVM.train(
    svm_rbf,
    xTest=["path/to/test_features.npy"],
    yTest=["path/to/test_labels.npy"],
    xTrain=["path/to/train_features.npy"],
    yTrain=["path/to/train_labels.npy"]
)

# Ejemplo 9: Grid search para encontrar mejores parámetros
# Nota: Esto requeriría implementación manual o usar sklearn.model_selection.GridSearchCV

kernels = ["linear", "rbf", "poly"]
C_values = [0.1, 1.0, 10.0]
gamma_values = ["scale", "auto", "0.001", "0.01"]

best_f1 = 0
best_config = None

for kernel in kernels:
    for C in C_values:
        for gamma in gamma_values:
            if kernel == "linear" and gamma != "scale":
                continue  # linear no usa gamma

            svm = SVM(kernel=kernel, C=C, gamma=gamma, probability=True)
            metrics = SVM.train(
                svm,
                xTest=["test.npy"],
                yTest=["test_labels.npy"],
                xTrain=["train.npy"],
                yTrain=["train_labels.npy"]
            )

            if metrics.f1_score > best_f1:
                best_f1 = metrics.f1_score
                best_config = {"kernel": kernel, "C": C, "gamma": gamma}

print(f"Best config: {best_config} with F1={best_f1:.3f}")

# Ejemplo 10: Uso con datos que necesitan reformateo
# Si los datos vienen como (n_features, n_samples), los metadatos ayudarán a transponerlos

metadata_transposed = [
    {
        "output_axes_semantics": {"axis0": "channels", "axis1": "sample"},
        "output_shape": (64, 1000)  # (n_features, n_samples) - se transpondrá automáticamente
    }
]

metrics = SVM.train(
    svm_rbf,
    xTest=["test.npy"],
    yTest=["test_labels.npy"],
    xTrain=["train.npy"],
    yTrain=["train_labels.npy"],
    metadata_train=metadata_transposed,
    metadata_test=metadata_transposed
)
"""
