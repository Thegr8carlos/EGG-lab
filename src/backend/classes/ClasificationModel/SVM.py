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


NDArray = np.ndarray


def _assert_npy_path(p: str) -> None:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Path not found: {p}")
    if not p.lower().endswith(".npy"):
        raise ValueError(f"Expected .npy file, got: {p}")


def _load_and_concat(paths: Sequence[str]) -> NDArray:
    """
    Loads one or more .npy files and concatenates along axis=0.
    Each file must have compatible first dimension.
    """
    if not paths:
        raise ValueError("No paths provided.")
    arrays: List[NDArray] = []
    for p in paths:
        _assert_npy_path(p)
        arr = np.load(p, allow_pickle=False)
        arrays.append(arr)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=0)




## Class SVM Definition
class SVM(Classifier):
    kernel: Literal["linear", "rbf", "poly", "sigmoid"] = Field(
        "rbf", description="Tipo de kernel para la SVM"
    )
    C: float = Field(1.0, ge=0.0, description="Término de penalización C")
    gamma: Optional[str] = Field(
        "scale",
        description="Escala de gamma ('scale', 'auto') o valor numérico (e.g., '0.01')",
    )
    probability: bool = Field(
        True,
        description="Habilita probabilidades para poder calcular AUC-ROC en binario/multiclase",
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

    @staticmethod
    def _prepare_xy(
        x_paths: Sequence[str],
        y_paths: Sequence[str],
    ) -> Tuple[NDArray, NDArray]:
        """
        Loads features X and labels y from the given .npy path lists.
        Accepts one or multiple shards for each split and concatenates them.
        y is expected to be 1D or (n,1); will be raveled to 1D.
        """
        X = _load_and_concat(x_paths)
        y = _load_and_concat(y_paths)
        y = np.ravel(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: X={X.shape[0]} vs y={y.shape[0]}")
        return X, y

    @classmethod
    def train(
        cls,
        instance: "SVM",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
    ) -> EvaluationMetrics:
        """
        Trains an SVM using ONLY the file paths provided (ignores any Experiment.dataset).
        Each list must contain one or more .npy files. Multiple files will be concatenated.

        - X files must be 2D: (n_samples, n_features)
        - y files must be 1D or (n_samples, 1)
        - Returns an EvaluationMetrics object with standard scores.
        """
        # We prepare data train/test
        X_train, y_train = cls._prepare_xy(xTrain, yTrain)
        X_test, y_test = cls._prepare_xy(xTest, yTest)

        # Build gamma parameter
        gamma_param: Union[str, float]
        if instance.gamma in {"scale", "auto"}:
            gamma_param = instance.gamma  # type: ignore[assignment]
        else:
            gamma_param = float(instance.gamma)  # type: ignore[arg-type]

        # Create and train model
        model = SVC(
            kernel=instance.kernel,
            C=instance.C,
            gamma=gamma_param,
            probability=instance.probability,
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

        return metrics
