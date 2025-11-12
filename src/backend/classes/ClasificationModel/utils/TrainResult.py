from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from backend.classes.Metrics import EvaluationMetrics


class TrainResult(BaseModel):
    """Paquete estándar devuelto por el nuevo flujo de entrenamiento.

    Contiene métricas calculadas, el objeto de modelo entrenado (sklearn, keras, etc.),
    y metadatos opcionales como historia de entrenamiento y hyperparams usados.

    Esta clase permite un contrato uniforme sin romper la firma legacy de `train()`
    que seguira devolviendo solo `EvaluationMetrics` para compatibilidad.
    """

    metrics: EvaluationMetrics = Field(..., description="Métricas evaluadas del conjunto de test.")
    model: Any = Field(..., description="Instancia del modelo entrenado (sklearn, keras.Model, etc.)")
    model_name: str = Field(..., description="Nombre lógico del modelo (ej. 'LSTM', 'SVM').")
    training_seconds: float = Field(..., ge=0, description="Duración total del entrenamiento en segundos.")
    history: Optional[Dict[str, list]] = Field(None, description="Historia de entrenamiento (loss/acc por época si aplica).")
    hyperparams: Optional[Dict[str, Any]] = Field(None, description="Hiperparámetros principales usados en entrenamiento.")

    def summary(self) -> Dict[str, Any]:
        """Resumen compacto para logging/serialización rápida."""
        return {
            "model_name": self.model_name,
            "accuracy": self.metrics.accuracy,
            "f1_score": self.metrics.f1_score,
            "auc_roc": self.metrics.auc_roc,
            "training_seconds": self.training_seconds,
        }

    # Accesos directos convenientes
    @property
    def confusion_matrix(self):  # noqa: D401
        return self.metrics.confusion_matrix

    @property
    def loss_curve(self):  # noqa: D401
        return self.metrics.loss

    def predict(self, X) -> Any:
        """Intento genérico de predecir usando el objeto subyacente si expone predict()."""
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise AttributeError("El modelo subyacente no expone método predict().")
