from pydantic import BaseModel, Field
from typing import List

class EvaluationMetrics(BaseModel):
    accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's accuracy score"
    )
    precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's precision score"
    )
    recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's recall score"
    )
    f1_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's F1 score"
    )
    confusion_matrix: List[List[int]] = Field(
        ...,
        description="Confusion matrix of the model"
    )
    auc_roc: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Area under the ROC curve"
    )
    loss: List[float] = Field(
        ...,
        description="List of loss values during evaluation"
    )
    evaluation_time: str = Field(
        ...,
        description="Time taken to evaluate the model"
    )
