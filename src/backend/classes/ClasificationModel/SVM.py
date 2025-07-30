from pydantic import  Field
from backend.classes.ClasificationModel.ClsificationModels import Classifier 
from typing import Literal, Optional



class SVM(Classifier):
    kernel: Literal['linear', 'rbf', 'poly', 'sigmoid'] = Field(
        'rbf', description="Tipo de kernel para la SVM"
    )
    C: float = Field(1.0, ge=0.0, description="Término de penalización C")
    gamma: Optional[str] = Field(
        'scale',
        description="Escala de gamma ('scale', 'auto') o valor específico"
    )