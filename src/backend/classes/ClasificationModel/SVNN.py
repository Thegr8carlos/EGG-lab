from pydantic import  Field
from backend.classes.ClasificationModel.ClsificationModels import Classifier 


class SVNN(Classifier):
    hidden_size: int = Field(64, ge=1, description="Tamaño del vector oculto")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")
    epochs: int = Field(100, ge=1, le=1000, description="Épocas (sobrescribe default base)")
    batch_size: int = Field(16, ge=1, le=512, description="Tamaño del batch")