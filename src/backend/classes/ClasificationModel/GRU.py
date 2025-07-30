from pydantic import  Field
from backend.classes.ClasificationModel.ClsificationModels import Classifier 


class GRU(Classifier):
    hidden_size: int = Field(128, ge=1, description="Tamaño del vector oculto")
    num_layers: int = Field(2, ge=1, le=10, description="Número de capas GRU")
    dropout: float = Field(0.3, ge=0.0, le=1.0, description="Dropout entre capas")
    bidirectional: bool = Field(True, description="Usar GRU bidireccional")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")