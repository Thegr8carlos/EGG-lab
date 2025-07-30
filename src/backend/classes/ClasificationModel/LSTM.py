from pydantic import  Field
from backend.classes.ClasificationModel.ClsificationModels import Classifier 


class LSTM(Classifier):
    hidden_size: int = Field(128, ge=1, description="Tamaño del vector oculto")
    num_layers: int = Field(2, ge=1, le=10, description="Número de capas LSTM")
    bidirectional: bool = Field(False, description="Usar LSTM bidireccional")
    dropout: float = Field(0.2, ge=0.0, le=1.0, description="Dropout entre capas")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")