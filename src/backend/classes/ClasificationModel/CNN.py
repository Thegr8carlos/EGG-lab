from pydantic import  Field
from backend.classes.ClasificationModel.ClsificationModels import Classifier 



class CNN(Classifier):
    num_filters: int = Field(64, ge=1, le=512, description="Número de filtros por capa conv")
    kernel_size: int = Field(3, ge=1, le=11, description="Tamaño del kernel convolucional")
    pool_size: int = Field(2, ge=1, le=5, description="Tamaño del pooling")
    dropout: float = Field(0.25, ge=0.0, le=1.0, description="Dropout después del pooling")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")