from pydantic import  Field
from backend.classes.ClasificationModel.ClsificationModels import Classifier 



class CNN(Classifier):
    num_filters: int = Field(64, ge=1, le=512, description="Número de filtros por capa conv")
    kernel_size: int = Field(3, ge=1, le=11, description="Tamaño del kernel convolucional")
    pool_size: int = Field(2, ge=1, le=5, description="Tamaño del pooling")
    dropout: float = Field(0.25, ge=0.0, le=1.0, description="Dropout después del pooling")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")

    @classmethod
    def train(cls, instance: "CNN") -> None:
        """
        Simula el entrenamiento del modelo CNN.
        """
        print(f"[SIMULACIÓN] Entrenando modelo CNN:")
        print(f"  Número de filtros: {instance.num_filters}")
        print(f"  Tamaño del kernel: {instance.kernel_size}")
        print(f"  Tamaño del pooling: {instance.pool_size}")
        print(f"  Dropout: {instance.dropout}")
        print(f"  Learning rate: {instance.learning_rate}")
        print("[SIMULACIÓN] Entrenamiento completado.")

    @classmethod
    def query(cls, instance: "CNN", input_data: str) -> str:
        """
        Simula la consulta del modelo CNN.
        """
        print(f"[SIMULACIÓN] Ejecutando inferencia sobre entrada: {input_data}")
        print(f"  Usando modelo CNN con {instance.num_filters} filtros y kernel_size={instance.kernel_size}")
        return "[SIMULACIÓN] Resultado de la inferencia: etiqueta_ficticia"
