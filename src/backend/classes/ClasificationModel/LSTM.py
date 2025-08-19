from pydantic import  Field
from backend.classes.ClasificationModel.ClsificationModels import Classifier 


class LSTM(Classifier):
    hidden_size: int = Field(128, ge=1, description="Tamaño del vector oculto")
    num_layers: int = Field(2, ge=1, le=10, description="Número de capas LSTM")
    bidirectional: bool = Field(False, description="Usar LSTM bidireccional")
    dropout: float = Field(0.2, ge=0.0, le=1.0, description="Dropout entre capas")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")

    @classmethod
    def train(cls, instance: "LSTM") -> None:
        """
        Simula el entrenamiento del modelo LSTM.
        """
        print(f"[SIMULACIÓN] Entrenando modelo LSTM:")
        print(f"  Hidden size: {instance.hidden_size}")
        print(f"  Número de capas: {instance.num_layers}")
        print(f"  Bidireccional: {instance.bidirectional}")
        print(f"  Dropout: {instance.dropout}")
        print(f"  Learning rate: {instance.learning_rate}")
        print("[SIMULACIÓN] Entrenamiento completado.")
    

    @classmethod
    def query(cls, instance: "LSTM", input_data: str) -> str:
        """
        Simula la consulta del modelo LSTM.
        """
        print(f"[SIMULACIÓN] Ejecutando inferencia sobre entrada: {input_data}")
        print(f"  Usando modelo LSTM con hidden_size={instance.hidden_size}, capas={instance.num_layers}")
        return "[SIMULACIÓN] Resultado de la inferencia: etiqueta_ficticia"
