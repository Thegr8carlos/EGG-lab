from pydantic import  Field
from backend.classes.ClasificationModel.ClsificationModels import Classifier 
from typing import List

class SVNN(Classifier):
    hidden_size: int = Field(64, ge=1, description="Tamaño del vector oculto")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")
    epochs: int = Field(100, ge=1, le=1000, description="Épocas (sobrescribe default base)")
    batch_size: int = Field(16, ge=1, le=512, description="Tamaño del batch")
    @classmethod
    def train(cls, instance: "SVNN", xTest: List[str], yTest: List[str], xTrain: List[str], yTrain: List[str]) -> None:
        """
        This funciton trains the SVNN model. 
        Recieves training and testing data as a list of strings.
        Each string represents a path to a data file. 
        In this case, we validate that de file paths exist, its type is npy
        and we check the last experiment to get the adition information needed for the proceso
        like filters and tranforms applied.
        From this, we calculate the input size for the model.
        Finally, we considerate the yTrain and yTest as a vector of labels,
        where each register is a label for each npy data file.
        """
        print(f"[SIMULACIÓN] Entrenando modelo SVNN:")
        print(f"  Hidden size: {instance.hidden_size}")
        print(f"  Learning rate: {instance.learning_rate}")
        print(f"  Épocas: {instance.epochs}")
        print(f"  Batch size: {instance.batch_size}")
        print("[SIMULACIÓN] Entrenamiento completado.")

    @classmethod
    def query(cls, instance: "SVNN", input_data: str) -> str:
        """
        Simula la consulta del modelo SVNN.
        """
        print(f"[SIMULACIÓN] Ejecutando inferencia sobre entrada: {input_data}")
        print(f"  Usando modelo con hidden_size={instance.hidden_size}, lr={instance.learning_rate}")
        return "[SIMULACIÓN] Resultado de la inferencia: etiqueta_ficticia"
