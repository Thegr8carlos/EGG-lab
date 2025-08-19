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
    @classmethod
    def train(cls, instance: "SVM") -> None:
        """
        Simula el entrenamiento del modelo SVM.
        """
        print(f"[SIMULACIÓN] Entrenando modelo SVM:")
        print(f"  Kernel: {instance.kernel}")
        print(f"  C (penalización): {instance.C}")
        print(f"  Gamma: {instance.gamma}")
        print("[SIMULACIÓN] Entrenamiento completado.")

    @classmethod
    def query(cls, instance: "SVM", input_data: str) -> str:
        """
        Simula la consulta del modelo SVM.
        """
        print(f"[SIMULACIÓN] Ejecutando inferencia sobre entrada: {input_data}")
        print(f"  Usando modelo con kernel={instance.kernel}, C={instance.C}, gamma={instance.gamma}")
        return "[SIMULACIÓN] Resultado de la inferencia: etiqueta_ficticia"
