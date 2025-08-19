from pydantic import  Field
from backend.classes.ClasificationModel.ClsificationModels import Classifier 
from typing import Literal, Optional



class RandomForest(Classifier):
    n_estimators: int = Field(100, ge=1, le=1000, description="Número de árboles")
    max_depth: Optional[int] = Field(
        None,
        ge=1,
        description="Profundidad máxima del árbol (None para ilimitado)"
    )
    criterion: Literal['gini', 'entropy'] = Field('gini', description="Función de impureza")
    
    @classmethod
    def train(cls, instance: "RandomForest") -> None:
        """
        Simula el entrenamiento del modelo Random Forest.
        """
        print(f"[SIMULACIÓN] Entrenando modelo Random Forest:")
        print(f"  Número de árboles (n_estimators): {instance.n_estimators}")
        print(f"  Profundidad máxima: {instance.max_depth}")
        print(f"  Criterio de impureza: {instance.criterion}")
        print("[SIMULACIÓN] Entrenamiento completado.")

    @classmethod
    def query(cls, instance: "RandomForest", input_data: str) -> str:
        """
        Simula la consulta del modelo Random Forest.
        """
        print(f"[SIMULACIÓN] Ejecutando inferencia sobre entrada: {input_data}")
        print(f"  Usando modelo con {instance.n_estimators} árboles, criterio: {instance.criterion}")
        return "[SIMULACIÓN] Resultado de la inferencia: etiqueta_ficticia"
