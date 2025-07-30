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