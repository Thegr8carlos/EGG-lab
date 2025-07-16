from pydantic import BaseModel
from typing import Optional, Literal, List, Dict, Any


# ---------------------------- BASE ----------------------------

class Classifier(BaseModel):
    epochs: int = 50
    batch_size: int = 32


# ---------------------------- MODELOS CLASIFICADORES ----------------------------

class LSTMClassifier(Classifier):
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.2
    learning_rate: float = 0.001


class GRUClassifier(Classifier):
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    learning_rate: float = 0.001


class SVMClassifier(Classifier):
    kernel: Literal['linear', 'rbf', 'poly', 'sigmoid'] = 'rbf'
    C: float = 1.0
    gamma: Optional[str] = 'scale'


class SVNNClassifier(Classifier):
    hidden_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 100  # se sobreescribe default
    batch_size: int = 16


class RandomForestClassifier(Classifier):
    n_estimators: int = 100
    max_depth: Optional[int] = None
    criterion: Literal['gini', 'entropy'] = 'gini'


class CNNClassifier(Classifier):
    num_filters: int = 64
    kernel_size: int = 3
    pool_size: int = 2
    dropout: float = 0.25
    learning_rate: float = 0.001


class ClassifierSchemaFactory:
    """
    Genera esquemas detallados para clasificadores.
    """
    available_classifiers = {
        "lstm": LSTMClassifier,
        "gru": GRUClassifier,
        "svm": SVMClassifier,
        "svnn": SVNNClassifier,
        "random_forest": RandomForestClassifier,
        "cnn": CNNClassifier
    }

    @classmethod
    def get_all_classifier_schemas(cls) -> Dict[str, Dict[str, Any]]:
        schemas = {}
        for key, model in cls.available_classifiers.items():
            schema = model.model_json_schema()
            schemas[key] = schema
        return schemas
if __name__ == "__main__":
    from pprint import pprint

    print("🧠 Esquemas de Modelos de Clasificación:")
    pprint(ClassifierSchemaFactory.get_all_classifier_schemas())
