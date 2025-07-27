from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Literal, List, Dict, Any
from backend.classes.mapaValidacion import generar_mapa_validacion_inputs
from dash import callback, Input, Output, State, no_update
# ---------------------------- BASE ----------------------------

class Classifier(BaseModel):
    epochs: int = Field(
        50,
        ge=1,
        le=1000,
        description="Número de épocas para el entrenamiento"
    )
    batch_size: int = Field(
        32,
        ge=1,
        le=512,
        description="Tamaño del batch (lote) para entrenamiento"
    )


# ---------------------------- MODELOS CLASIFICADORES ----------------------------

class LSTMClassifier(Classifier):
    hidden_size: int = Field(128, ge=1, description="Tamaño del vector oculto")
    num_layers: int = Field(2, ge=1, le=10, description="Número de capas LSTM")
    bidirectional: bool = Field(False, description="Usar LSTM bidireccional")
    dropout: float = Field(0.2, ge=0.0, le=1.0, description="Dropout entre capas")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")


class GRUClassifier(Classifier):
    hidden_size: int = Field(128, ge=1, description="Tamaño del vector oculto")
    num_layers: int = Field(2, ge=1, le=10, description="Número de capas GRU")
    dropout: float = Field(0.3, ge=0.0, le=1.0, description="Dropout entre capas")
    bidirectional: bool = Field(True, description="Usar GRU bidireccional")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")


class SVMClassifier(Classifier):
    kernel: Literal['linear', 'rbf', 'poly', 'sigmoid'] = Field(
        'rbf', description="Tipo de kernel para la SVM"
    )
    C: float = Field(1.0, ge=0.0, description="Término de penalización C")
    gamma: Optional[str] = Field(
        'scale',
        description="Escala de gamma ('scale', 'auto') o valor específico"
    )


class SVNNClassifier(Classifier):
    hidden_size: int = Field(64, ge=1, description="Tamaño del vector oculto")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")
    epochs: int = Field(100, ge=1, le=1000, description="Épocas (sobrescribe default base)")
    batch_size: int = Field(16, ge=1, le=512, description="Tamaño del batch")


class RandomForestClassifier(Classifier):
    n_estimators: int = Field(100, ge=1, le=1000, description="Número de árboles")
    max_depth: Optional[int] = Field(
        None,
        ge=1,
        description="Profundidad máxima del árbol (None para ilimitado)"
    )
    criterion: Literal['gini', 'entropy'] = Field('gini', description="Función de impureza")


class CNNClassifier(Classifier):
    num_filters: int = Field(64, ge=1, le=512, description="Número de filtros por capa conv")
    kernel_size: int = Field(3, ge=1, le=11, description="Tamaño del kernel convolucional")
    pool_size: int = Field(2, ge=1, le=5, description="Tamaño del pooling")
    dropout: float = Field(0.25, ge=0.0, le=1.0, description="Dropout después del pooling")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")


class ClassifierSchemaFactory:
    """
    Genera esquemas detallados para clasificadores.
    """
    available_classifiers = {
        "LSTMClassifier": LSTMClassifier,
        "GRUClassifier": GRUClassifier,
        "SVMClassifier": SVMClassifier,
        "SVNNClassifier": SVNNClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "CNNClassifier": CNNClassifier
    }

    @classmethod
    def get_all_classifier_schemas(cls) -> Dict[str, Dict[str, Any]]:
        schemas = {}
        for key, model in cls.available_classifiers.items():
            schema = model.model_json_schema()
            schemas[key] = schema
        return schemas
    








def registrar_callback(boton_id: str, inputs_map: dict):

    available_classifiers = {
        "LSTMClassifier": LSTMClassifier,
        "GRUClassifier": GRUClassifier,
        "SVMClassifier": SVMClassifier,
        "SVNNClassifier": SVNNClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "CNNClassifier": CNNClassifier
    }
    input_ids = list(inputs_map.keys())

    @callback(
        Output(boton_id, "children"),
        Input(boton_id, "n_clicks"),
        [State(input_id, "value") for input_id in input_ids]
    )
    def manejar_formulario(n_clicks, *values, input_ids=input_ids, validadores=inputs_map):
        if not n_clicks:
            return no_update

        filtro_nombre = boton_id.replace("btn-aplicar-", "")
        clase_validadora = available_classifiers.get(filtro_nombre)

        datos = {}
        for input_id, value in zip(input_ids, values):
            _, field = input_id.split("-", 1)
            datos[field] = value

        try:
            instancia_valida = clase_validadora(**datos) ##             Aqui se instancia la clase para validar de manera automatica.
            #print(f"✅ Datos válidos para {filtro_nombre}: {instancia_valida}")
            """
            -------------------------------------------------------------------------------------------------------------
            Aqui se puede definir Cómo queremos maneajr los experimentos. 
            
            Yo propongo que generemos un auxiliar json llamado Experimeento o algo así
            y para no meternos en problemas de qué y cómo guardar las coasas
            mejor que se mantengan estatica.
            Ya solo haría falta hacer una clase de experimento y su mandero. Esto es basicamente un CRUD
            -------------------------------------------------------------------------------------------------------------
            """
            return no_update
        except ValidationError as e:
            #print(f"❌ Errores en {filtro_nombre}: {e}")
            errores = e.errors()
            msg = "\n".join(f"{err['loc'][0]}: {err['msg']}" for err in errores)
            return no_update
for grupo in generar_mapa_validacion_inputs(ClassifierSchemaFactory.get_all_classifier_schemas()):
    for boton_id, inputs_map in grupo.items():
        registrar_callback(boton_id, inputs_map)


