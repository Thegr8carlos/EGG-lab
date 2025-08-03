import os
import json
from typing import Dict, Any, Type
from pydantic import BaseModel
from backend.classes.ClasificationModel.CNN import CNN
from backend.classes.ClasificationModel.LSTM import LSTM
from backend.classes.ClasificationModel.GRU import GRU
from backend.classes.ClasificationModel.SVM import SVM
from backend.classes.ClasificationModel.SVNN import SVNN
from backend.classes.ClasificationModel.RandomForest import RandomForest
from backend.helpers.mapaValidacion import generar_mapa_validacion_inputs
from dash import callback, Input, Output, State, no_update
from pydantic import ValidationError
from backend.classes.Experiment import Experiment
class ClassifierSchemaFactory:
    """
    Genera esquemas detallados para clasificadores.
    """
    available_classifiers = {
        "LSTM": LSTM,
        "GRU": GRU,
        "SVM": SVM,
        "SVNN": SVNN,
        "RandomForest": RandomForest,
        "CNN": CNN
    }
    

    @classmethod
    def get_all_classifier_schemas(cls) -> Dict[str, Dict[str, Any]]:
        schemas = {}
        for key, model in cls.available_classifiers.items():
            schema = model.model_json_schema()
            schemas[key] = schema
        return schemas
    @classmethod
    def add_classifier_to_experiment(cls, directory: str, experiment_id: str, classifier_name: str, classifier_instance: BaseModel) -> str:
        """
        Carga un experimento JSON existente y añade una instancia por defecto del clasificador solicitado
        al campo 'filter'. Guarda el archivo y retorna una descripción del cambio.
        """

        # Validar clasificador
        classifier_class = cls.available_classifiers.get(classifier_name)
        if classifier_class is None:
            raise ValueError(f"Classifier '{classifier_name}' is not supported.")

        # Ruta del experimento
        experiment_filename = f"experiment_{experiment_id}.json"
        experiment_path = os.path.join(directory, experiment_filename)

        if not os.path.exists(experiment_path):
            raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

        # Cargar el JSON del experimento
        with open(experiment_path, "r") as f:
            experiment_data = json.load(f)

        # Asegurarse de que 'filter' sea una lista
        if "classifier" not in experiment_data or not isinstance(experiment_data["classifier"], list):
            experiment_data["classifier"] = []

        # Crear instancia del clasificador
        
        experiment_data["classifier"].append({
            "type": classifier_name,
            "config": classifier_instance.dict()
        })

        # Guardar cambios
        with open(experiment_path, "w") as f:
            json.dump(experiment_data, f, indent=4)

        return f"Added {classifier_name} to experiment {experiment_id}"
    




def registrar_callback(boton_id: str, inputs_map: dict):
    available_classifiers = {
        "LSTM": LSTM,
        "GRU": GRU,
        "SVM": SVM,
        "SVNN": SVNN,
        "RandomForest": RandomForest,
        "CNN": CNN  # ✅ corregido
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

        if clase_validadora is None:
            print(f"❌ Clasificador '{filtro_nombre}' no reconocido.")
            return no_update

        datos = {}
        for input_id, value in zip(input_ids, values):
            _, field = input_id.split("-", 1)
            datos[field] = value

        try:
            instancia_valida = clase_validadora(**datos)
            print(f"✅ Datos válidos para {filtro_nombre}: {instancia_valida}")

            # Lógica de escritura en JSON de experimento
            experiment_id = "3"  # 🔁 puedes reemplazarlo por un State dinámico
            directory = Experiment.get_experiments_dir()

            msg = ClassifierSchemaFactory.add_classifier_to_experiment(
                directory=directory,
                experiment_id=experiment_id,
                classifier_name=filtro_nombre
            )

            print(msg)
            return no_update
        except ValidationError as e:
            print(f"❌ Errores en {filtro_nombre}: {e}")
            errores = e.errors()
            msg = "\n".join(f"{err['loc'][0]}: {err['msg']}" for err in errores)
            return no_update
