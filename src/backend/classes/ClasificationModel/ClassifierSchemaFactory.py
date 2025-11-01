import os
import json
from typing import Dict, Any, Type
from pydantic import BaseModel
from backend.classes.ClasificationModel.CNN import CNN
from backend.classes.ClasificationModel.LSTM import LSTMNet as LSTM
from backend.classes.ClasificationModel.GRU import GRUNet as GRU
from backend.classes.ClasificationModel.SVM import SVM
from backend.classes.ClasificationModel.SVNN import SVNN
from backend.classes.ClasificationModel.RandomForest import RandomForest
from backend.helpers.mapaValidacion import generar_mapa_validacion_inputs
from dash import callback, Input, Output, State, no_update
from pydantic import ValidationError
from backend.classes.Experiment import Experiment
class ClassifierSchemaFactory:
    """
    In this classs, we define a factory for generating schemas for various classifiers.
    It provides methods to retrieve all classifier schemas and to add a classifier instance
    to an experiment JSON file.
    In adition, it includes a method to register a callback for handling form submissions
    """


    # Available classifiers with their respective classes

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
        """
        Genera esquemas JSON para cada clasificador disponible.
        Retorna un diccionario con el schema JSON de cada modelo.
        """
        schemas = {}
        for key, model in cls.available_classifiers.items():
            try:
                schema = model.model_json_schema()
                schemas[key] = schema
            except Exception as e:
                print(f"⚠️ Skipping {key}: {e.__class__.__name__}: {str(e)}")
                continue
        return schemas
    @classmethod#######################------------------------------------------------------------------------------------------------------------------------------Chane+--
    def add_classifier_to_experiment(cls, directory: str, experiment_id: str, classifier_name: str, classifier_instance: BaseModel) -> str:
        """
        Adds a default instance of the classifier to the experiment JSON file.
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
    



def ClassifierCallbackRegister(boton_id: str, inputs_map: dict):
    available_classifiers = {
        "LSTM": LSTM,
        "GRU": GRU,
        "SVM": SVM,
        "SVNN": SVNN,
        "RandomForest": RandomForest,
        "CNN": CNN
    }

    input_ids = list(inputs_map.keys())

    @callback(
        Output(boton_id, "children"),
        Input(boton_id, "n_clicks"),
        [State(input_id, "value") for input_id in input_ids],
        prevent_initial_call=True
    )
    def formManager(n_clicks, *values, input_ids=input_ids, validadores=inputs_map):
        if not n_clicks:
            return no_update

        # btn-aplicar-{type} donde {type} = "LSTM_p300" | "LSTM_inner" | "LSTM"
        type_raw = boton_id.replace("btn-aplicar-", "")
        parts = type_raw.split("_")
        classifier_name = parts[0]                        # "LSTM"
        ctx = parts[1].lower() if len(parts) > 1 else ""  # "p300" | "inner" | ""

        validatingClass = available_classifiers.get(classifier_name)
        if validatingClass is None:
            print(f"❌ Clasificador '{classifier_name}' no reconocido.")
            return no_update

        # Construir payload desde los inputs: "{type}-{field}"
        data = {}
        for input_id, value in zip(input_ids, values):
            # ejemplo: "LSTM_p300-epochs" -> "epochs"
            try:
                field = input_id.split("-", 1)[1]
            except IndexError:
                continue

            # Preprocesar campos que pueden ser arrays
            # Si el valor es string con comas, convertir a lista de números
            if isinstance(value, str) and "," in value:
                try:
                    # Intentar parsear como lista de números
                    valores_separados = [float(v.strip()) for v in value.split(",")]
                    data[field] = valores_separados
                except (ValueError, AttributeError):
                    # Si falla el parseo, dejar el valor original
                    data[field] = value
            else:
                data[field] = value

        try:
            valid_instance = validatingClass(**data)
            print(f"✅ Datos válidos para {classifier_name} ({ctx}): {valid_instance}")
            validatingClass.train(valid_instance)
            # Enrutar según contexto
            if ctx == "p300":
                print(f"✅ Registrando {classifier_name} como P300Classifier.")
                Experiment.add_P300_classifier(valid_instance)
            elif ctx in ("inner"):
                Experiment.add_inner_speech_classifier(valid_instance)
           

            return no_update

        except ValidationError as e:
            print(f"❌ Errores en {classifier_name} ({ctx}): {e}")
            return no_update
