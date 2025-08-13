import os
import json
from typing import Dict
from pydantic import BaseModel
from backend.classes.FeatureExtracture.FFDT import FFTTransform
from backend.classes.FeatureExtracture.HilbertsTransform import DCTTransform
from backend.classes.FeatureExtracture.WaveletsTransform import WaveletTransform
from backend.classes.Experiment import Experiment
from dash import callback, Input, Output, State, no_update
from pydantic import ValidationError
from backend.helpers.mapaValidacion import generar_mapa_validacion_inputs
from backend.classes.Experiment import Experiment

class TransformSchemaFactory:
    """
    Factory class to generate schemas for various transforms.
    It provides methods to retrieve all transform schemas and to add a transform instance
    """
    available_transforms = {
        "WaveletTransform": WaveletTransform,
        "FFTTransform": FFTTransform,
        "DCTTransform": DCTTransform
    }

    @classmethod
    def get_all_transform_schemas(cls) -> Dict[str, Dict]:
        schemas = {}
        for key, model in cls.available_transforms.items():
            schema = model.model_json_schema()
            schemas[key] = schema
        return schemas

    @classmethod###################------------------------------------------------------------------------------------------------------------------------------Change+--
    def add_transform_to_experiment(cls, directory: str, experiment_id: str, transform_name: str, transform_instance: BaseModel) -> str:
        """
        Agrega una instancia por defecto de la transformada al experimento indicado.
        Si el archivo no existe, lo crea automáticamente con estructura mínima.
        """
        transform_class = cls.available_transforms.get(transform_name)
        if transform_class is None:
            raise ValueError(f"Transform '{transform_name}' is not supported.")

        # Crear nombre de archivo y path absoluto
        filename = f"experiment_{experiment_id}.json"
        path = os.path.join(directory, filename)

        # Crear el directorio si no existe
        os.makedirs(directory, exist_ok=True)

        # Cargar o crear experimento
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"⚠️ El archivo {path} está vacío o corrupto. Se creará desde cero.")
                    data = {
                        "id": experiment_id,
                        "transform": [],
                        "classifier": [],
                        "filter": [],
                        "evaluation": {}
                    }
        else:
            print(f"🆕 Creando nuevo experimento: {path}")
            data = {
                "id": experiment_id,
                "transform": [],
                "classifier": [],
                "filter": [],
                "evaluation": {}
            }


        # Asegurar estructura
        if "transform" not in data or not isinstance(data["transform"], list):
            data["transform"] = []

        # Crear instancia y agregar
        
        data["transform"].append({
            "type": transform_name,
            "config": transform_instance.dict()
        })

        # Guardar siempre el archivo (ya sea creado o modificado)
        with open(path, "a+", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return f"✅ Transform '{transform_name}' added to experiment_{experiment_id}.json"










def TransformCallbackRegister(boton_id: str, inputs_map: dict):
    """
    Function to register a callback for the transform buttons.
    It generates a callback that validates the inputs and adds the transform to the experiment.
    This function is used to dynamically create callbacks for each transform button, and
    its invoked from the RightColumn page.
    """
    available_transforms = {
        "WaveletTransform": WaveletTransform,
        "FFTTransform": FFTTransform,
        "DCTTransform": DCTTransform
    }

    input_ids = list(inputs_map.keys())

    @callback(
        Output(boton_id, "children"),
        Input(boton_id, "n_clicks"),
        [State(input_id, "value") for input_id in input_ids]
    )
    def formManager(n_clicks, *values, input_ids=input_ids, validadores=inputs_map):
        if not n_clicks:
            return no_update
        # we extract the transform name from the button ID. 
        # this is of the form: btn-aplicar-<transform_name>
        transform_name = boton_id.replace("btn-aplicar-", "")
        # we check if the transform is available
        transform_class = available_transforms.get(transform_name)

        datos = {}
        # We create a instance of the transform with the input values
        for input_id, value in zip(input_ids, values):
            _, field = input_id.split("-", 1)
            datos[field] = value
        #print(datos)

        try:
            instancia_valida = transform_class(**datos)
            print(f"✅ Datos válidos para {transform_name}: {instancia_valida}")

            instancia_valida.apply(instancia_valida)

            Experiment.add_transform_config(instancia_valida)

            return no_update
        except ValidationError as e:
            print(f"❌ Errores en {transform_name}: {e}")
            errores = e.errors()
            msg = "\n".join(f"{err['loc'][0]}: {err['msg']}" for err in errores)
            return no_update


