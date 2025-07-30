import os
import json
from typing import Dict
from pydantic import BaseModel
from backend.classes.FeatureExtracture.FFDT import FFTTransform
from backend.classes.FeatureExtracture.HilbertsTransform import DCTTransform
from backend.classes.FeatureExtracture.WaveletsTransform import WaveletTransform
from dash import callback, Input, Output, State, no_update
from pydantic import ValidationError
from backend.helpers.mapaValidacion import generar_mapa_validacion_inputs

class TransformSchemaFactory:
    """
    Genera esquemas detallados para transformadas.
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

    @classmethod
    def add_transform_to_experiment(cls, directory: str, experiment_id: str, transform_name: str) -> str:
        """
        Agrega una instancia por defecto de la transformada al experimento indicado.
        Si el archivo no existe, lo crea automáticamente con estructura mínima.
        """
        transform_class = cls.available_transforms.get(transform_name)
        if transform_class is None:
            raise ValueError(f"Transform '{transform_name}' is not supported.")

        filename = f"experiment_{experiment_id}.json"
        path = os.path.join(directory, filename)

        os.makedirs(directory, exist_ok=True)

        # Si el archivo NO existe, lo creamos con estructura mínima
        if not os.path.exists(path):
            print(f"⚠️ Archivo no encontrado. Creando nuevo experimento: {path}")
            data = {
                "id": experiment_id,
                "transform": [],
                "classifier": [],
                "filter": [],
                "evaluation": {}
            }
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "transform" not in data or not isinstance(data["transform"], list):
                data["transform"] = []

        # Crear instancia y agregarla
        transform_instance: BaseModel = transform_class()
        data["transform"].append({
            "type": transform_name,
            "config": transform_instance.dict()
        })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return f"✅ Added {transform_name} to experiment_{experiment_id}.json"





def get_experiments_dir() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
    experiments_dir = os.path.join(base, "backend", "Experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    return experiments_dir



def registrar_callback(boton_id: str, inputs_map: dict):
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
    def manejar_formulario(n_clicks, *values, input_ids=input_ids, validadores=inputs_map):
        if not n_clicks:
            return no_update

        transform_name = boton_id.replace("btn-aplicar-", "")
        transform_class = available_transforms.get(transform_name)

        datos = {}
        for input_id, value in zip(input_ids, values):
            _, field = input_id.split("-", 1)
            datos[field] = value

        try:
            instancia_valida = transform_class(**datos)
            print(f"✅ Datos válidos para {transform_name}: {instancia_valida}")

            # 🧠 Lógica de integración con JSON de experimento
            experiment_id = "3"  # Este valor podría venir de otro State
            directory = get_experiments_dir()

            msg = TransformSchemaFactory.add_transform_to_experiment(directory, experiment_id, transform_name)
            print(msg)

            return no_update
        except ValidationError as e:
            print(f"❌ Errores en {transform_name}: {e}")
            errores = e.errors()
            msg = "\n".join(f"{err['loc'][0]}: {err['msg']}" for err in errores)
            return no_update

# Registrar todos los callbacks
for grupo in generar_mapa_validacion_inputs(TransformSchemaFactory.get_all_transform_schemas()):
    for boton_id, inputs_map in grupo.items():
        registrar_callback(boton_id, inputs_map)