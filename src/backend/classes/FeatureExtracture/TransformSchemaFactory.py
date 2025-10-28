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
        schemas: Dict[str, Dict] = {}
        for name, model in cls.available_transforms.items():
            # Pydantic v2
            try:
                schema = model.model_json_schema()
            except AttributeError:
                # Fallback Pydantic v1
                schema = model.schema()

            # Remover 'id' del schema publicado
            props = schema.get("properties") or {}
            if "id" in props:
                props.pop("id", None)
                if not props:
                    schema.pop("properties", None)
                else:
                    schema["properties"] = props

            req = schema.get("required")
            if isinstance(req, list) and "id" in req:
                new_req = [x for x in req if x != "id"]
                if new_req:
                    schema["required"] = new_req
                else:
                    schema.pop("required", None)

            schemas[name] = schema
        return schemas


    @classmethod
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
        [
            Output(boton_id, "children"),
            Output("transformed-signal-store-extractores", "data", allow_duplicate=True)
        ],
        Input(boton_id, "n_clicks"),
        [
            State(input_id, "value") for input_id in input_ids
        ] + [
            State("selected-file-path", "data"),
            State("selected-dataset", "data")
        ],
        prevent_initial_call=True
    )
    def formManager(n_clicks, *values, input_ids=input_ids, validadores=inputs_map):
        if not n_clicks:
            return no_update, no_update

        # Extraer selected_file_path y selected_dataset del final de values
        *field_values, event_file_path, dataset_name = values

        if not event_file_path:
            print(f"❌ No hay archivo seleccionado para aplicar {transform_name}")
            return no_update, no_update

        # we extract the transform name from the button ID.
        # this is of the form: btn-aplicar-<transform_name>
        transform_name = boton_id.replace("btn-aplicar-", "")
        # we check if the transform is available
        transform_class = available_transforms.get(transform_name)

        if transform_class is None:
            print(f"❌ Transform '{transform_name}' no encontrada")
            return no_update, no_update

        datos = {}
        experiment = Experiment._load_latest_experiment()
        prev_id = Experiment._extract_last_id_from_list(experiment.filters)
        new_id = prev_id + 1  # if prev_id == -1 -> 0

        for input_id, value in zip(input_ids, field_values):
            _, field = input_id.split("-", 1)
            # Preprocesar campos que pueden ser arrays
            # Si el valor es string con comas, convertir a lista de números
            if isinstance(value, str) and "," in value:
                try:
                    # Intentar parsear como lista de números
                    valores_separados = [float(v.strip()) for v in value.split(",")]
                    datos[field] = valores_separados
                except (ValueError, AttributeError):
                    # Si falla el parseo, dejar el valor original
                    datos[field] = value
            else:
                datos[field] = value
        datos["id"] = str(new_id)  # ✅ Convertir a string

        try:
            from pathlib import Path
            import numpy as np
            import time

            instancia_valida = transform_class(**datos)
            print(f"✅ Datos válidos para {transform_name}: {instancia_valida}")
            Experiment.add_transform_config(instancia_valida)

            # Preparar directorios
            p_in = Path(event_file_path)
            dir_out = p_in.parent / "transformed"
            dir_labels_out = p_in.parent / "transformed_labels"
            labels_dir = p_in.parent  # Las etiquetas originales están en el mismo directorio

            dir_out.mkdir(parents=True, exist_ok=True)
            dir_labels_out.mkdir(parents=True, exist_ok=True)

            # Aplicar transformada
            success = transform_class.apply(
                instancia_valida,
                file_path_in=str(p_in),
                directory_path_out=str(dir_out),
                labels_directory=str(labels_dir),
                labels_out_path=str(dir_labels_out)
            )

            if not success:
                print(f"❌ La transformada {transform_name} no se aplicó correctamente")
                return no_update, no_update

            # Construir path del archivo transformado
            transform_suffixes = {
                "WaveletTransform": "wavelet",
                "FFTTransform": "fft",
                "DCTTransform": "dct"
            }
            suffix = transform_suffixes.get(transform_name, "transformed")
            out_name = f"{p_in.stem}_{suffix}_{new_id}.npy"
            out_path = dir_out / out_name

            if not out_path.exists():
                print(f"❌ No se encontró el archivo transformado: {out_path}")
                return no_update, no_update

            # Cargar datos transformados
            arr = np.load(str(out_path), allow_pickle=False)

            # Obtener nombres de canales
            try:
                from backend.classes.dataset import Dataset
                channel_names = Dataset.get_all_channel_names(dataset_name)
                channel_names_for_plots = channel_names if channel_names else [f"Ch{i}" for i in range(arr.shape[1] if arr.ndim == 2 else 1)]
            except:
                channel_names_for_plots = [f"Ch{i}" for i in range(arr.shape[1] if arr.ndim == 2 else 1)]

            # Extraer información del archivo
            import os
            file_name = os.path.basename(event_file_path)
            parts = file_name.split("[")
            event_class = parts[0].strip() if len(parts) > 0 else "desconocida"
            session = parts[1].split("]")[0] if len(parts) > 1 else ""

            # Crear payload con datos transformados
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            # Asegurar forma (n_channels, n_times)
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T

            matrix_list = [row.tolist() for row in arr]

            # Obtener colores de clase
            from shared.class_colors import get_class_color
            class_color = get_class_color(event_class)

            transformed_payload = {
                "matrix": matrix_list,
                "channel_names": channel_names_for_plots,
                "file_name": file_name,
                "event_class": event_class,
                "session": session,
                "class_name": event_class,
                "class_colors": {event_class: class_color},
                "transformed_file_path": str(out_path),
                "transform_type": transform_name,
                "ts": time.time()
            }

            print(f"✅ Transformada aplicada: {out_path}")
            print(f"✅ Shape transformado: {arr.shape}")

            return no_update, transformed_payload

        except ValidationError as e:
            print(f"❌ Errores en {transform_name}: {e}")
            return no_update, no_update
        except Exception as e:
            print(f"❌ Error al aplicar {transform_name}: {e}")
            import traceback
            traceback.print_exc()
            return no_update, no_update


