import os
import json
from typing import Dict
from pydantic import BaseModel
from backend.classes.FeatureExtracture.FFDT import FFTTransform
from backend.classes.FeatureExtracture.HilbertsTransform import DCTTransform
from backend.classes.FeatureExtracture.WaveletsTransform import WaveletTransform
from backend.classes.FeatureExtracture.WindowingTransform import WindowingTransform
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
        "DCTTransform": DCTTransform,
        "WindowingTransform": WindowingTransform
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return f"✅ Transform '{transform_name}' added to experiment_{experiment_id}.json"










def TransformCallbackRegister(boton_id: str, inputs_map: dict, model_type: str = "legacy"):
    """
    Function to register a callback for the transform buttons.
    It generates a callback that validates the inputs and adds the transform to the experiment.
    This function is used to dynamically create callbacks for each transform button, and
    its invoked from the RightColumn page.

    Args:
        boton_id: ID of the button (e.g., "btn-aplicar-WaveletTransform_p300")
        inputs_map: Map of input IDs to validation functions
        model_type: Type of model ("legacy", "p300", or "inner") to determine where to save the transform
    """
    available_transforms = {
        "WaveletTransform": WaveletTransform,
        "FFTTransform": FFTTransform,
        "DCTTransform": DCTTransform,
        "WindowingTransform": WindowingTransform
    }

    input_ids = list(inputs_map.keys())

    # Determine prefix from boton_id (e.g., "btn-aplicar-WaveletTransform_p300" -> "p300")
    # or use model_type mapping
    prefix_map = {
        "p300": "p300",
        "inner": "inner",
        "legacy": "extractores"
    }
    prefix = prefix_map.get(model_type, "extractores")

    @callback(
        [
            Output(boton_id, "children"),
            Output(f"transformed-signal-store-{prefix}", "data", allow_duplicate=True),
            Output(f"pipeline-update-trigger-{prefix}", "data", allow_duplicate=True)
        ],
        Input(boton_id, "n_clicks"),
        [
            State(input_id, "value") for input_id in input_ids
        ] + [
            State(f"signal-store-{prefix}", "data"),
            State("selected-dataset", "data"),
            State(f"pipeline-update-trigger-{prefix}", "data")
        ],
        prevent_initial_call=True
    )
    def formManager(n_clicks, *values, input_ids=input_ids, validadores=inputs_map):
        if not n_clicks:
            return no_update, no_update, no_update

        # Extraer signal_data, dataset_name y trigger del final de values
        *field_values, signal_data, dataset_name, trigger_value = values

        if not signal_data or "source" not in signal_data:
            print(f"❌ No hay señal cargada en signal-store-{prefix}")
            return "❌ No hay señal cargada", no_update, no_update

        # Obtener path del evento actual (ya incluye prefijo Aux/ si es necesario)
        event_file_path = signal_data.get("source")

        # we extract the transform name from the button ID.
        # this is of the form: btn-aplicar-<transform_name> or btn-aplicar-<transform_name>_p300
        transform_name_full = boton_id.replace("btn-aplicar-", "")
        # Remove suffix if present (_p300 or _inner)
        transform_name = transform_name_full.replace("_p300", "").replace("_inner", "")
        # we check if the transform is available
        transform_class = available_transforms.get(transform_name)

        if transform_class is None:
            print(f"❌ Transform '{transform_name}' no encontrada")
            return "❌ Transform no encontrada", no_update, no_update

        datos = {}
        experiment = Experiment._load_latest_experiment()
        prev_id = Experiment._extract_last_id_from_list(experiment.filters)
        new_id = prev_id + 1  # if prev_id == -1 -> 0

        for input_id, value in zip(input_ids, field_values):
            _, field = input_id.split("-", 1)
            # Convertir string "None" a Python None (para campos Optional[Literal[..., None]])
            if isinstance(value, str) and value == "None":
                datos[field] = None
            # Preprocesar campos que pueden ser arrays
            # Si el valor es string con comas, convertir a lista de números
            elif isinstance(value, str) and "," in value:
                try:
                    # Intentar parsear como lista de números
                    valores_separados = [float(v.strip()) for v in value.split(",")]
                    datos[field] = valores_separados
                except (ValueError, AttributeError):
                    # Si falla el parseo, dejar el valor original
                    datos[field] = value
            else:
                datos[field] = value

        # ✅ Agregar id y sp (frecuencia de muestreo)
        datos["id"] = str(new_id)

        # Obtener sp del signal_data si no viene del formulario
        if "sp" not in datos or datos.get("sp") is None:
            sfreq = signal_data.get("sfreq", 1024.0)
            datos["sp"] = float(sfreq)
            print(f"[TransformSchemaFactory] 📊 Usando frecuencia de muestreo: {sfreq} Hz")

        try:
            from pathlib import Path
            import numpy as np
            import time

            instancia_valida = transform_class(**datos)
            print(f"✅ Datos válidos para {transform_name}: {instancia_valida}")

            # Save transform to the appropriate location based on model_type
            if model_type == "p300":
                Experiment.set_P300_transform(instancia_valida)
                print(f"💾 Transformación guardada en P300Classifier.transform")
            elif model_type == "inner":
                Experiment.set_inner_speech_transform(instancia_valida)
                print(f"💾 Transformación guardada en innerSpeachClassifier.transform")
            else:  # legacy
                Experiment.add_transform_config(instancia_valida)
                print(f"💾 Transformación guardada en experiment.transform (legacy)")

            # Preparar directorios
            p_in = Path(event_file_path)
            dir_out = p_in.parent / "transformed"
            dir_labels_out = p_in.parent / "transformed_labels"

            dir_out.mkdir(parents=True, exist_ok=True)
            dir_labels_out.mkdir(parents=True, exist_ok=True)

            # ✅ Generar etiquetas temporales para eventos individuales
            # Extraer clase del nombre del archivo (ej: "abajo[439.357]{441.908}.npy" → "abajo")
            file_name = p_in.stem
            event_class = file_name.split('[')[0].strip() if '[' in file_name else file_name

            # Crear directorio temporal para etiquetas
            labels_dir = p_in.parent / "temp_labels"
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Generar array de etiquetas (todas con la misma clase)
            arr_signal = np.load(str(p_in), allow_pickle=False)
            if arr_signal.ndim == 1:
                n_samples = arr_signal.shape[0]
            elif arr_signal.ndim == 2:
                n_samples = arr_signal.shape[1]  # (n_channels, n_times)
            else:
                n_samples = arr_signal.shape[0]

            labels_array = np.array([event_class] * n_samples, dtype=str)

            # Guardar etiquetas temporales
            temp_labels_file = labels_dir / p_in.name
            np.save(str(temp_labels_file), labels_array)
            print(f"📝 Etiquetas temporales generadas: {temp_labels_file}")

            # Aplicar transformada (con manejo de diferentes firmas de apply)
            try:
                # Intentar primero con labels_out_path (WaveletTransform, DCTTransform)
                success = transform_class.apply(
                    instancia_valida,
                    file_path_in=str(p_in),
                    directory_path_out=str(dir_out),
                    labels_directory=str(labels_dir),
                    labels_out_path=str(dir_labels_out)
                )
            except TypeError as e:
                # Si falla, intentar con dir_out_labels (FFTTransform)
                if "labels_out_path" in str(e):
                    success = transform_class.apply(
                        instancia_valida,
                        file_path_in=str(p_in),
                        directory_path_out=str(dir_out),
                        labels_directory=str(labels_dir),
                        dir_out_labels=str(dir_labels_out)
                    )
                else:
                    raise

            # Limpiar archivo temporal después de aplicar
            if temp_labels_file.exists():
                temp_labels_file.unlink()
                print(f"🧹 Limpiado archivo temporal: {temp_labels_file}")

            if not success:
                print(f"❌ La transformada {transform_name} no se aplicó correctamente")
                return "❌ Error al aplicar transform", no_update, no_update

            # Construir path del archivo transformado
            transform_suffixes = {
                "WaveletTransform": "wavelet",
                "FFTTransform": "fft",
                "DCTTransform": "dct",
                "WindowingTransform": "window"
            }
            suffix = transform_suffixes.get(transform_name, "transformed")

            # Buscar el archivo más reciente con ese sufijo (puede tener diferente ID)
            # Escapar caracteres especiales en glob pattern
            import glob
            import re

            # Escapar los caracteres especiales [], {}, etc.
            stem_escaped = re.escape(p_in.stem)
            pattern_safe = f"{stem_escaped}_{suffix}_*.npy"

            # Buscar todos los archivos en el directorio que coincidan
            all_files = list(dir_out.glob("*.npy"))
            matching_files = [
                f for f in all_files
                if f.stem.startswith(f"{p_in.stem}_{suffix}_")
            ]
            matching_files = sorted(matching_files, key=lambda x: x.stat().st_mtime, reverse=True)

            if matching_files:
                out_path = matching_files[0]
                print(f"✅ Archivo transformado encontrado: {out_path}")
            else:
                # Intentar con el ID esperado como fallback
                out_name = f"{p_in.stem}_{suffix}_{new_id}.npy"
                out_path = dir_out / out_name

                if not out_path.exists():
                    print(f"❌ No se encontró el archivo transformado: {out_path}")
                    print(f"   Archivos en directorio: {[f.name for f in all_files[:5]]}")
                    return "❌ Archivo no encontrado", no_update, no_update

            # Cargar datos transformados
            arr = np.load(str(out_path), allow_pickle=False)

            # ✅ Manejo de arrays 3D (transformadas ventaneadas)
            # Formato: (n_frames, frame_size, n_channels) → (n_channels, n_frames * frame_size)
            if arr.ndim == 3:
                n_frames, frame_size, n_channels = arr.shape
                print(f"📊 Array 3D detectado: {arr.shape} (frames, frame_size, canales)")

                # Paso 1: Transponer → (n_channels, n_frames, frame_size)
                arr_transposed = arr.transpose(2, 0, 1)

                # Paso 2: Concatenar frames → (n_channels, n_frames * frame_size)
                arr = arr_transposed.reshape(n_channels, n_frames * frame_size)

                print(f"✅ Array 3D concatenado: {arr.shape} (canales x tiempo)")

            # Obtener nombres de canales
            try:
                from backend.classes.dataset import Dataset
                channel_names = Dataset.get_all_channel_names(dataset_name)
                channel_names_for_plots = channel_names if channel_names else [f"Ch{i}" for i in range(arr.shape[0] if arr.ndim == 2 else 1)]
            except:
                channel_names_for_plots = [f"Ch{i}" for i in range(arr.shape[0] if arr.ndim == 2 else 1)]

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
            if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
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

            # Incrementar trigger para actualizar el historial
            new_trigger = (trigger_value or 0) + 1

            return no_update, transformed_payload, new_trigger

        except ValidationError as e:
            print(f"❌ Errores en {transform_name}: {e}")
            errores = e.errors()
            # Construir mensaje de error legible
            error_fields = [err['loc'][0] for err in errores if err['loc']]
            msg_short = f"❌ Error: {', '.join(error_fields)}" if error_fields else "❌ Error de validación"
            msg_full = "\n".join(f"{err['loc'][0]}: {err['msg']}" for err in errores)
            print(f"❌ Detalles: {msg_full}")
            return msg_short, no_update, no_update
        except Exception as e:
            print(f"❌ Error al aplicar {transform_name}: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error inesperado", no_update, no_update


