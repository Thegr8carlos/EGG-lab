
from pydantic import  ValidationError
from typing import  Dict, Any
from dash import callback, Input, Output, State, no_update

from backend.classes.Filter.WaveletsBase import WaveletsBase
from backend.classes.Filter.Notch import Notch
from backend.classes.Filter.ICA import ICA
from backend.classes.Filter.BandPass import BandPass
from backend.classes.Experiment import Experiment
import os
import json
from pydantic import BaseModel

# ---------------------------- FACTORY ----------------------------

class FilterSchemaFactory:
    """
    In this class, we define a factory for generate various Transforms.
    """
    
    available_filters = {
        'ICA': ICA,
        'WaveletsBase': WaveletsBase,
        'BandPass': BandPass,
        'Notch': Notch
    }
    
    @classmethod
    def get_all_filter_schemas(cls) -> Dict[str, Dict[str, Any]]:
        schemas = {}
        for key, model in cls.available_filters.items():
            schema = model.model_json_schema()
          
            schemas[key] = schema
        return schemas

    @classmethod
    def add_filter_to_experiment(cls, directory: str, experiment_id: str, filter_name: str, filter_instance: BaseModel) -> str:
        """
        Añade una instancia por defecto del filtro al archivo de experimento.
        Si el archivo no existe, lo crea con estructura mínima.
        """
        filter_class = cls.available_filters.get(filter_name)
        if filter_class is None:
            raise ValueError(f"Filter '{filter_name}' is not supported.")

        filename = f"experiment_{experiment_id}.json"
        path = os.path.join(directory, filename)

        # Crear el directorio si no existe
        os.makedirs(directory, exist_ok=True)

        # Si el archivo no existe, crear estructura mínima
        if not os.path.exists(path):
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

            if "filter" not in data or not isinstance(data["filter"], list):
                data["filter"] = []

        # Crear instancia del filtro y agregarlo
        
        data["filter"].append({
            "type": filter_name,
            "config": filter_instance.dict()
        })

        with open(path, "a+", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return no_update

#---------------------------------------------Call backs-----------------------------------




#Para hacer los callbacks de cada uno de los componentes generados dinamicamente, usamos la misma función para generar 
# el esquema json, con la cual: Obtenedremos los id y los campos especificos. Luego, también se generan dinamicamente
# los states del callback, los cuales son guardados en un diciconario y valiadados por las clas. 
# 




def filterCallbackRegister(boton_id: str, inputs_map: dict):

    available_filters = {
        'ICA': ICA,
        'WaveletsBase': WaveletsBase,
        'BandPass': BandPass,
        'Notch': Notch
    }
    input_ids = list(inputs_map.keys())

    @callback(
        [
            Output(boton_id, "children"),
            Output("filtered-signal-store-filtros", "data", allow_duplicate=True)
        ],
        Input(boton_id, "n_clicks"),
        [State(input_id, "value") for input_id in input_ids] + [State("signal-store-filtros", "data")],
        prevent_initial_call=True
    )
    def formManager(n_clicks, *values, input_ids=input_ids, validadores=inputs_map):


        if not n_clicks:
            return no_update, no_update

        filtro_nombre = boton_id.replace("btn-aplicar-", "")
        clase_validadora = available_filters.get(filtro_nombre)

        # El último valor es el signal_data store
        signal_data = values[-1]
        values = values[:-1]  # Los demás son los valores de los inputs

        datos = {}
        for input_id, value in zip(input_ids, values):
            _, field = input_id.split("-", 1)
            # Solo agregar el campo si tiene un valor (no None ni vacío)
            if value is not None and value != "":
                datos[field] = value

        print(f"[FilterCallback] 📋 Datos del formulario: {datos}")

        # Obtener el path del evento actual desde el store
        if not signal_data or not isinstance(signal_data, dict):
            print(f"[FilterCallback] ❌ No hay datos de señal cargados")
            return no_update, no_update

        event_file_path = signal_data.get("source")
        if not event_file_path:
            print(f"[FilterCallback] ❌ No se encontró el path del evento en el store")
            return no_update, no_update

        print(f"[FilterCallback] 📂 Aplicando filtro {filtro_nombre} sobre: {event_file_path}")

        try:
            # ✅ Validación con pydantic
            instancia_valida = clase_validadora(**datos)
            print(f"✅ Datos válidos para {filtro_nombre}: {instancia_valida}")

            # 🔧 Aplicación del filtro - devuelve ruta del archivo filtrado
            filtered_file_path = clase_validadora.apply(instancia_valida, file_path=event_file_path)

            # 📊 Cargar datos filtrados y actualizar store
            import numpy as np
            import time

            try:
                arr = np.load(filtered_file_path, allow_pickle=False)
                filtered_data_payload = {
                    "source": filtered_file_path,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "matrix": arr.tolist(),
                    "ts": time.time(),
                    "filter_applied": filtro_nombre
                }
                print(f"[FilterCallback] ✅ Datos filtrados cargados desde: {filtered_file_path}")
            except Exception as e:
                print(f"[FilterCallback] ❌ Error cargando datos filtrados: {e}")
                filtered_data_payload = no_update

            # 📝 Registrar filtro en experimento
            Experiment.add_filter_config(instancia_valida)

            return no_update, filtered_data_payload
        except ValueError as e:
            # Error de valor (ej: wavelet inválido)
            print(f"❌ Error de validación en {filtro_nombre}: {e}")
            return no_update, no_update
        except ValidationError as e:
            print(f"❌ Errores en {filtro_nombre}: {e}")
            errores = e.errors()
            msg = "\n".join(f"{err['loc'][0]}: {err['msg']}" for err in errores)
            return no_update, no_update



# ---------------------------- USO ----------------------------

# if __name__ == "__main__":
#     from pprint import pprint

#     print("🧱 Esquema base Filter:")
#     pprint(FilterSchemaFactory.get_base_schema())

#     print("\n🔧 Esquemas simplificados de filtros hijos:")
#     pprint(FilterSchemaFactory.get_all_filter_schemas())
