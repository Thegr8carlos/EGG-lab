
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
        Output(boton_id, "children"),
        Input("selected-file-path", "data"),
        Input(boton_id, "n_clicks"),
        [State(input_id, "value") for input_id in input_ids]
    )
    def formManager(selected_file_path, n_clicks, *values, input_ids=input_ids, validadores=inputs_map):
   

        if not n_clicks:
            return no_update

        filtro_nombre = boton_id.replace("btn-aplicar-", "")
        clase_validadora = available_filters.get(filtro_nombre)

        datos = {}
        for input_id, value in zip(input_ids, values):
            _, field = input_id.split("-", 1)
            datos[field] = value

        try:
            # ✅ Validación con pydantic
            instancia_valida = clase_validadora(**datos)
            print(f"✅ Datos válidos para {filtro_nombre}: {instancia_valida}")

            # 🧪 Simulación de aplicación del filtro

            ####################################------------------------------------------------}
            clase_validadora.apply(instancia_valida, file_path=selected_file_path)
            #clase_validadora.apply(instancia_valida)

            
            Experiment.add_filter_config(instancia_valida)
            

            return no_update
        except ValidationError as e:
            print(f"❌ Errores en {filtro_nombre}: {e}")
            errores = e.errors()
            msg = "\n".join(f"{err['loc'][0]}: {err['msg']}" for err in errores)
            return no_update



# ---------------------------- USO ----------------------------

# if __name__ == "__main__":
#     from pprint import pprint

#     print("🧱 Esquema base Filter:")
#     pprint(FilterSchemaFactory.get_base_schema())

#     print("\n🔧 Esquemas simplificados de filtros hijos:")
#     pprint(FilterSchemaFactory.get_all_filter_schemas())
