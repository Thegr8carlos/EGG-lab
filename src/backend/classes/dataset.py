from pathlib import Path
from shared.fileUtils import is_folder_not_empty, get_files_by_extensions

import mne, sys
import numpy as np 

class Dataset:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.extensions_enabled = [".bdf"]
        if is_folder_not_empty("Data/_aux"):
            print("Dataset cargado ")
        else : 
            print("Se hara la carga de datos")
            self.upload_dataset(path)


    def read_npy(self, path):
        # lee el numpy array y lo retorna 
        data = np.load( "Data/"+ path)
        return data

    def upload_dataset(self,path_to_folder):
        print("Entering upload dataset ")
        print("getting all the files with .bdf extensions, just by now ....")

        if is_folder_not_empty(path_to_folder):
            print("Okay so know we are getting the files ")
            files = get_files_by_extensions(path_to_folder,self.extensions_enabled)
            if len(files) == 0 : 
                return {"status": 400, "message": f"No se han encontrado archivos con extension {self.extensions_enabled}"} 
            
            print("reading the files ")
            for file in files :
                raw_data = self.read_bdf(str(file))
                
            return {"status": 200, "message": f"Se han encontrado {len(files)}  sesiones ", "files" : files}

        else:
            return {"status": 400, "message": "Se ha seleccionado una carpeta Vacia "}

    def get_info(self):
        return f"Info about the dataset hehe "

    def read_bdf(self, path):
        try:
            file_path = Path(path)
            if not file_path.exists():
                return {
                    "status": 400,
                    "mensaje": f"El archivo '{file_path}' no existe.",
                    "archivos_guardados": []
                }

            # Leer archivo BDF
            raw = mne.io.read_raw_bdf(str(file_path), verbose=True, infer_types=True)
            numpy_data = raw.get_data()
            numpy_events = mne.find_events(raw, stim_channel='Status')
            file_name = file_path.name

            # Definir rutas de guardado
            signal_path = Path("Data/_aux") / file_name
            events_path = Path("Data/_aux") / f"{file_name}.event"

            # Guardar datos (asumiendo que carpeta ya existe)
            np.save(signal_path, numpy_data)
            np.save(events_path, numpy_events)

            print(f"Shape of signal {numpy_data.shape} and shape of events {numpy_events.shape}")

            return {
                "status": 200,
                "mensaje": f"Archivo procesado correctamente: {file_name}",
                "archivos_guardados": [str(signal_path), str(events_path)]
            }

        except Exception as e:
            return {
                "status": 400,
                "mensaje": f"Error al procesar archivo '{path}': {str(e)}",
                "archivos_guardados": []
            }
