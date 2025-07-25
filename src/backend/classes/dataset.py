from pathlib import Path
from shared.fileUtils import is_folder_not_empty, get_files_by_extensions

import mne, sys


class Dataset:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.extensions_enabled = [".bdf"]

    def upload_dataset(self,path_to_folder):
        print("Entering upload dataset ")
        print("getting all the files with .bdf extensions, just by now ....")

        if is_folder_not_empty:
            print("Okay so know we are getting the files ")
            files = get_files_by_extensions(path_to_folder,self.extensions_enabled)
            if len(files) == 0 : 
                return {"status": 400, "message": f"No se han encontrado archivos con extension {self.extensions_enabled}"} 
            
            print("reading the files ")
            for file in files :
                raw_data = self.read_bdf(str(file))
                print(f"se ha completado la lectura de {str(file)}, {raw_data.info.ch_names}")
                events = mne.find_events(raw_data, stim_channel='Status')
                print(events)
            return {"status": 200, "message": f"Se han encontrado {len(files)}  sesiones ", "files" : files}

        else:
            return {"status": 400, "message": "Se ha seleccionado una carpeta Vacia "}

    def find_datasets():
        print("finding datasets")

    def get_info(self):
        return f"Info about the dataset hehe "

    def read_bdf(self, path):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
        # MNE espera una cadena, no un Path
        raw = mne.io.read_raw_bdf(str(file_path), verbose=True, infer_types=True)
        return raw
