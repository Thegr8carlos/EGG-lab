from pathlib import Path
import os
import json

def get_files_by_extensions(path, extensions):
    root = Path(path)
    exts = {ext.lower() for ext in extensions}
    return [
        f for f in root.rglob("*")
        if f.is_file() and f.suffix.lower() in exts
    ]

def is_folder_not_empty(path):
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return False
    return any(root.iterdir())

def get_data_folders():
    data_dir = Path("Data")
    if not data_dir.exists() or not data_dir.is_dir():
        return []

    folders = [
        folder.name for folder in data_dir.iterdir()
        if folder.is_dir() and folder.name != "_aux"
    ]
    return folders


def get_file_extension(fileName):
     
    
    return Path(fileName).suffix



def get_dataset_metadata(file_path: str) -> dict:
    """
    Lee Aux/<dataset>/dataset_metadata.json y regresa un dict.
    Admite:
      - "nieto_inner_speech"
      - "Data/nieto_inner_speech"
      - "Aux/nieto_inner_speech"
      - ".../dataset_metadata.json"
    """
    # Normaliza separadores y quita slashes iniciales/finales
    norm = file_path.replace("\\", "/").strip("/")

    # Si ya me pasaron el JSON directamente, úsalo tal cual
    if norm.endswith("dataset_metadata.json"):
        meta_path = Path(norm)
        if not meta_path.is_absolute():
            meta_path = Path(meta_path)  # relativa al cwd actual
    else:
        # Quita prefijos "Data/" o "Aux/" si vienen
        if norm.startswith("Data/"):
            rel = norm[len("Data/"):]
        elif norm.startswith("Aux/"):
            rel = norm[len("Aux/"):]
        else:
            rel = norm

        # Construye la ruta al JSON dentro de Aux/<dataset>/
        meta_path = Path("Aux") / rel / "dataset_metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"No existe el metadata JSON en: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"dataset_metadata.json inválido: {e}") from e

    return data

def get_Data_filePath(file_path: str, extension: str = ".npy" ) -> str:
    
    # Replace Data/ with Aux/ for corresponding mapping
    

    if file_path.startswith("Data/"): 
        new_file_path = file_path.replace("Data/", "Aux/",1)
        
    else:
        raise ValueError("Input path is wrong not belongining to 'Data/'")
    
    
    newFilePath = os.path.splitext(new_file_path)[0] + extension
    
    return newFilePath
    
def get_Label_filePath(file_path: str, extension: str = ".npy") -> str:
    
    
    if not file_path.startswith("Data/"):
        raise ValueError("Input path is wrong not belonging to 'Data/'")
    
    #Replace Data/ with Aux/ for correct mapping of the labels
    
    
    base_path = file_path.replace("Data/","Aux/", 1)
    
    
    dir_path, fileName = os.path.split(base_path)
    
    #Add labels folder before the filename
    
    label_dir = os.path.join(dir_path, 'Labels')
    
    #Change the extension accordingly 
    
    labelFileName = os.path.splitext(fileName)[0] + extension
    
    labelPath = os.path.join(label_dir, labelFileName)
    
    return labelPath