from pathlib import Path

import mne, sys

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

def is_data_loaded():
    """
    Retorna True si:
    - hay datos “raw” en dataset/raw, o
    - hay archivos .bdf (u otras extensiones que quieras) en dataset/
    """
    RAW   = Path("dataset/raw")
    BASE  = Path("dataset")
    # 1) Si hay algo en raw → ya cargado
    if RAW.exists() and is_folder_not_empty(RAW):
        return True

    # 2) Si no hay raw pero dataset contiene archivos .bdf → también cargado
    exts = [".bdf", ".npy", ".npz"]
    files = get_files_by_extensions(BASE, exts)
    return len(files) > 0


def read_bdf(path):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
    # MNE espera una cadena, no un Path
    raw = mne.io.read_raw_bdf(str(file_path), preload=preload, verbose=True)
    return raw