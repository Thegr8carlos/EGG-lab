from pathlib import Path

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