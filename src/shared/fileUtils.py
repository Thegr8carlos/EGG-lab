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