from pathlib import Path
import os

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