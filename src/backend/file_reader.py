from pathlib import Path

DATASET_PATH = Path("dataset")

def is_data_loaded():
    print("✅ [backend] Revisando archivos en:", DATASET_PATH)

    if not DATASET_PATH.exists():
        print("⚠️  El directorio no existe.")
        return False

    archivos = [f for f in DATASET_PATH.iterdir() if f.is_file()]
    
    if archivos:
        print("📂 Archivos encontrados:")
        for f in archivos:
            print(" └──", f.name)
        return True
    else:
        print("📭 No hay archivos en el directorio.")
        return False
