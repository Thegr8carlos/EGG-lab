
from pathlib import Path
import  numpy as np


def _load_numeric_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(str(p), allow_pickle=False) as npz:
            key = next(iter(npz.files))
            arr = npz[key]
    else:
        try:
            arr = np.load(str(p), mmap_mode=None, allow_pickle=False)
        except ValueError:
            arr = np.load(str(p), mmap_mode=None, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                obj = arr.item() if arr.ndim == 0 else (arr[0] if arr.ndim == 1 and arr.size == 1 else None)
                if isinstance(obj, dict):
                    for k in ("data", "signal", "x", "array"):
                        if k in obj:
                            arr = np.asarray(obj[k]); break
                    else:
                        raise ValueError("Objeto pickled sin claves conocidas ('data','signal','x','array').")
                elif isinstance(obj, np.ndarray):
                    arr = obj
                elif arr.ndim == 1 and all(isinstance(x, np.ndarray) for x in arr):
                    arr = np.vstack(arr) if arr[0].ndim == 1 else np.array(arr.tolist())
                else:
                    raise ValueError("No se pudo extraer arreglo numérico del objeto pickled.")
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"El archivo no contiene datos numéricos (dtype={arr.dtype}).")
    return arr