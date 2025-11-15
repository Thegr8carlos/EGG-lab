from __future__ import annotations

from pathlib import Path
from shared.fileUtils import (
    get_file_extension,
    is_folder_not_empty,
    get_files_by_extensions,
    get_Data_filePath,
    get_Label_filePath,
)
from dash import no_update
from sklearn.preprocessing import LabelEncoder
import mne, sys
import numpy as np
import os, re
import pandas as pd
import json
from collections import Counter
from typing import List, Sequence


LABELS = {31: "arriba", 32: "abajo", 33: "derecha", 34: "izquierda"}
CUE_IDS = set(LABELS.keys())


def _inner_speech_cues(events):
    """
    events: array Nx3 de mne.find_events (col3 = event_id)
    Devuelve lista de (sample, event_id) SOLO para trials dentro de runs de inner speech.
    """
    # 1) quita el evento espurio 65536
    events = events[events[:, 2] != 65536]

    in_inner_run = False
    cues = []
    inner_speech_runs_count = 0

    for sample, _, eid in events:
        if eid == 15:  # start of run
            in_inner_run = False
        elif eid == 22:  # start of inner speech run
            in_inner_run = True
            inner_speech_runs_count += 1
        elif eid == 16:  # end of run
            in_inner_run = False
        elif in_inner_run and eid in CUE_IDS:  # cue de clase dentro del run inner
            cues.append((int(sample), int(eid)))

    print(f"[NIETO] Runs de inner speech detectados: {inner_speech_runs_count}")
    print(f"[NIETO] Eventos extraídos de runs inner speech: {len(cues)}")

    # Mostrar conteo por clase
    class_counts = {label: 0 for label in LABELS.values()}
    for _, eid in cues:
        if eid in LABELS:
            class_counts[LABELS[eid]] += 1
    print(f"[NIETO] Distribución por clase: {class_counts}")

    return cues


def _detect_dataset_type(events):
    """
    Detecta el tipo de dataset basándose en los event IDs encontrados.

    Returns:
        "inner_speech" si detecta markers de Inner Speech (15, 22, 31-34)
        "generic" para cualquier otro dataset
    """
    event_ids = set(events[:, 2])

    # Inner Speech tiene markers específicos: 15 (start run), 22 (inner speech run), 31-34 (clases)
    inner_speech_markers = {15, 22, 31, 32, 33, 34}
    if inner_speech_markers.issubset(event_ids):
        print("=" * 80)
        print("🧠 DATASET DE NIETO DETECTADO - Usando proceso específico de Inner Speech")
        print("=" * 80)
        print(f"[NIETO] Markers encontrados: {sorted(event_ids)}")
        print(f"[NIETO] Usando mapeo de IDs: {LABELS}")
        print(f"[NIETO] Filtrando eventos dentro de runs de inner speech (marker 22)")
        print("=" * 80)
        return "inner_speech"

    print(f"[Dataset] Tipo detectado: Genérico (event IDs: {sorted(event_ids)})")
    return "generic"


def _generic_event_extraction(events, event_id_mapping=None):
    """
    Extrae eventos de datasets genéricos (todos los event IDs únicos).

    Args:
        events: array Nx3 de mne.find_events
        event_id_mapping: dict opcional {event_id: "nombre_clase"}

    Returns:
        tuple: (cues, labels_dict)
            - cues: lista de (sample, event_id)
            - labels_dict: dict {event_id: "nombre_clase"}
    """
    # Limpiar eventos espurios comunes
    events = events[events[:, 2] != 65536]
    events = events[events[:, 2] != 0]

    # Obtener todos los event IDs únicos
    unique_ids = sorted(set(events[:, 2]))
    print(f"[Dataset] Event IDs únicos encontrados: {unique_ids}")

    # Si no hay mapping, crear nombres genéricos
    if not event_id_mapping:
        event_id_mapping = {int(eid): f"Evento_{eid}" for eid in unique_ids}
        print(f"[Dataset] Mapping generado: {event_id_mapping}")

    # Extraer todos los eventos (sin filtrado por runs como Inner Speech)
    cues = [(int(sample), int(eid)) for sample, _, eid in events if eid in unique_ids]
    print(f"[Dataset] Total de eventos extraídos: {len(cues)}")

    return cues, event_id_mapping


def _aux_root_for(path_to_folder: str) -> str:
    """
    Mapea Data/... -> Aux/... (solo primer ocurrencia), crea la carpeta y la retorna.
    Si no empieza con 'Data/', usa la ruta original.
    """
    if path_to_folder.startswith("Data/"):
        aux_root = path_to_folder.replace("Data/", "Aux/", 1)
    else:
        aux_root = path_to_folder
    os.makedirs(aux_root, exist_ok=True)
    return aux_root


def _filter_largest_per_session(files):
    """
    Agrupa archivos por directorio (sesión) y retorna solo el más grande de cada sesión.

    Args:
        files: Lista de Path objects

    Returns:
        Lista filtrada con solo el archivo más grande por sesión
    """
    from collections import defaultdict

    # Agrupar archivos por directorio padre
    sessions = defaultdict(list)
    for file_path in files:
        session_dir = file_path.parent
        sessions[session_dir].append(file_path)

    # Por cada sesión, elegir el archivo más grande
    filtered_files = []
    for session_dir, session_files in sessions.items():
        if len(session_files) == 1:
            # Solo un archivo, usarlo directamente
            filtered_files.append(session_files[0])
        else:
            # Múltiples archivos: elegir el más grande
            largest = max(session_files, key=lambda f: f.stat().st_size)
            skipped = [f for f in session_files if f != largest]

            print(f"[SESSION] {session_dir.name}: {len(session_files)} archivos encontrados")
            print(f"[SESSION] ✅ Usando: {largest.name} ({largest.stat().st_size / 1024:.1f} KB)")
            for skipped_file in skipped:
                print(f"[SESSION] ⏭️  Ignorando: {skipped_file.name} ({skipped_file.stat().st_size / 1024:.1f} KB)")

            filtered_files.append(largest)

    return filtered_files


# ===== Utilidad para json.dump: castea objetos no serializables a str/isoformat =====
def _json_fallback(o):
    try:
        if hasattr(o, "isoformat"):
            return o.isoformat()
    except Exception:
        pass
    try:
        return float(o)
    except Exception:
        pass
    try:
        return int(o)
    except Exception:
        pass
    try:
        return list(o)
    except Exception:
        pass
    return str(o)


# ===== Funciones auxiliares para calcular metadata adicional =====

def _calculate_channel_stats(raw_data):
    """
    Calcula estadísticas descriptivas por canal.

    Args:
        raw_data: objeto mne.io.Raw

    Returns:
        dict: {channel_name: {mean, std, min, max, variance, rms}}
    """
    data, _ = raw_data.get_data(return_times=True)
    channel_names = raw_data.info['ch_names']

    stats = {}
    for i, ch_name in enumerate(channel_names):
        ch_data = data[i, :]
        stats[ch_name] = {
            "mean": float(np.mean(ch_data)),
            "std": float(np.std(ch_data)),
            "min": float(np.min(ch_data)),
            "max": float(np.max(ch_data)),
            "variance": float(np.var(ch_data)),
            "rms": float(np.sqrt(np.mean(ch_data**2)))
        }

    return stats


def _infer_montage(channel_names):
    """
    Intenta inferir el montage estándar basándose en los nombres de canales.

    Args:
        channel_names: lista de nombres de canales

    Returns:
        dict: {
            "type": tipo de montage ("standard_1020", "biosemi64", etc.) o "unknown",
            "positions": {channel: {"x": float, "y": float, "z": float}},
            "has_positions": bool
        }
    """
    # Lista de montages estándar a probar
    montage_names = [
        "standard_1020",
        "standard_1005",
        "biosemi64",
        "biosemi128",
        "biosemi256",
        "easycap-M1",
        "easycap-M10"
    ]

    # Normalizar nombres de canales (uppercase, sin espacios)
    ch_names_normalized = [ch.strip().upper() for ch in channel_names]

    for montage_name in montage_names:
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            montage_ch_names = [ch.upper() for ch in montage.ch_names]

            # Verificar cuántos canales coinciden
            matches = sum(1 for ch in ch_names_normalized if ch in montage_ch_names)
            match_ratio = matches / len(ch_names_normalized) if ch_names_normalized else 0

            # Si al menos 50% de canales coinciden, usar este montage
            if match_ratio >= 0.5:
                positions = {}
                for ch_name in channel_names:
                    ch_normalized = ch_name.strip().upper()
                    if ch_normalized in montage_ch_names:
                        idx = montage_ch_names.index(ch_normalized)
                        pos = montage.get_positions()['ch_pos'][montage.ch_names[idx]]
                        positions[ch_name] = {
                            "x": float(pos[0]),
                            "y": float(pos[1]),
                            "z": float(pos[2])
                        }

                return {
                    "type": montage_name,
                    "positions": positions,
                    "has_positions": len(positions) > 0,
                    "matched_channels": len(positions),
                    "total_channels": len(channel_names)
                }
        except Exception as e:
            continue

    # Si no se encontró montage, retornar unknown
    return {
        "type": "unknown",
        "positions": {},
        "has_positions": False,
        "matched_channels": 0,
        "total_channels": len(channel_names)
    }


def _calculate_frequency_bands(raw_data, sfreq):
    """
    Calcula la potencia promedio en bandas de frecuencia clásicas.

    Args:
        raw_data: objeto mne.io.Raw o numpy array (channels × time)
        sfreq: frecuencia de muestreo

    Returns:
        dict: {band_name: {"range": [low, high], "mean_power": float}}
    """
    # Definir bandas clásicas
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, min(100, sfreq / 2 - 1))  # No exceder Nyquist
    }

    # Obtener datos
    if isinstance(raw_data, np.ndarray):
        data = raw_data
    else:
        data, _ = raw_data.get_data(return_times=True)

    # Calcular PSD usando Welch
    from scipy import signal as scipy_signal

    result = {}
    for band_name, (low, high) in bands.items():
        try:
            # Calcular PSD por canal y promediar
            band_powers = []
            for ch_data in data:
                freqs, psd = scipy_signal.welch(ch_data, fs=sfreq, nperseg=min(256, len(ch_data)))
                # Integrar potencia en la banda
                freq_mask = (freqs >= low) & (freqs <= high)
                if np.any(freq_mask):
                    band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                    band_powers.append(band_power)

            mean_power = float(np.mean(band_powers)) if band_powers else 0.0

            result[band_name] = {
                "range": [float(low), float(high)],
                "mean_power": mean_power
            }
        except Exception as e:
            result[band_name] = {
                "range": [float(low), float(high)],
                "mean_power": 0.0,
                "error": str(e)
            }

    return result


def _detect_bad_channels(raw_data, threshold_std=3.0):
    """
    Detecta canales potencialmente ruidosos basándose en varianza extrema.

    Args:
        raw_data: objeto mne.io.Raw
        threshold_std: número de desviaciones estándar para considerar outlier

    Returns:
        list: nombres de canales detectados como malos
    """
    data, _ = raw_data.get_data(return_times=True)
    channel_names = raw_data.info['ch_names']

    # Calcular varianza por canal
    variances = np.var(data, axis=1)

    # Detectar outliers (canales con varianza muy alta o muy baja)
    mean_var = np.mean(variances)
    std_var = np.std(variances)

    bad_channels = []
    for i, ch_name in enumerate(channel_names):
        if abs(variances[i] - mean_var) > threshold_std * std_var:
            bad_channels.append(ch_name)

    return bad_channels


def _calculate_session_stats(raw, events, labels_dict, file_path, sfreq):
    """
    Calcula estadísticas para una sesión individual.

    Args:
        raw: objeto mne.io.Raw
        events: array de eventos (N × 3)
        labels_dict: diccionario {event_id: label_name}
        file_path: path al archivo
        sfreq: frecuencia de muestreo

    Returns:
        dict con estadísticas de la sesión
    """
    from pathlib import Path

    path_parts = Path(file_path).parts

    # Extraer subject y session del path
    subject = "unknown"
    session = "unknown"
    for i, part in enumerate(path_parts):
        if part.startswith("sub-"):
            subject = part
        if part.startswith("ses-"):
            session = part

    # Contar eventos por clase
    events_per_class = {label: 0 for label in set(labels_dict.values())}
    for _, _, eid in events:
        if eid in labels_dict:
            events_per_class[labels_dict[eid]] += 1

    # Duración en segundos
    n_times = raw.n_times
    duration_sec = n_times / sfreq

    return {
        "subject": subject,
        "session": session,
        "file_path": str(file_path),
        "duration_sec": float(duration_sec),
        "n_events": int(len(events)),
        "events_per_class": events_per_class,
        "sampling_rate": float(sfreq),
        "n_channels": int(len(raw.info['ch_names'])),
        "n_samples": int(n_times)
    }


class Dataset:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.extensions_enabled = [".bdf", ".edf", ".vhdr"]  # Agregado soporte BrainVision

    def get_events_by_class(path_to_folder, class_name=None):
        """
        Busca archivos .npy en la carpeta Events, opcionalmente filtrando por clase.

        Args:
            path_to_folder: Ruta al dataset (ej: "Data/nieto_inner_speech" o archivo .bdf)
            class_name: Nombre de la clase a filtrar (ej: "abajo", "arriba"). Si es None, retorna todos.

        Returns:
            dict con:
                - status: código de estado
                - message: mensaje descriptivo
                - events_dir: directorio de eventos
                - event_files: lista de archivos .npy filtrados por clase
                - first_event_file: primer archivo encontrado
        """
        print(f"\n[get_events_by_class] Buscando eventos para clase: {class_name}")
        print(f"[get_events_by_class] Path: {path_to_folder}")

        if not path_to_folder or not isinstance(path_to_folder, (str, os.PathLike)):
            msg = "Ruta inválida"
            print(f"[get_events_by_class] {msg}")
            return {"status": 400, "message": msg, "event_files": []}

        p_in = Path(path_to_folder)

        # Resolver ruta al directorio Events en Aux
        if p_in.exists() and p_in.is_dir():
            aux_root = Path(_aux_root_for(str(p_in if str(p_in).startswith("Data/") else str(p_in))))
            events_dirs = list(aux_root.rglob("Events"))
            if not events_dirs:
                msg = f"No se encontró carpeta Events en {aux_root}"
                print(f"[get_events_by_class] {msg}")
                return {"status": 404, "message": msg, "event_files": []}
            events_dir = events_dirs[0]  # Tomar el primer directorio Events encontrado
        else:
            # Si es un archivo, mapear a Aux y buscar Events
            if not p_in.is_absolute() and not str(p_in).startswith("Data/"):
                data_full = Path("Data") / p_in
            else:
                data_full = p_in

            try:
                mapped = get_Data_filePath(str(data_full))
                npy_path = Path(mapped)
                events_dir = npy_path.parent / "Events"
            except Exception as e:
                msg = f"Error mapeando a Aux: {e}"
                print(f"[get_events_by_class] {msg}")
                return {"status": 500, "message": msg, "event_files": []}

        if not events_dir.exists() or not events_dir.is_dir():
            msg = f"Directorio Events no existe: {events_dir}"
            print(f"[get_events_by_class] {msg}")
            return {"status": 404, "message": msg, "event_files": []}

        # Buscar archivos .npy
        all_events = sorted([p for p in events_dir.glob("*.npy")])

        # Filtrar por clase si se especifica
        if class_name:
            # Los archivos tienen formato: <clase>[inicio]{fin}.npy
            # Ejemplo: abajo[348.903]{351.453}.npy
            filtered_events = [p for p in all_events if p.name.startswith(f"{class_name}[")]
            print(f"[get_events_by_class] Encontrados {len(filtered_events)} eventos de clase '{class_name}'")
        else:
            filtered_events = all_events
            print(f"[get_events_by_class] Encontrados {len(filtered_events)} eventos totales")

        first_event = filtered_events[0] if filtered_events else None

        return {
            "status": 200,
            "message": f"OK - {len(filtered_events)} eventos encontrados",
            "events_dir": str(events_dir),
            "event_files": [str(p) for p in filtered_events],
            "first_event_file": str(first_event) if first_event else None,
            "n_events": len(filtered_events)
        }

    def load_events(path_to_folder):
        """
        Inspecciona la carpeta Events junto al .npy en Aux/.
        Además, si encuentra al menos un .npy en Events, carga el primero y lo imprime.
        """
        print("\n[load_events] Entering load events")
        print(f"[load_events] Input path: {path_to_folder!r}")

        if not path_to_folder or not isinstance(path_to_folder, (str, os.PathLike)):
            msg = "Se ha proporcionado una ruta inválida (no es str/os.PathLike)."
            print("[load_events]", msg)
            return {"status": 400, "message": msg}

        p_in = Path(path_to_folder)

        # --- Resolver a ruta del .npy bajo Aux (igual que tu flujo actual) ---
        if p_in.exists() and p_in.is_dir():
            print(f"[load_events] Es carpeta existente: {p_in}")
            aux_root = Path(_aux_root_for(str(p_in if str(p_in).startswith("Data/") else str(p_in))))
            print(f"[load_events] Aux root espejo (creado si no existe): {aux_root}")
            npy_candidates = sorted(aux_root.rglob("*.npy"))
            if not npy_candidates:
                msg = f"No se encontraron .npy en {aux_root} (Aux espejo de la carpeta)."
                print("[load_events]", msg)
                return {"status": 404, "message": msg}
            npy_path = npy_candidates[0]
            print(f"[load_events] Primer .npy encontrado en Aux: {npy_path}")
        else:
            if not p_in.is_absolute() and not str(p_in).startswith("Data/"):
                data_full = Path("Data") / p_in
            else:
                data_full = p_in

            if data_full.suffix.lower() == ".npy":
                if "Aux" in str(data_full):
                    npy_path = data_full
                    print(f"[load_events] .npy ya bajo Aux: {npy_path}")
                else:
                    try:
                        mapped = get_Data_filePath(str(data_full))
                        npy_path = Path(mapped)
                        print(f"[load_events] Mapeado .npy Data->Aux: {data_full} -> {npy_path}")
                    except Exception as e:
                        msg = f"Error mapeando .npy a Aux: {e}"
                        print("[load_events]", msg)
                        return {"status": 500, "message": msg}
            else:
                if not str(data_full).startswith("Data/"):
                    data_full = Path("Data") / data_full
                try:
                    mapped = get_Data_filePath(str(data_full))
                    npy_path = Path(mapped)
                    print(f"[load_events] Mapeado Data file -> Aux .npy: {data_full} -> {npy_path}")
                except Exception as e:
                    msg = f"Error mapeando archivo a Aux .npy: {e}"
                    print("[load_events]", msg)
                    return {"status": 500, "message": msg}

        # --- Ubicar carpeta Events junto al .npy ---
        events_dir = npy_path.parent / "Events"
        print(f"[load_events] Buscando Events en: {events_dir} | Existe? {events_dir.exists()}")

        events_files = []
        if events_dir.exists() and events_dir.is_dir():
            events_files = sorted([p for p in events_dir.glob("*.npy")])
            print(f"[load_events] Events encontrados: {len(events_files)}")
            for p in events_files[:20]:
                print("   -", p.name)
            if len(events_files) > 20:
                print(f"[load_events] ... y {len(events_files)-20} más")
        else:
            print("[load_events] Carpeta Events no encontrada.")

        # --- NUEVO: leer e imprimir el primer .npy de Events (si existe) ---
        first_preview = None
        if events_files:
            first_npy = events_files[0]
            try:
                print(f"[load_events] Cargando primer evento: {first_npy}")
                arr = np.load(first_npy, allow_pickle=False)  # eventos son float32 típicamente
                print(f"[load_events] npy shape={arr.shape}, dtype={arr.dtype}, ndim={arr.ndim}")

                # Pequeño preview para no saturar: primeras 3 filas x 10 columnas (si aplica)
                if arr.ndim == 2:
                    r = min(arr.shape[0], 3)
                    c = min(arr.shape[1], 10)
                    first_preview = arr[:r, :c]
                elif arr.ndim == 1:
                    c = min(arr.shape[0], 10)
                    first_preview = arr[:c]
                else:
                    # Para tensores más altos, solo muestra el primer "slice" comprimido
                    first_preview = np.array(arr).reshape(-1)[:10]

                print("[load_events] Preview del primer npy:")
                print(first_preview)
            except Exception as e:
                print(f"[load_events] ERROR leyendo {first_npy}: {e}")

        # No cambiamos el contrato: retornamos solo status/logs básicos
        return {
            "status": 200,
            "message": "OK (inspección de Events y lectura del primer .npy impresa en consola)",
            "events_dir": str(events_dir),
            "n_events": len(events_files),
            "first_event_file": str(events_files[0]) if events_files else None,
            "first_event_shape": tuple(first_preview.shape) if isinstance(first_preview, np.ndarray) else None
        }

        
    def upload_dataset(self, path_to_folder):
        print("Entering upload dataset ")
        print("getting all the files with .bdf, .edf extensions, just by now ....")

        if not is_folder_not_empty(path_to_folder):
            return {"status": 400, "message": "Se ha seleccionado una carpeta Vacia "}

        print("Okay so know we are getting the files ")
        files = get_files_by_extensions(path_to_folder, self.extensions_enabled)

        # ======= FILTRAR: Solo el archivo más grande por sesión =======
        print(f"\n[FILTER] Total archivos encontrados: {len(files)}")
        files = _filter_largest_per_session(files)
        print(f"[FILTER] Archivos después del filtrado: {len(files)}\n")

        for i in files:
            print(i)
        if len(files) == 0:
            return {
                "status": 400,
                "message": f"No se han encontrado archivos con extension {self.extensions_enabled}",
            }

        print("reading the files ")
        all_entries = []

        # ======= Acumuladores para METADATA GLOBAL (solo 1 JSON al final) =======
        # Iniciar vacío - se acumularán clases detectadas dinámicamente
        total_class_counts = Counter()
        unique_sfreqs = set()
        union_channels = set()
        ch_types_total = Counter()
        total_duration_sec = 0.0

        # ======= Nuevos acumuladores para metadata extendida =======
        all_sessions = []  # Lista de dicts con stats por sesión
        channel_stats_accumulated = {}  # Estadísticas por canal (se promediarán al final)
        montage_info = None  # Se inferirá del primer archivo
        frequency_bands_accumulated = {}  # Se promediarán al final
        all_bad_channels = set()  # Set de bad channels detectados
        total_files_processed = 0
        first_raw_for_analysis = None  # Guardar primer raw para análisis completo

        # Metadata "general" tomada de UN BDF (si existe), incluso si se hace skip pesado
        sampled_meta_done = False
        sampled_sfreq = None
        sampled_ch_names = []
        sampled_ch_types_count = {}

        # Carpeta raíz espejo Aux/
        aux_root = _aux_root_for(path_to_folder)

        # ---- Paso previo: si aún no tenemos metadata, intenta leer RÁPIDO el primer .bdf ----
        # (preload=False por defecto en MNE, leerá encabezados y estructura)
        if not sampled_meta_done:
            first_bdf = next((str(f) for f in files if get_file_extension(f) == ".bdf"), None)
            if first_bdf:
                try:
                    raw_hdr = self.read_bdf(first_bdf)
                    sampled_sfreq = float(raw_hdr.info["sfreq"])
                    sampled_ch_names = list(raw_hdr.info["ch_names"])
                    sampled_ch_types_count = dict(Counter(raw_hdr.get_channel_types()))
                    sampled_meta_done = True
                    # Inicializa acumuladores con esta muestra
                    unique_sfreqs.add(sampled_sfreq)
                    union_channels.update(sampled_ch_names)
                    ch_types_total.update(sampled_ch_types_count)
                    print(f"[META] Sampled from BDF: sfreq={sampled_sfreq}, n_channels={len(sampled_ch_names)}")
                except Exception as e:
                    print(f"[META] No se pudo muestrear encabezado BDF: {e}")

        for file in files:
            ext = get_file_extension(file)

            # ===================== BDF =====================
            if ext == ".bdf":
                # --- SKIP temprano si ya existen derivados para este archivo ---
                auxFilePath  = get_Data_filePath(str(file))    # .../Aux/.../<file>.npy (data)
                auxLabelPath = get_Label_filePath(str(file))   # .../Aux/.../Labels/<file>.npy (labels)
                labels_dir   = os.path.dirname(auxLabelPath)   # .../Aux/.../Labels
                events_dir   = os.path.join(os.path.dirname(labels_dir), "Events")  # .../Aux/.../Events

                data_exists  = os.path.exists(auxFilePath)
                labels_exist = os.path.exists(auxLabelPath)
                events_ready = os.path.isdir(events_dir) and any(
                    fn.endswith(".npy") for fn in os.listdir(events_dir)
                )

                if data_exists and labels_exist and events_ready:
                    print(f"[SKIP-BDF] Derivados ya existen para {file}. Saltando lectura del BDF.")
                    # AUN ASÍ, si no hemos muestreado metadata (caso raro sin primer BDF), muestrea aquí:
                    if not sampled_meta_done:
                        try:
                            raw_hdr = self.read_bdf(str(file))
                            sampled_sfreq = float(raw_hdr.info["sfreq"])
                            sampled_ch_names = list(raw_hdr.info["ch_names"])
                            sampled_ch_types_count = dict(Counter(raw_hdr.get_channel_types()))
                            sampled_meta_done = True
                            unique_sfreqs.add(sampled_sfreq)
                            union_channels.update(sampled_ch_names)
                            ch_types_total.update(sampled_ch_types_count)
                            print(f"[META] Sampled (skip branch): sfreq={sampled_sfreq}, n_channels={len(sampled_ch_names)}")
                        except Exception as e:
                            print(f"[META] Error sampling on skip: {e}")
                    continue

                # --- Procesamiento normal (solo si falta algo) ---
                raw_data = self.read_bdf(str(file))
                print(f"se ha completado la lectura de {str(file)}, {raw_data.info.ch_names}")

                # Extraer eventos - buscar canal de estímulo automáticamente
                stim_channels = mne.pick_types(raw_data.info, stim=True, exclude=[])
                print(f"[EVENTS] Canales de estímulo detectados: {[raw_data.ch_names[i] for i in stim_channels]}")

                events = None

                if len(stim_channels) == 0:
                    # No hay canal de estímulo - intentar buscar 'Status' explícitamente
                    if 'Status' in raw_data.ch_names:
                        stim_channel = 'Status'
                        print(f"[EVENTS] Usando canal de estímulo: {stim_channel}")
                        events = mne.find_events(
                            raw_data, stim_channel=stim_channel, shortest_event=1, verbose=True
                        )
                    else:
                        # Intentar leer desde anotaciones
                        print(f"[EVENTS] No hay canal de estímulo. Intentando leer desde anotaciones...")
                        annotations = raw_data.annotations
                        if annotations is not None and len(annotations) > 0:
                            print(f"[EVENTS] Encontradas {len(annotations)} anotaciones")
                            events, event_id = mne.events_from_annotations(raw_data)
                            print(f"[EVENTS] Event IDs desde anotaciones: {event_id}")
                        else:
                            # Último intento: buscar archivo evt.bdf asociado
                            evt_file = file.parent / "evt.bdf"
                            if evt_file.exists():
                                print(f"[EVENTS] Encontrado archivo de eventos asociado: {evt_file.name}")
                                print(f"[EVENTS] Leyendo eventos desde {evt_file.name}...")
                                try:
                                    evt_raw = mne.io.read_raw_bdf(str(evt_file), verbose=True, infer_types=True)
                                    print(f"[EVENTS] evt.bdf canales: {evt_raw.ch_names}")
                                    print(f"[EVENTS] evt.bdf duración: {evt_raw.times[-1]:.2f}s")
                                    print(f"[EVENTS] data.bdf duración: {raw_data.times[-1]:.2f}s")

                                    evt_stim_channels = mne.pick_types(evt_raw.info, stim=True, exclude=[])
                                    print(f"[EVENTS] evt.bdf canales stim: {[evt_raw.ch_names[i] for i in evt_stim_channels]}")

                                    if len(evt_stim_channels) > 0:
                                        evt_stim_channel = evt_raw.ch_names[evt_stim_channels[0]]
                                        print(f"[EVENTS] Usando canal de estímulo de evt.bdf: {evt_stim_channel}")
                                        events = mne.find_events(evt_raw, stim_channel=evt_stim_channel, shortest_event=1, verbose=True)
                                    elif 'Status' in evt_raw.ch_names:
                                        print(f"[EVENTS] Usando 'Status' de evt.bdf")
                                        events = mne.find_events(evt_raw, stim_channel='Status', shortest_event=1, verbose=True)
                                    else:
                                        # Intentar desde anotaciones del evt.bdf
                                        print(f"[EVENTS] Intentando leer anotaciones de evt.bdf...")
                                        if evt_raw.annotations is not None and len(evt_raw.annotations) > 0:
                                            print(f"[EVENTS] evt.bdf tiene {len(evt_raw.annotations)} anotaciones")
                                            events, event_id = mne.events_from_annotations(evt_raw)
                                            print(f"[EVENTS] Event IDs desde anotaciones de evt.bdf: {event_id}")
                                except Exception as e:
                                    print(f"[EVENTS] Error leyendo evt.bdf: {e}")
                                    import traceback
                                    traceback.print_exc()

                            if events is None:
                                print(f"[EVENTS] ⚠️ No se pudieron extraer eventos de {file.name}")
                                print(f"[EVENTS] Canales disponibles: {raw_data.ch_names}")
                                print(f"[EVENTS] ⚠️ Saltando archivo")
                                continue
                else:
                    stim_channel = raw_data.ch_names[stim_channels[0]]
                    print(f"[EVENTS] Usando canal de estímulo: {stim_channel}")
                    events = mne.find_events(
                        raw_data, stim_channel=stim_channel, shortest_event=1, verbose=True
                    )

                print(f"[BDF] Eventos encontrados: {len(events)}")

                # Detectar tipo de dataset (Inner Speech vs genérico)
                dataset_type = _detect_dataset_type(events)

                if dataset_type == "inner_speech":
                    # Usar extracción específica de Inner Speech
                    inner_cues = _inner_speech_cues(events)
                    labels_dict = LABELS  # {31: "arriba", 32: "abajo", ...}
                    print(f"[BDF] Usando extracción Inner Speech: {len(inner_cues)} cues")
                else:
                    # Extracción genérica: TODOS los event IDs
                    inner_cues, labels_dict = _generic_event_extraction(events)
                    print(f"[BDF] Usando extracción genérica: {len(inner_cues)} eventos, clases: {list(labels_dict.values())}")

                # Data raw
                data, _ = raw_data.get_data(return_times=True)  # (n_channels, n_times)
                sfreq = float(raw_data.info["sfreq"])

                # ===== Etiquetas por muestra =====
                label_array = np.zeros((1, data.shape[1]), dtype=object)

                label_duration_sec = 3.2  # mantenemos tu valor actual
                label_duration_samples = int(label_duration_sec * sfreq)

                # Etiquetar eventos
                for sample_idx, eid in inner_cues:
                    start = sample_idx
                    end = min(sample_idx + label_duration_samples, data.shape[1])
                    label_array[0, start:end] = labels_dict[eid]

                # ===== RE-ETIQUETAR BACKGROUND (resto de la señal) =====
                # Siempre usar "rest" como etiqueta de background
                # Las transformadas se encargarán del re-etiquetado específico (P300 binario, etc.)
                background_label = "rest"

                # Llenar zonas sin etiquetar (donde es 0 o vacío)
                background_mask = (label_array[0] == 0) | (label_array[0] == '') | (label_array[0] == '0')
                label_array[0, background_mask] = background_label
                print(f"[LABELS] Background re-etiquetado como 'rest': {background_mask.sum()} muestras")

                label_array = label_array.astype(str)

                # ===== Guardado estándar (Data/Labels) con SKIP si existe =====
                os.makedirs(os.path.dirname(auxFilePath), exist_ok=True)
                os.makedirs(os.path.dirname(auxLabelPath), exist_ok=True)

                if os.path.exists(auxFilePath):
                    print(f"[SKIP] Data ya existe: {auxFilePath}")
                else:
                    np.save(auxFilePath, data)

                if os.path.exists(auxLabelPath):
                    print(f"[SKIP] Labels ya existe: {auxLabelPath}")
                else:
                    np.save(auxLabelPath, label_array)

                # Conteo por clase (para metadata agregada)
                counts = {labels_dict[k]: 0 for k in labels_dict}
                for _, eid in inner_cues:
                    counts[labels_dict[eid]] += 1
                print(f"[BDF] Eventos por clase en {Path(file).name}: {counts}")
                total_class_counts.update(counts)

                # ===== Un archivo .npy por evento en Events/ con formato <clase>[ini]{fin}.npy =====
                os.makedirs(events_dir, exist_ok=True)
                prefer_action_tags = True  # usa 44-45 si existen; si no, cae a 3.2s desde el cue

                for (cue_sample, eid) in inner_cues:
                    class_name = labels_dict[eid]

                    # Delimitación del evento
                    start_sample = cue_sample
                    end_sample = min(cue_sample + label_duration_samples, data.shape[1])

                    if prefer_action_tags:
                        # primer 44 >= cue, y primer 45 >= ese 44
                        next44 = events[(events[:, 0] >= cue_sample) & (events[:, 2] == 44)]
                        if next44.size > 0:
                            start_sample = int(next44[0, 0])
                            next45 = events[(events[:, 0] >= start_sample) & (events[:, 2] == 45)]
                            if next45.size > 0:
                                end_sample = int(next45[0, 0])

                    # Validación
                    if end_sample <= start_sample or start_sample < 0 or end_sample > data.shape[1]:
                        print(
                            f"[Events] Límites inválidos para clase {class_name}: "
                            f"{start_sample}-{end_sample}. Se omite."
                        )
                        continue

                    # Extrae matriz del evento y tiempos
                    X_event = data[:, start_sample:end_sample].astype(np.float32)
                    start_time = start_sample / sfreq
                    end_time = end_sample / sfreq

                    # Nombre: <clase>[ini]{fin}.npy (sin nombre original)
                    safe_class = re.sub(r'[\\/:*?"<>|]', "_", class_name)
                    out_name = f"{safe_class}[{start_time:.3f}]{{{end_time:.3f}}}.npy"
                    out_path = os.path.join(events_dir, out_name)

                    # SKIP si el evento ya existe
                    if os.path.exists(out_path):
                        print(f"[SKIP] Event ya existe: {out_path}")
                        continue

                    np.save(out_path, X_event)
                    print(
                        f"[Events] Guardado {out_path} | clase={class_name} | "
                        f"samples={start_sample}-{end_sample} | shape={X_event.shape}"
                    )

                # ===== EXTRAER VENTANAS DE BACKGROUND =====
                print(f"[BDF] Extrayendo ventanas de background (etiquetadas como '{background_label}')...")

                # Encontrar índices donde está el background
                background_indices = np.where(label_array[0] == background_label)[0]

                if len(background_indices) > 0:
                    # Segmentar en ventanas del mismo tamaño que eventos
                    window_size = label_duration_samples
                    n_windows = len(background_indices) // window_size

                    background_count = 0
                    for i in range(n_windows):
                        start_idx = background_indices[i * window_size]
                        end_idx = start_idx + window_size

                        # Verificar que la ventana completa sea background (contigua)
                        if end_idx <= data.shape[1]:
                            window_labels = label_array[0, start_idx:end_idx]
                            if np.all(window_labels == background_label):
                                # Extraer ventana
                                X_background = data[:, start_idx:end_idx].astype(np.float32)

                                # Calcular tiempos
                                start_time = start_idx / sfreq
                                end_time = end_idx / sfreq

                                # Nombre del archivo
                                safe_label = re.sub(r'[\\/:*?"<>|]', "_", background_label)
                                out_name = f"{safe_label}[{start_time:.3f}]{{{end_time:.3f}}}.npy"
                                out_path = os.path.join(events_dir, out_name)

                                # SKIP si ya existe
                                if os.path.exists(out_path):
                                    continue

                                np.save(out_path, X_background)
                                background_count += 1

                    print(f"[BDF] Extraídas {background_count} ventanas de background | shape={X_background.shape if background_count > 0 else 'N/A'}")

                    # Actualizar conteos (acumular correctamente)
                    if background_label not in counts:
                        counts[background_label] = 0
                    counts[background_label] += background_count
                    # Acumular en total (no sobrescribir)
                    if background_label not in total_class_counts:
                        total_class_counts[background_label] = 0
                    total_class_counts[background_label] += background_count

                # ===== Acumular METADATA GLOBAL =====
                ch_names = list(raw_data.info["ch_names"])
                ch_types = raw_data.get_channel_types()
                ch_types_count = dict(Counter(ch_types))
                duration_sec = float(data.shape[1] / sfreq)

                unique_sfreqs.add(float(sfreq))
                union_channels.update(ch_names)
                ch_types_total.update(ch_types_count)
                total_duration_sec += duration_sec

                # ===== Calcular y acumular METADATA EXTENDIDA =====
                try:
                    # Guardar primer raw para inferir montage una sola vez
                    if first_raw_for_analysis is None:
                        first_raw_for_analysis = raw_data
                        # Inferir montage del primer archivo
                        montage_info = _infer_montage(ch_names)
                        print(f"[META] Montage detectado: {montage_info['type']} ({montage_info['matched_channels']}/{montage_info['total_channels']} canales)")

                    # Calcular estadísticas de sesión
                    session_stats = _calculate_session_stats(raw_data, events, labels_dict, str(file), sfreq)
                    all_sessions.append(session_stats)
                    print(f"[META] Sesión: {session_stats['subject']}/{session_stats['session']}, duración: {session_stats['duration_sec']:.1f}s, eventos: {session_stats['n_events']}")

                    # Detectar bad channels
                    bad_chans = _detect_bad_channels(raw_data)
                    if bad_chans:
                        all_bad_channels.update(bad_chans)
                        print(f"[META] Bad channels detectados: {bad_chans}")

                    # Calcular estadísticas por canal (solo primeros 3 archivos para no sobrecargar)
                    if total_files_processed < 3:
                        ch_stats = _calculate_channel_stats(raw_data)
                        # Acumular para promediar después
                        for ch, stats in ch_stats.items():
                            if ch not in channel_stats_accumulated:
                                channel_stats_accumulated[ch] = []
                            channel_stats_accumulated[ch].append(stats)

                    # Calcular bandas de frecuencia (solo primeros 2 archivos para no sobrecargar)
                    if total_files_processed < 2:
                        freq_bands = _calculate_frequency_bands(raw_data, sfreq)
                        # Acumular para promediar después
                        for band, band_data in freq_bands.items():
                            if band not in frequency_bands_accumulated:
                                frequency_bands_accumulated[band] = []
                            frequency_bands_accumulated[band].append(band_data['mean_power'])
                        print(f"[META] Bandas de frecuencia calculadas")

                    total_files_processed += 1

                except Exception as e:
                    print(f"[META] Error calculando metadata extendida: {e}")

            # ===================== EDF (mismo patrón de skip Data/Labels) =====================
            if ext == ".edf":
                print("using .edf")
                raw_data = self.read_edf(str(file))
                data, times = raw_data.get_data(return_times=True)

                # Guarda Data/Labels con SKIP si existen
                auxFilePath = get_Data_filePath(str(file))
                auxLabelPath = get_Label_filePath(str(file))
                os.makedirs(os.path.dirname(auxFilePath), exist_ok=True)
                os.makedirs(os.path.dirname(auxLabelPath), exist_ok=True)

                if os.path.exists(auxFilePath):
                    print(f"[SKIP] Data ya existe: {auxFilePath}")
                else:
                    np.save(auxFilePath, data)

                # Etiquetas EDF (tu lógica actual)
                print(raw_data.info["ch_names"])
                events = mne.find_events(raw_data, stim_channel="Trigger")
                events = events[events[:, 2] == 65380]

                current_file = os.path.basename(str(file))
                match = re.search(r"run-(\d+)", current_file)
                if match:
                    run_number = int(match.group(1))
                else:
                    run_number = None
                print(f"Run number is: {run_number}")

                if run_number is not None:
                    label_file = os.path.join(
                        "Data/ciscoEEG/ds005170-1.1.2/textdataset",
                        f"split_data_{run_number}.xlsx",
                    )
                    labels_df = pd.read_excel(label_file)
                    label_list = labels_df.iloc[:, 0].tolist()
                    if len(label_list) != len(events):
                        print(
                            f"Warning: Number of labels ({len(label_list)}) != Number of events ({len(events)}). "
                            f"Truncating to min length."
                        )
                        min_len = min(len(label_list), len(events))
                        label_list = label_list[:min_len]
                        events = events[:min_len]
                else:
                    label_list = []

                label_array = np.zeros((1, data.shape[1]), dtype=object)
                sfreq_f = float(raw_data.info["sfreq"])
                label_duration_sec = 3.2
                label_duration_samples = int(label_duration_sec * sfreq_f)

                # Etiquetar eventos
                for event, label in zip(events, label_list):
                    sample_idx = event[0]
                    start = sample_idx
                    end = min(sample_idx + label_duration_samples, data.shape[1])
                    label_array[0, start:end] = label

                # ===== RE-ETIQUETAR BACKGROUND (resto de la señal) =====
                # Siempre usar "rest" como etiqueta de background
                # Las transformadas se encargarán del re-etiquetado específico (P300 binario, etc.)
                background_label = "rest"

                # Llenar zonas sin etiquetar (donde es 0 o vacío)
                background_mask = (label_array[0] == 0) | (label_array[0] == '') | (label_array[0] == '0')
                label_array[0, background_mask] = background_label
                print(f"[LABELS] Background re-etiquetado como 'rest': {background_mask.sum()} muestras")

                label_array = label_array.astype(str)

                if os.path.exists(auxLabelPath):
                    print(f"[SKIP] Labels ya existe: {auxLabelPath}")
                else:
                    np.save(auxLabelPath, label_array)

                # Acumular METADATA GLOBAL
                ch_names = list(raw_data.info["ch_names"])
                ch_types = raw_data.get_channel_types()
                ch_types_count = dict(Counter(ch_types))
                duration_sec = float(data.shape[1] / sfreq_f)
                unique_sfreqs.add(float(sfreq_f))
                union_channels.update(ch_names)
                ch_types_total.update(ch_types_count)
                total_duration_sec += duration_sec

            # ===================== BRAINVISION (.vhdr) =====================
            if ext == ".vhdr":
                print(f"[VHDR] Procesando archivo BrainVision: {file}")

                # --- SKIP temprano si ya existen derivados ---
                auxFilePath  = get_Data_filePath(str(file))
                auxLabelPath = get_Label_filePath(str(file))
                labels_dir   = os.path.dirname(auxLabelPath)
                events_dir   = os.path.join(os.path.dirname(labels_dir), "Events")

                data_exists  = os.path.exists(auxFilePath)
                labels_exist = os.path.exists(auxLabelPath)
                events_ready = os.path.isdir(events_dir) and any(
                    fn.endswith(".npy") for fn in os.listdir(events_dir)
                )

                if data_exists and labels_exist and events_ready:
                    print(f"[SKIP-VHDR] Derivados ya existen para {file}. Saltando lectura.")
                    # Muestrear metadata si es necesario
                    if not sampled_meta_done:
                        try:
                            raw_hdr = self.read_brainvision(str(file))
                            sampled_sfreq = float(raw_hdr.info["sfreq"])
                            sampled_ch_names = list(raw_hdr.info["ch_names"])
                            sampled_ch_types_count = dict(Counter(raw_hdr.get_channel_types()))
                            sampled_meta_done = True
                            unique_sfreqs.add(sampled_sfreq)
                            union_channels.update(sampled_ch_names)
                            ch_types_total.update(sampled_ch_types_count)
                            print(f"[META] Sampled (skip branch VHDR): sfreq={sampled_sfreq}, n_channels={len(sampled_ch_names)}")
                        except Exception as e:
                            print(f"[META] Error sampling VHDR on skip: {e}")
                    continue

                # --- Procesamiento normal ---
                raw_data = self.read_brainvision(str(file))
                print(f"[VHDR] Leído archivo BrainVision: {str(file)}")
                print(f"[VHDR] Canales: {raw_data.info.ch_names}")

                # Extraer eventos desde annotations (.vmrk markers)
                events, event_id_dict = mne.events_from_annotations(raw_data, verbose=True)
                print(f"[VHDR] Eventos encontrados: {len(events)}")
                print(f"[VHDR] Event ID mapping desde annotations: {event_id_dict}")

                # Crear mapping inverso: ID numérico → nombre limpio
                # "Comment/Down" → "Down", "Stimulus/S  1" → "S1"
                vhdr_mapping = {}
                for annotation_name, numeric_id in event_id_dict.items():
                    clean_name = str(annotation_name).split('/')[-1].strip().replace(' ', '')
                    vhdr_mapping[int(numeric_id)] = clean_name
                print(f"[VHDR] Mapping limpio: {vhdr_mapping}")

                # Detectar tipo de dataset (Inner Speech vs genérico)
                dataset_type = _detect_dataset_type(events)

                if dataset_type == "inner_speech":
                    # Usar extracción específica de Inner Speech
                    cues = _inner_speech_cues(events)
                    labels_dict = LABELS  # {31: "arriba", 32: "abajo", ...}
                    print(f"[VHDR] Usando extracción Inner Speech: {len(cues)} cues")
                else:
                    # Extracción genérica: usar nombres de annotations
                    cues, labels_dict = _generic_event_extraction(events, event_id_mapping=vhdr_mapping)
                    print(f"[VHDR] Usando extracción genérica: {len(cues)} eventos, clases: {list(labels_dict.values())}")

                # Extraer datos
                data, _ = raw_data.get_data(return_times=True)
                sfreq = float(raw_data.info["sfreq"])
                print(f"[VHDR] Data shape: {data.shape}, sfreq: {sfreq} Hz")

                # ===== Etiquetas por muestra =====
                label_array = np.zeros((1, data.shape[1]), dtype=object)
                label_duration_sec = 3.2
                label_duration_samples = int(label_duration_sec * sfreq)

                # Etiquetar eventos
                for sample_idx, eid in cues:
                    start = sample_idx
                    end = min(sample_idx + label_duration_samples, data.shape[1])
                    label_array[0, start:end] = labels_dict[eid]

                # ===== RE-ETIQUETAR BACKGROUND (resto de la señal) =====
                # Siempre usar "rest" como etiqueta de background
                # Las transformadas se encargarán del re-etiquetado específico (P300 binario, etc.)
                background_label = "rest"

                # Llenar zonas sin etiquetar (donde es 0 o vacío)
                background_mask = (label_array[0] == 0) | (label_array[0] == '') | (label_array[0] == '0')
                label_array[0, background_mask] = background_label
                print(f"[LABELS] Background re-etiquetado como 'rest': {background_mask.sum()} muestras")

                label_array = label_array.astype(str)

                # ===== Guardar Data/Labels =====
                os.makedirs(os.path.dirname(auxFilePath), exist_ok=True)
                os.makedirs(os.path.dirname(auxLabelPath), exist_ok=True)

                if os.path.exists(auxFilePath):
                    print(f"[SKIP] Data ya existe: {auxFilePath}")
                else:
                    np.save(auxFilePath, data)
                    print(f"[VHDR] Guardado raw data: {auxFilePath}")

                if os.path.exists(auxLabelPath):
                    print(f"[SKIP] Labels ya existe: {auxLabelPath}")
                else:
                    np.save(auxLabelPath, label_array)
                    print(f"[VHDR] Guardado labels: {auxLabelPath}")

                # Conteo por clase
                counts = {labels_dict[k]: 0 for k in labels_dict}
                for _, eid in cues:
                    counts[labels_dict[eid]] += 1
                print(f"[VHDR] Eventos por clase en {Path(file).name}: {counts}")
                total_class_counts.update(counts)

                # ===== Generar archivos individuales en Events/ =====
                os.makedirs(events_dir, exist_ok=True)

                for (cue_sample, eid) in cues:
                    class_name = labels_dict[eid]

                    # Delimitación del evento
                    start_sample = cue_sample
                    end_sample = min(cue_sample + label_duration_samples, data.shape[1])

                    # Extraer segmento
                    X_event = data[:, start_sample:end_sample]

                    # Calcular tiempos
                    start_time = start_sample / sfreq
                    end_time = end_sample / sfreq

                    # Nombre del archivo: clase[inicio]{fin}.npy
                    safe_class = re.sub(r'[\\/:*?"<>|]', "_", class_name)
                    out_name = f"{safe_class}[{start_time:.3f}]{{{end_time:.3f}}}.npy"
                    out_path = os.path.join(events_dir, out_name)

                    # SKIP si ya existe
                    if os.path.exists(out_path):
                        print(f"[SKIP] Event ya existe: {out_path}")
                        continue

                    np.save(out_path, X_event)
                    print(f"[VHDR] Guardado evento: {out_path} | clase={class_name} | shape={X_event.shape}")

                # ===== EXTRAER VENTANAS DE BACKGROUND =====
                print(f"[VHDR] Extrayendo ventanas de background (etiquetadas como '{background_label}')...")

                # Encontrar índices donde está el background
                background_indices = np.where(label_array[0] == background_label)[0]

                if len(background_indices) > 0:
                    # Segmentar en ventanas del mismo tamaño que eventos
                    window_size = label_duration_samples
                    n_windows = len(background_indices) // window_size

                    background_count = 0
                    for i in range(n_windows):
                        start_idx = background_indices[i * window_size]
                        end_idx = start_idx + window_size

                        # Verificar que la ventana completa sea background (contigua)
                        if end_idx <= data.shape[1]:
                            window_labels = label_array[0, start_idx:end_idx]
                            if np.all(window_labels == background_label):
                                # Extraer ventana
                                X_background = data[:, start_idx:end_idx]

                                # Calcular tiempos
                                start_time = start_idx / sfreq
                                end_time = end_idx / sfreq

                                # Nombre del archivo
                                out_name = f"{background_label}[{start_time:.3f}]{{{end_time:.3f}}}.npy"
                                out_path = os.path.join(events_dir, out_name)

                                # SKIP si ya existe
                                if os.path.exists(out_path):
                                    continue

                                np.save(out_path, X_background)
                                background_count += 1

                    print(f"[VHDR] Extraídas {background_count} ventanas de background | shape={X_background.shape if background_count > 0 else 'N/A'}")

                    # Actualizar conteos (acumular correctamente)
                    if background_label not in counts:
                        counts[background_label] = 0
                    counts[background_label] += background_count
                    # Acumular en total (no sobrescribir)
                    if background_label not in total_class_counts:
                        total_class_counts[background_label] = 0
                    total_class_counts[background_label] += background_count

                # ===== Acumular metadata global =====
                ch_names = list(raw_data.info["ch_names"])
                ch_types = raw_data.get_channel_types()
                ch_types_count = dict(Counter(ch_types))
                duration_sec = float(data.shape[1] / sfreq)

                unique_sfreqs.add(float(sfreq))
                union_channels.update(ch_names)
                ch_types_total.update(ch_types_count)
                total_duration_sec += duration_sec

                # ===== Calcular y acumular METADATA EXTENDIDA (VHDR) =====
                try:
                    # Guardar primer raw para inferir montage una sola vez
                    if first_raw_for_analysis is None:
                        first_raw_for_analysis = raw_data
                        # Inferir montage del primer archivo
                        montage_info = _infer_montage(ch_names)
                        print(f"[META] Montage detectado: {montage_info['type']} ({montage_info['matched_channels']}/{montage_info['total_channels']} canales)")

                    # Calcular estadísticas de sesión
                    session_stats = _calculate_session_stats(raw_data, events, labels_dict, str(file), sfreq)
                    all_sessions.append(session_stats)
                    print(f"[META] Sesión: {session_stats['subject']}/{session_stats['session']}, duración: {session_stats['duration_sec']:.1f}s, eventos: {session_stats['n_events']}")

                    # Detectar bad channels
                    bad_chans = _detect_bad_channels(raw_data)
                    if bad_chans:
                        all_bad_channels.update(bad_chans)
                        print(f"[META] Bad channels detectados: {bad_chans}")

                    # Calcular estadísticas por canal (solo primeros 3 archivos para no sobrecargar)
                    if total_files_processed < 3:
                        ch_stats = _calculate_channel_stats(raw_data)
                        # Acumular para promediar después
                        for ch, stats in ch_stats.items():
                            if ch not in channel_stats_accumulated:
                                channel_stats_accumulated[ch] = []
                            channel_stats_accumulated[ch].append(stats)

                    # Calcular bandas de frecuencia (solo primeros 2 archivos para no sobrecargar)
                    if total_files_processed < 2:
                        freq_bands = _calculate_frequency_bands(raw_data, sfreq)
                        # Acumular para promediar después
                        for band, band_data in freq_bands.items():
                            if band not in frequency_bands_accumulated:
                                frequency_bands_accumulated[band] = []
                            frequency_bands_accumulated[band].append(band_data['mean_power'])
                        print(f"[META] Bandas de frecuencia calculadas")

                    total_files_processed += 1

                except Exception as e:
                    print(f"[META] Error calculando metadata extendida: {e}")

        print("ending")

        # ====== ESCRIBIR JSON GLOBAL (una sola vez) EN LA RAÍZ Aux/ ======
        # Si por alguna razón no logramos samplear, deja None/[] para esos campos.
        sampling_frequency_hz = (sampled_sfreq if sampled_meta_done else (sorted(unique_sfreqs)[0] if unique_sfreqs else None))
        n_channels = (len(sampled_ch_names) if sampled_meta_done else (len(union_channels) if union_channels else None))
        channel_names_out = (sampled_ch_names if sampled_meta_done else sorted(union_channels))
        channel_types_out = (sampled_ch_types_count if sampled_meta_done else dict(ch_types_total))

        # Usar clases detectadas dinámicamente (de total_class_counts) en lugar de LABELS hardcodeado
        detected_classes = sorted(total_class_counts.keys())

        # ===== Procesar metadata extendida acumulada =====
        # Promediar estadísticas de canales
        channel_stats_final = {}
        for ch, stats_list in channel_stats_accumulated.items():
            if stats_list:
                channel_stats_final[ch] = {
                    "mean": float(np.mean([s["mean"] for s in stats_list])),
                    "std": float(np.mean([s["std"] for s in stats_list])),
                    "min": float(np.min([s["min"] for s in stats_list])),
                    "max": float(np.max([s["max"] for s in stats_list])),
                    "variance": float(np.mean([s["variance"] for s in stats_list])),
                    "rms": float(np.mean([s["rms"] for s in stats_list]))
                }

        # Promediar bandas de frecuencia
        frequency_bands_final = {}

        # Definir rangos predefinidos
        ranges = {
            "delta": [0.5, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "beta": [13, 30],
            "gamma": [30, min(100, sampling_frequency_hz / 2 - 1) if sampling_frequency_hz else 100]
        }

        for band, powers in frequency_bands_accumulated.items():
            if powers:
                frequency_bands_final[band] = {
                    "range": ranges.get(band, [0, 100]),
                    "mean_power": float(np.mean(powers))
                }

        # Calcular métricas de calidad
        quality_metrics = {
            "total_bad_channels": len(all_bad_channels),
            "bad_channels_list": sorted(list(all_bad_channels)),
            "n_events_total": sum(total_class_counts.values()),
            "n_sessions": len(all_sessions),
            "events_balance": "balanced" if (max(total_class_counts.values()) / min(total_class_counts.values()) if total_class_counts and min(total_class_counts.values()) > 0 else 1) <= 1.5 else "imbalanced",
            "class_imbalance_ratio": float(max(total_class_counts.values()) / min(total_class_counts.values()) if total_class_counts and min(total_class_counts.values()) > 0 else 1.0)
        }

        metadata = {
            # ===== Metadata básica (ya existente) =====
            "dataset_name": self.name,
            "num_classes": len(detected_classes),
            "classes": detected_classes,
            "sampling_frequency_hz": sampling_frequency_hz,
            "n_channels": n_channels,
            "channel_names": channel_names_out,
            "channel_types_count": channel_types_out,
            "total_duration_sec": round(total_duration_sec, 6),
            "class_counts_total": dict(total_class_counts),
            "eeg_unit": "V",

            # ===== Nueva metadata extendida =====
            "sessions": all_sessions,
            "montage": montage_info if montage_info else {"type": "unknown", "positions": {}, "has_positions": False},
            "channel_stats": channel_stats_final,
            "frequency_bands": frequency_bands_final,
            "quality_metrics": quality_metrics,
            "total_files": len(files),
            "files_processed": total_files_processed
        }

        print(f"[META] ✅ Metadata extendida calculada:")
        print(f"  - Sesiones: {len(all_sessions)}")
        print(f"  - Montage: {montage_info['type'] if montage_info else 'unknown'}")
        print(f"  - Canales con estadísticas: {len(channel_stats_final)}")
        print(f"  - Bandas de frecuencia: {len(frequency_bands_final)}")
        print(f"  - Bad channels: {len(all_bad_channels)}")
        print(f"  - Calidad: {quality_metrics['events_balance']}")

        meta_out = os.path.join(aux_root, "dataset_metadata.json")
        try:
            with open(meta_out, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=_json_fallback)
            print(f"[META] Escrito {meta_out}")
        except Exception as e:
            print(f"[META] Error escribiendo {meta_out}: {e}")

        return {
            "status": 200,
            "message": f"Se han encontrado {len(files)}  sesiones ",
            "files": files,
            "entries": all_entries,
        }

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

    def read_edf(self, path):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
        raw = mne.io.read_raw_edf(str(file_path), verbose=True, infer_types=True, preload=True)
        return raw

    def read_brainvision(self, path):
        """
        Lee archivos BrainVision (.vhdr + .eeg + .vmrk).

        Args:
            path: Ruta al archivo .vhdr (MNE busca automáticamente .eeg y .vmrk)

        Returns:
            raw: Objeto Raw de MNE con los datos EEG
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
        # MNE lee el .vhdr y automáticamente busca .eeg y .vmrk en el mismo directorio
        raw = mne.io.read_raw_brainvision(str(file_path), verbose=True, preload=True)
        return raw

    def load_signal_data(selected_file_path):
        print(f"Selected path is: {selected_file_path}")
        base = Path("Data")
        full_path = base / selected_file_path
        #We check if file passed is valid
        if not selected_file_path:
            print("Invalid or missing file.")
            return no_update, True  # Keep interval disabled

        #Now we check if file is a valid format
        if not selected_file_path.endswith(".npy"):
            #We check if there's a corresponding .npy in Aux
            if not os.path.exists(f"Data/{selected_file_path}"):
                return no_update, True

            mappedFilePath = get_Data_filePath(f"Data/{selected_file_path}")

            if os.path.exists(mappedFilePath):
                print(mappedFilePath)
                signal = np.load(mappedFilePath, mmap_mode='r')
                full_path = Path(mappedFilePath)
            else:
                return no_update, True
        else:
            # Load the signal
            signal = np.load(full_path, mmap_mode='r')

        # Load metadata JSON if available
        metadata = None
        metadata_path = full_path.parent.parent / "dataset_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"✅ Metadata cargada desde: {metadata_path}")
            except Exception as e:
                print(f"⚠️ Error cargando metadata: {e}")
                metadata = None

        # we want to extract the parent path and the file name to obtain the label that is in the parent directory and in a folder named labels with the same file name
        labels_path = full_path.parent / "Labels" / full_path.name
        print("Buscando etiquetas en:", labels_path, "| Existe?", labels_path.exists())

        if not labels_path.exists():
            print("Labels don't exist")
            # We create dummy labels for no crash the application
            labels = np.zeros(signal.shape[0], dtype=int)
        else:
            labels = np.load(labels_path, allow_pickle=True)

        if signal.shape[0] < signal.shape[1]:
            signal = signal.T

        length_of_segment = 60

        for i in range(0, signal.shape[0], length_of_segment):
            randint = np.random.randint(0, 10)
            if randint < 3:
                labels[i:i + length_of_segment] = np.random.randint(1, 5)

        labels = labels.astype(str)
        unique_labels = np.unique(labels)

        # Usar sistema centralizado de colores para consistencia
        from shared.class_colors import get_class_color
        label_color_map = {}

        for idx, label in enumerate(unique_labels):
            label_color_map[str(label)] = get_class_color(str(label), idx)

        '''
            Change this later when optimizing right now it's as is because the file wont load in time and the server times out, we need to optimize to load in chunks
            or something like that, therefore we only load the first 50k points
        '''

        signal = signal[:5000, :]
        labels = labels.reshape(-1)[:5000]

        print(f"signal shape: {signal.shape}")
        print(f"labels shape: {labels.shape}")

        # We encode the vector
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels.ravel())

        print(f"One hot encoded labels shape: {labels.shape}")

        # Serialize the signal (convert to list to make it JSON serializable)
        signal_dict = {
            "data": signal.tolist(),
            "num_channels": signal.shape[1],
            "num_timepoints": signal.shape[0],
            "labels": labels.tolist(),
            "label_color_map": label_color_map,
            "metadata": metadata  # Include metadata from JSON
        }

        return signal_dict, False  # Enable interval



    # =========================================================================
    # Sistema de Mapeo de Canales
    # =========================================================================

    @staticmethod
    def get_channel_mapping(dataset_name):
        """
        Obtiene el mapeo completo de nombres de canales → índices de fila.

        Args:
            dataset_name: Nombre del dataset (ej: "nieto_inner_speech")

        Returns:
            dict: Mapeo {channel_name: row_index}
            Ejemplo: {"A1": 0, "A2": 1, ..., "Status": 136}
        """
        try:
            # Construir ruta al metadata
            aux_path = Path("Aux") / dataset_name / "dataset_metadata.json"

            if not aux_path.exists():
                print(f"[get_channel_mapping] ERROR: No se encontró {aux_path}")
                return {}

            # Leer metadata
            with open(aux_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            channel_names = metadata.get("channel_names", [])

            if not channel_names:
                print(f"[get_channel_mapping] WARN: dataset_metadata.json no tiene channel_names")
                return {}

            # Crear mapeo: nombre → índice
            mapping = {name: idx for idx, name in enumerate(channel_names)}

            print(f"[get_channel_mapping] Mapeo creado: {len(mapping)} canales")
            return mapping

        except Exception as e:
            print(f"[get_channel_mapping] ERROR: {e}")
            return {}

    @staticmethod
    def get_channel_index(dataset_name, channel_name):
        """
        Obtiene el índice de fila de un canal específico.

        Args:
            dataset_name: Nombre del dataset
            channel_name: Nombre del canal (ej: "A1", "B5")

        Returns:
            int o None: Índice de fila (0-indexed), o None si no existe

        Ejemplo:
            >>> Dataset.get_channel_index("nieto_inner_speech", "A1")
            0
            >>> Dataset.get_channel_index("nieto_inner_speech", "Status")
            136
        """
        mapping = Dataset.get_channel_mapping(dataset_name)
        return mapping.get(channel_name)

    @staticmethod
    def get_channels_indices(dataset_name, channel_names):
        """
        Obtiene los índices de fila de múltiples canales.

        Args:
            dataset_name: Nombre del dataset
            channel_names: Lista de nombres de canales

        Returns:
            list[int]: Lista de índices válidos (omite canales no encontrados)

        Ejemplo:
            >>> Dataset.get_channels_indices("nieto_inner_speech", ["A1", "A2", "B5"])
            [0, 1, 36]
        """
        mapping = Dataset.get_channel_mapping(dataset_name)
        indices = []

        for ch_name in channel_names:
            idx = mapping.get(ch_name)
            if idx is not None:
                indices.append(idx)
            else:
                print(f"[get_channels_indices] WARN: Canal '{ch_name}' no encontrado en dataset '{dataset_name}'")

        return indices

    @staticmethod
    def extract_channels(data_array, channel_names, dataset_name):
        """
        Extrae canales específicos de un array de datos.

        Args:
            data_array: Array NumPy con shape (n_channels, n_samples)
            channel_names: Lista de nombres de canales a extraer
            dataset_name: Nombre del dataset (para mapeo)

        Returns:
            np.ndarray: Array filtrado con shape (len(channel_names), n_samples)

        Ejemplo:
            >>> data = np.load("evento.npy")  # Shape: (137, 2612)
            >>> filtered = Dataset.extract_channels(data, ["A1", "A2"], "nieto_inner_speech")
            >>> filtered.shape
            (2, 2612)
        """
        indices = Dataset.get_channels_indices(dataset_name, channel_names)

        if not indices:
            print(f"[extract_channels] ERROR: No se encontraron canales válidos")
            return np.array([])

        # Validar shape
        if data_array.ndim != 2:
            print(f"[extract_channels] ERROR: data_array debe ser 2D, recibido shape={data_array.shape}")
            return np.array([])

        # Extraer filas correspondientes
        filtered = data_array[indices, :]

        print(f"[extract_channels] Extraídos {len(indices)} canales: {channel_names}")
        print(f"[extract_channels] Shape original: {data_array.shape} → Shape filtrado: {filtered.shape}")

        return filtered

    @staticmethod
    def load_event_with_channels(event_path, channel_names, dataset_name):
        """
        Carga un archivo de evento (.npy) y extrae solo los canales especificados.

        Args:
            event_path: Ruta al archivo .npy del evento
            channel_names: Lista de nombres de canales a cargar
            dataset_name: Nombre del dataset

        Returns:
            dict: {
                "data": np.ndarray con shape (len(channel_names), n_samples),
                "channel_names": list[str] de canales cargados,
                "channel_indices": list[int] de índices originales,
                "original_shape": tuple del shape original,
                "event_path": str ruta del archivo
            }

        Ejemplo:
            >>> result = Dataset.load_event_with_channels(
            ...     "Aux/.../Events/abajo[439.357]{441.908}.npy",
            ...     ["A1", "A2", "B5"],
            ...     "nieto_inner_speech"
            ... )
            >>> result["data"].shape
            (3, 2612)
        """
        try:
            # Cargar evento completo
            event_full = np.load(event_path, allow_pickle=False)
            original_shape = event_full.shape

            print(f"[load_event_with_channels] Cargando: {event_path}")
            print(f"[load_event_with_channels] Shape original: {original_shape}")

            # Extraer canales
            filtered_data = Dataset.extract_channels(event_full, channel_names, dataset_name)

            # Obtener índices para referencia
            indices = Dataset.get_channels_indices(dataset_name, channel_names)

            return {
                "data": filtered_data,
                "channel_names": channel_names,
                "channel_indices": indices,
                "original_shape": original_shape,
                "event_path": str(event_path)
            }

        except Exception as e:
            print(f"[load_event_with_channels] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {
                "data": np.array([]),
                "channel_names": [],
                "channel_indices": [],
                "original_shape": None,
                "event_path": str(event_path)
            }

    @staticmethod
    def get_all_channel_names(dataset_name):
        """
        Obtiene la lista completa de nombres de canales del dataset.

        Args:
            dataset_name: Nombre del dataset

        Returns:
            list[str]: Lista de nombres de canales en orden (índice = row)

        Ejemplo:
            >>> Dataset.get_all_channel_names("nieto_inner_speech")
            ['A1', 'A2', ..., 'Status']
        """
        try:
            aux_path = Path("Aux") / dataset_name / "dataset_metadata.json"

            if not aux_path.exists():
                return []

            with open(aux_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            return metadata.get("channel_names", [])

        except Exception as e:
            print(f"[get_all_channel_names] ERROR: {e}")
            return []
        



import os
from typing import List,  Sequence

import numpy as np


NDArray = np.ndarray

# helper to verify if exist the path
def _assert_npy_path(p: str) -> None:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Path not found: {p}")
    if not p.lower().endswith(".npy"):
        raise ValueError(f"Expected .npy file, got: {p}")


def _load_and_concat(paths: Sequence[str]) -> NDArray:
    """
    Loads one or more .npy files and concatenates along axis=0.
    Each file must have compatible first dimension.
    """
    if not paths:
        raise ValueError("No paths provided.")
    arrays: List[NDArray] = []
    for p in paths:
        _assert_npy_path(p)
        arr = np.load(p, allow_pickle=False)
        arrays.append(arr)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=0)


def create_subset_dataset(dataset_name: str, percentage: float, train_split: float, seed: int = 42, materialize: bool = False, model_type: str = None):
    """
    Crea un subset del dataset especificado.

    Args:
        dataset_name: Nombre del dataset base
        percentage: Porcentaje de datos a incluir (0-100)
        train_split: Porcentaje para entrenamiento (0-100)
        seed: Semilla para reproducibilidad
        materialize: Si crear archivos físicos o solo metadata
        model_type: Tipo de modelo ('p300' o 'inner') para filtrar y balancear clases

    Returns:
        dict: {"status": int, "message": str, "subset_dir": str}
    """
    from datetime import datetime
    import random
    from pathlib import Path
    import json
    from backend.classes.Experiment import Experiment

    try:
        # Validar parámetros
        if not (0 < percentage <= 100):
            return {"status": 400, "message": f"percentage debe estar entre 0 y 100, recibido: {percentage}"}

        if not (0 < train_split <= 100):
            return {"status": 400, "message": f"train_split debe estar entre 0 y 100, recibido: {train_split}"}

        # Buscar eventos en estructura BIDS jerárquica: Aux/{dataset}/**/Events/*.npy
        dataset_aux_path = Path("Aux") / dataset_name
        dataset_raw_path = Path(dataset_name)

        # Intentar primero en Aux (estructura BIDS)
        if dataset_aux_path.exists():
            # Buscar recursivamente en toda la jerarquía de subdirectorios
            event_files = list(dataset_aux_path.glob("**/Events/*.npy"))
            labels_base_dir = dataset_aux_path
            events_source = str(dataset_aux_path)
        elif dataset_raw_path.exists():
            # Fallback a dataset raw
            event_files = list(dataset_raw_path.glob("**/Events/*.npy"))
            labels_base_dir = dataset_raw_path
            events_source = str(dataset_raw_path)
        else:
            return {"status": 404, "message": f"Dataset no encontrado: {dataset_name}"}

        if len(event_files) == 0:
            return {"status": 404, "message": f"No se encontraron eventos en {dataset_name}. Buscado en: Aux/{dataset_name}/**/Events/ y {dataset_name}/**/Events/"}

        print(f"[create_subset_dataset] Encontrados {len(event_files)} eventos en {events_source}")

        # Clasificar eventos por clase (extraer del nombre del archivo)
        events_by_class = {}
        for event_file in event_files:
            # Extraer clase del nombre: "abajo[123.45]{678.90}.npy" -> "abajo"
            filename = event_file.stem
            class_name = filename.split('[')[0].strip() if '[' in filename else filename.split('_')[0]

            if class_name not in events_by_class:
                events_by_class[class_name] = []
            events_by_class[class_name].append(event_file)

        # ✅ Filtrar clases según model_type
        if model_type == "inner":
            # Para Inner Speech: excluir "rest" (solo abajo, arriba, derecha, izquierda)
            events_by_class = {k: v for k, v in events_by_class.items() if k.lower() != "rest"}
            print(f"[create_subset_dataset] Filtradas clases para Inner Speech (sin 'rest')")
        # Para P300: incluir todas las clases (se re-etiquetarán a Target/NonTarget después)

        classes = list(events_by_class.keys())
        print(f"[create_subset_dataset] Clases seleccionadas: {classes}")

        # Configurar seed para reproducibilidad
        random.seed(seed)
        np.random.seed(seed)

        # ✅ BALANCEO DE CLASES: Tomar el mismo número de muestras por clase
        # Encontrar la clase con menos muestras
        min_samples = min(len(events) for events in events_by_class.values())

        # Calcular número de muestras por clase respetando el porcentaje
        samples_per_class = max(1, int(min_samples * percentage / 100.0))

        print(f"[create_subset_dataset] Balanceando clases: {samples_per_class} muestras por clase")

        # Seleccionar muestras balanceadas
        selected_events = []
        for class_name, class_events in events_by_class.items():
            # Tomar exactamente samples_per_class de cada clase
            selected = random.sample(class_events, min(samples_per_class, len(class_events)))
            selected_events.extend(selected)
            print(f"[create_subset_dataset]   {class_name}: {len(selected)} eventos")

        # Mezclar eventos seleccionados
        random.shuffle(selected_events)

        # Dividir en train/test
        n_total = len(selected_events)
        n_train = int(n_total * train_split / 100.0)
        n_test = n_total - n_train

        train_events = selected_events[:n_train]
        test_events = selected_events[n_train:]

        print(f"[create_subset_dataset] Total: {n_total}, Train: {n_train}, Test: {n_test}")

        # ===== CALCULAR DISTRIBUCIÓN DE CLASES POR SPLIT =====
        class_distribution = {}
        for class_name in classes:
            # Contar cuántos eventos de esta clase hay en train
            train_count = sum(1 for event_path in train_events
                            if Path(event_path).stem.split('[')[0].strip() == class_name)

            # Contar cuántos eventos de esta clase hay en test
            test_count = sum(1 for event_path in test_events
                           if Path(event_path).stem.split('[')[0].strip() == class_name)

            total_class = train_count + test_count

            class_distribution[class_name] = {
                "total_selected": total_class,
                "train": train_count,
                "test": test_count,
                "train_pct": round(train_count / n_train * 100, 2) if n_train > 0 else 0.0,
                "test_pct": round(test_count / n_test * 100, 2) if n_test > 0 else 0.0
            }

            print(f"[create_subset_dataset]   {class_name}: Train={train_count} ({class_distribution[class_name]['train_pct']:.1f}%) | Test={test_count} ({class_distribution[class_name]['test_pct']:.1f}%)")

        # Crear directorio de salida con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subset_dir = Path("Aux") / dataset_name / "generated_datasets" / timestamp
        subset_dir.mkdir(parents=True, exist_ok=True)

        # Obtener configuración actual del experimento
        try:
            experiment = Experiment._load_latest_experiment()
            experiment_filters = experiment.filters or []
            experiment_transforms = {}

            if experiment.transform and len(experiment.transform) > 0:
                last_transform = experiment.transform[-1]
                for k, v in last_transform.items():
                    if k not in ["id", "dimensionality_change"]:
                        experiment_transforms[k] = v
                        break
        except Exception as e:
            print(f"[create_subset_dataset] Warning: No se pudo cargar experimento: {e}")
            experiment_filters = []
            experiment_transforms = {}

        # Crear metadata con rutas absolutas para eventos (estructura BIDS jerárquica)
        metadata = {
            "timestamp": timestamp,
            "dataset_name": dataset_name,
            "percentage": percentage,
            "train_split": train_split,
            "seed": seed,
            "n_total_events": n_total,
            "n_train_events": n_train,
            "n_test_events": n_test,
            "classes": classes,
            "class_distribution": class_distribution,  # ✅ Distribución detallada por clase
            "train_files": [str(f) for f in train_events],  # Ruta completa (BIDS jerárquico)
            "test_files": [str(f) for f in test_events],    # Ruta completa (BIDS jerárquico)
            "events_source": events_source,  # Directorio base donde se buscaron eventos
            "labels_base_dir": str(labels_base_dir),  # Base para buscar labels en estructura paralela
            "materialized": materialize,
            "experiment_snapshot": {  # ✅ Requerido por _check_compatibility()
                "filters": experiment_filters,
                "transform": experiment_transforms
            }
        }

        # Guardar metadata
        metadata_file = subset_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[create_subset_dataset] ✅ Metadata guardada en {metadata_file}")

        # Crear manifests (train_manifest.json y test_manifest.json)
        # Formato: lista de dicts con "path", "class", "subject", "session"
        def create_manifest_entry(event_file: Path) -> dict:
            """Crea entrada de manifest a partir de archivo de evento"""
            filename = event_file.stem
            # Extraer clase del nombre: "abajo[123.45]{678.90}" -> "abajo"
            class_name = filename.split('[')[0].strip() if '[' in filename else filename.split('_')[0]

            # Intentar extraer subject/session de la ruta (estructura BIDS)
            # Ruta típica: .../sub-01/ses-01/eeg/Events/archivo.npy
            parts = event_file.parts
            subject = next((p for p in parts if p.startswith('sub-')), None)
            session = next((p for p in parts if p.startswith('ses-')), None)

            return {
                "path": str(event_file),
                "class": class_name,
                "subject": subject,
                "session": session
            }

        train_manifest = [create_manifest_entry(f) for f in train_events]
        test_manifest = [create_manifest_entry(f) for f in test_events]

        train_manifest_file = subset_dir / "train_manifest.json"
        test_manifest_file = subset_dir / "test_manifest.json"

        with open(train_manifest_file, 'w', encoding='utf-8') as f:
            json.dump(train_manifest, f, indent=2, ensure_ascii=False)

        with open(test_manifest_file, 'w', encoding='utf-8') as f:
            json.dump(test_manifest, f, indent=2, ensure_ascii=False)

        print(f"[create_subset_dataset] ✅ Train manifest guardado: {train_manifest_file}")
        print(f"[create_subset_dataset] ✅ Test manifest guardado: {test_manifest_file}")

        # Si materialize=True, copiar archivos físicamente
        if materialize:
            train_dir = subset_dir / "train"
            test_dir = subset_dir / "test"
            train_dir.mkdir(exist_ok=True)
            test_dir.mkdir(exist_ok=True)

            import shutil

            # Copiar eventos (solo señales, no hay labels individuales por evento)
            # La clase se extrae del nombre del archivo durante el entrenamiento
            for event_file in train_events:
                shutil.copy2(event_file, train_dir / event_file.name)

            for event_file in test_events:
                shutil.copy2(event_file, test_dir / event_file.name)

            print(f"[create_subset_dataset] ✅ Archivos materializados en {subset_dir}")
            print(f"[create_subset_dataset] ℹ️  Nota: Las labels se extraen del nombre del archivo (e.g., 'abajo[123]{456}.npy' → clase='abajo')")

        return {
            "status": 200,
            "message": f"Subset creado exitosamente: {n_train} train, {n_test} test",
            "subset_dir": str(subset_dir),
            "n_train": n_train,
            "n_test": n_test
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": 500,
            "message": f"Error creando subset: {str(e)}"
        }
