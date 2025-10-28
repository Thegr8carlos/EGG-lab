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
    for sample, _, eid in events:
        if eid == 15:  # start of run
            in_inner_run = False
        elif eid == 22:  # start of inner speech run
            in_inner_run = True
        elif eid == 16:  # end of run
            in_inner_run = False
        elif in_inner_run and eid in CUE_IDS:  # cue de clase dentro del run inner
            cues.append((int(sample), int(eid)))
    return cues


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


class Dataset:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.extensions_enabled = [".bdf", ".edf"]

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
        class_names = list(LABELS.values())
        total_class_counts = Counter({k: 0 for k in class_names})
        unique_sfreqs = set()
        union_channels = set()
        ch_types_total = Counter()
        total_duration_sec = 0.0

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

                # Eventos y cues de inner-speech (dentro de runs tag 22)
                events = mne.find_events(
                    raw_data, stim_channel="Status", shortest_event=1, verbose=True
                )
                inner_cues = _inner_speech_cues(events)  # (sample, eid) con eid en {31,32,33,34}
                print(f"[BDF] inner_speech_cues (run 22) encontrados: {len(inner_cues)}")

                # Data raw
                data, _ = raw_data.get_data(return_times=True)  # (n_channels, n_times)
                sfreq = float(raw_data.info["sfreq"])

                # ===== Etiquetas por muestra =====
                label_array = np.zeros((1, data.shape[1]), dtype=object)

                label_duration_sec = 3.2  # mantenemos tu valor actual
                label_duration_samples = int(label_duration_sec * sfreq)

                for sample_idx, eid in inner_cues:
                    start = sample_idx
                    end = min(sample_idx + label_duration_samples, data.shape[1])
                    label_array[0, start:end] = LABELS[eid]

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
                counts = {LABELS[k]: 0 for k in LABELS}
                for _, eid in inner_cues:
                    counts[LABELS[eid]] += 1
                print(f"Inner-speech en {Path(file).name}: {counts}")
                total_class_counts.update(counts)

                # ===== Un archivo .npy por evento en Events/ con formato <clase>[ini]{fin}.npy =====
                os.makedirs(events_dir, exist_ok=True)
                prefer_action_tags = True  # usa 44-45 si existen; si no, cae a 3.2s desde el cue

                for (cue_sample, eid) in inner_cues:
                    class_name = LABELS[eid]

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

                # ===== Acumular METADATA GLOBAL =====
                ch_names = list(raw_data.info["ch_names"])
                ch_types = raw_data.get_channel_types()
                ch_types_count = dict(Counter(ch_types))
                duration_sec = float(data.shape[1] / sfreq)

                unique_sfreqs.add(float(sfreq))
                union_channels.update(ch_names)
                ch_types_total.update(ch_types_count)
                total_duration_sec += duration_sec

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

                for event, label in zip(events, label_list):
                    sample_idx = event[0]
                    start = sample_idx
                    end = min(sample_idx + label_duration_samples, data.shape[1])
                    label_array[0, start:end] = label

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

        print("ending")

        # ====== ESCRIBIR JSON GLOBAL (una sola vez) EN LA RAÍZ Aux/ ======
        # Si por alguna razón no logramos samplear, deja None/[] para esos campos.
        sampling_frequency_hz = (sampled_sfreq if sampled_meta_done else (sorted(unique_sfreqs)[0] if unique_sfreqs else None))
        n_channels = (len(sampled_ch_names) if sampled_meta_done else (len(union_channels) if union_channels else None))
        channel_names_out = (sampled_ch_names if sampled_meta_done else sorted(union_channels))
        channel_types_out = (sampled_ch_types_count if sampled_meta_done else dict(ch_types_total))

        metadata = {
            "dataset_name": self.name,
            "num_classes": len(LABELS),
            "classes": class_names,
            "sampling_frequency_hz": sampling_frequency_hz,
            "n_channels": n_channels,
            "channel_names": channel_names_out,
            "channel_types_count": channel_types_out,
            "total_duration_sec": round(total_duration_sec, 6),
            "class_counts_total": dict(total_class_counts),
            "eeg_unit": "V",
        }

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

        label_color_map = {}

        for idx, label in enumerate(unique_labels):
            hue = (idx * 47) % 360
            label_color_map[str(label)] = f"hsl({hue}, 70%, 50%)"

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
        

from __future__ import annotations

import os
from typing import List,  Sequence

import numpy as np


NDArray = np.ndarray
# helperr to verify if exist the path
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
