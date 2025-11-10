# Soporte para BrainVision (.vhdr, .eeg, .vmrk)

## ✅ SÍ es posible generar lo mismo con MNE

Los archivos BrainVision funcionan así:
- **`.vhdr`** - Header file (MNE lee este)
- **`.eeg`** - Datos binarios (leído automáticamente)
- **`.vmrk`** - Markers/eventos (leído automáticamente)

MNE los lee igual que BDF: `mne.io.read_raw_brainvision(vhdr_file)`

---

## Cambios necesarios en `dataset.py`

### 1. Agregar extensión a la lista (línea 87)

```python
class Dataset:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.extensions_enabled = [".bdf", ".edf", ".vhdr"]  # ← Agregar .vhdr
```

### 2. Agregar método `read_brainvision()` (después de línea 611)

```python
def read_brainvision(self, path):
    """Lee archivos BrainVision (.vhdr + .eeg + .vmrk)"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"El archivo '{file_path}' no existe.")

    # MNE lee el .vhdr, y automáticamente busca .eeg y .vmrk en el mismo directorio
    raw = mne.io.read_raw_brainvision(str(file_path), verbose=True, preload=True)
    return raw
```

### 3. Agregar procesamiento en `upload_dataset()` (después de línea 577)

```python
# ===================== BRAINVISION (.vhdr) =====================
if ext == ".vhdr":
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
        print(f"[SKIP-VHDR] Derivados ya existen para {file}. Saltando.")
        continue

    # --- Lectura del archivo ---
    raw_data = self.read_brainvision(str(file))
    print(f"Leído BrainVision: {str(file)}, canales: {raw_data.info.ch_names}")

    # Extraer eventos (automáticamente desde .vmrk)
    events = mne.find_events(raw_data, verbose=True)
    print(f"[VHDR] Eventos encontrados: {len(events)}")

    # Detectar tipo de dataset (Inner Speech vs genérico)
    dataset_type = _detect_dataset_type(events)  # ← Nueva función

    if dataset_type == "inner_speech":
        cues = _inner_speech_cues(events)
        labels_dict = LABELS  # {31: "arriba", ...}
    else:
        # Genérico: extraer TODOS los event IDs
        cues, labels_dict = _generic_event_extraction(events)  # ← Nueva función

    print(f"[VHDR] Cues extraídos: {len(cues)} (tipo: {dataset_type})")

    # --- Procesamiento igual que BDF ---
    data, _ = raw_data.get_data(return_times=True)
    sfreq = float(raw_data.info["sfreq"])

    # Labels por muestra
    label_array = np.zeros((1, data.shape[1]), dtype=object)
    label_duration_sec = 3.2
    label_duration_samples = int(label_duration_sec * sfreq)

    for sample_idx, eid in cues:
        start = sample_idx
        end = min(sample_idx + label_duration_samples, data.shape[1])
        label_array[0, start:end] = labels_dict[eid]

    label_array = label_array.astype(str)

    # Guardar Data/Labels
    os.makedirs(os.path.dirname(auxFilePath), exist_ok=True)
    os.makedirs(os.path.dirname(auxLabelPath), exist_ok=True)

    if not os.path.exists(auxFilePath):
        np.save(auxFilePath, data)
    if not os.path.exists(auxLabelPath):
        np.save(auxLabelPath, label_array)

    # Conteo por clase
    counts = {labels_dict[k]: 0 for k in labels_dict}
    for _, eid in cues:
        counts[labels_dict[eid]] += 1
    print(f"Eventos BrainVision en {Path(file).name}: {counts}")
    total_class_counts.update(counts)

    # Generar Events/ individuales
    os.makedirs(events_dir, exist_ok=True)

    for (cue_sample, eid) in cues:
        class_name = labels_dict[eid]

        start_sample = cue_sample
        end_sample = min(cue_sample + label_duration_samples, data.shape[1])

        X_event = data[:, start_sample:end_sample]

        start_time = start_sample / sfreq
        end_time = end_sample / sfreq

        safe_class = re.sub(r'[\\/:*?"<>|]', "_", class_name)
        out_name = f"{safe_class}[{start_time:.3f}]{{{end_time:.3f}}}.npy"
        out_path = os.path.join(events_dir, out_name)

        if os.path.exists(out_path):
            print(f"[SKIP] Event ya existe: {out_path}")
            continue

        np.save(out_path, X_event)
        print(f"[Events] Guardado {out_path} | clase={class_name}")

    # Acumular metadata
    ch_names = list(raw_data.info["ch_names"])
    ch_types = raw_data.get_channel_types()
    ch_types_count = dict(Counter(ch_types))
    duration_sec = float(data.shape[1] / sfreq)

    unique_sfreqs.add(float(sfreq))
    union_channels.update(ch_names)
    ch_types_total.update(ch_types_count)
    total_duration_sec += duration_sec
```

---

## Funciones auxiliares necesarias (PASO 4)

### `_detect_dataset_type()` - Nueva

```python
def _detect_dataset_type(events):
    """Detecta si es Inner Speech o genérico basándose en event IDs"""
    event_ids = set(events[:, 2])

    # Inner Speech: tiene IDs 15, 22, 31-34
    inner_speech_markers = {15, 22, 31, 32, 33, 34}
    if inner_speech_markers.issubset(event_ids):
        return "inner_speech"

    return "generic"
```

### `_generic_event_extraction()` - Nueva

```python
def _generic_event_extraction(events, event_id_mapping=None):
    """Extrae eventos de datasets genéricos (todos los IDs únicos)"""
    # Limpiar espurios
    events = events[events[:, 2] != 65536]

    # Obtener IDs únicos
    unique_ids = sorted(set(events[:, 2]))

    # Si no hay mapping, crear genérico: Evento_1, Evento_2, etc.
    if not event_id_mapping:
        event_id_mapping = {eid: f"Evento_{eid}" for eid in unique_ids}

    # Extraer todos los eventos (sin filtrar por runs como Inner Speech)
    cues = [(int(sample), int(eid)) for sample, _, eid in events if eid in unique_ids]

    return cues, event_id_mapping
```

---

## Estructura generada (igual para BrainVision)

```
Aux/{dataset_name}/
├── dataset_metadata.json
└── sub-XX/ses-XX/eeg/
    ├── archivo.npy              ← Raw (channels × time)
    ├── Labels/
    │   └── archivo.npy          ← Labels (1 × time)
    └── Events/
        ├── Evento_1[123.4]{125.6}.npy
        ├── Evento_2[234.5]{236.7}.npy
        └── ...
```

---

## Testing

1. Coloca archivos BrainVision en `Data/mi_dataset/`:
   ```
   Data/mi_dataset/
   └── recording_001.vhdr
       recording_001.eeg
       recording_001.vmrk
   ```

2. En la app:
   - Click "Cargar Dataset"
   - Selecciona `mi_dataset`
   - Procesa

3. Verifica que se generaron:
   - `Aux/mi_dataset/recording_001.npy`
   - `Aux/mi_dataset/Labels/recording_001.npy`
   - `Aux/mi_dataset/Events/Evento_X[...].npy`
   - `Aux/mi_dataset/dataset_metadata.json`

---

## Resumen

✅ MNE soporta BrainVision nativamente
✅ Solo necesitas:
  1. Agregar `.vhdr` a `extensions_enabled`
  2. Crear método `read_brainvision()`
  3. Agregar bloque de procesamiento en `upload_dataset()`
  4. Implementar funciones PASO 4 (detectar tipo, extracción genérica)

✅ Genera exactamente los mismos 4 archivos que BDF:
  - Raw.npy
  - Labels.npy
  - Events/*.npy
  - metadata.json
