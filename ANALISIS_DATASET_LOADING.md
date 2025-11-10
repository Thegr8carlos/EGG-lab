# Análisis: Dataset Loading (Carga de Datos)

## ❌ Problema Actual: HARDCODEADO para Inner Speech

### Código problemático (`dataset.py`)

**Línea 22-23**: Labels hardcodeados
```python
LABELS = {31: "arriba", 32: "abajo", 33: "derecha", 34: "izquierda"}
CUE_IDS = set(LABELS.keys())
```

**Línea 26-45**: Función `_inner_speech_cues()` hardcodeada
```python
def _inner_speech_cues(events):
    for sample, _, eid in events:
        if eid == 15:  # start of run ← HARDCODED
            in_inner_run = False
        elif eid == 22:  # inner speech run ← HARDCODED
            in_inner_run = True
        elif eid == 16:  # end of run ← HARDCODED
            in_inner_run = False
        elif in_inner_run and eid in CUE_IDS:  # ← Solo {31,32,33,34}
            cues.append((int(sample), int(eid)))
```

**Línea 378**: `upload_dataset()` usa la función hardcodeada
```python
inner_cues = _inner_speech_cues(events)  # Solo funciona con Inner Speech
```

---

## 🔥 Qué pasa si subes OTRO dataset BDF

**Escenario**: Subes un dataset P300, Motor Imagery, o cualquier otro

**Resultado**:
1. ❌ `_inner_speech_cues()` NO encuentra eventos (busca IDs 15, 22, 16, 31-34)
2. ❌ `inner_cues` = lista vacía
3. ❌ NO se generan archivos en `Events/`
4. ❌ `Labels/` tiene todo ceros (no hay eventos detectados)
5. ❌ La app NO puede cargar eventos para visualizar
6. ❌ Filtros y transformadas NO funcionan (no hay datos)

---

## 📂 Archivos que DEBE generar para que funcione

### Estructura requerida:
```
Aux/{dataset_name}/
├── dataset_metadata.json         ← Metadata global
└── sub-XX/ses-XX/eeg/
    ├── archivo.npy                ← Señal raw completa (channels × time)
    ├── Labels/
    │   └── archivo.npy            ← Labels por muestra (1 × time)
    └── Events/
        ├── clase1[123.4]{125.6}.npy   ← Evento segmentado
        ├── clase2[234.5]{236.7}.npy
        └── ...
```

### Archivo `dataset_metadata.json`:
```json
{
  "name": "nieto_inner_speech",
  "classes": ["arriba", "abajo", "derecha", "izquierda"],
  "sampling_frequency_hz": 1024.0,
  "channel_names": ["Fp1", "Fp2", "A1", ...],
  "n_events_total": 1280,
  "class_distribution": {
    "arriba": 320,
    "abajo": 320,
    "derecha": 320,
    "izquierda": 320
  }
}
```

---

## ✅ Solución: Hacerlo Genérico

### Cambios necesarios en `dataset.py`:

**1. Detectar tipo de dataset**
```python
def _detect_dataset_type(events):
    """Detecta si es Inner Speech o genérico"""
    event_ids = set(events[:, 2])

    # Inner Speech: tiene event IDs 15, 22, 31-34
    if {15, 22, 31, 32, 33, 34}.issubset(event_ids):
        return "inner_speech"

    return "generic"
```

**2. Extraer eventos genéricos**
```python
def _generic_event_extraction(events, event_id_mapping=None):
    """Extrae TODOS los event IDs únicos, sin filtrar"""
    # Eliminar espurios
    events = events[events[:, 2] != 65536]

    # Obtener IDs únicos
    unique_ids = sorted(set(events[:, 2]))

    # Si no hay mapping, crear genérico
    if not event_id_mapping:
        event_id_mapping = {eid: f"Evento_{eid}" for eid in unique_ids}

    cues = [(int(sample), int(eid)) for sample, _, eid in events if eid in unique_ids]
    return cues, event_id_mapping
```

**3. Modificar `upload_dataset()`**
```python
def upload_dataset(self, path_to_folder, event_id_mapping=None):
    # ...

    for file in files:
        if ext == ".bdf":
            raw_data = self.read_bdf(str(file))
            events = mne.find_events(raw_data, stim_channel="Status")

            # Detectar tipo de dataset
            dataset_type = _detect_dataset_type(events)

            if dataset_type == "inner_speech":
                inner_cues = _inner_speech_cues(events)
                labels_dict = LABELS  # {31: "arriba", ...}
            else:
                inner_cues, labels_dict = _generic_event_extraction(events, event_id_mapping)

            # Resto del código usa inner_cues y labels_dict
```

**4. Agregar UI para configurar mapping**

En `cargar_datos.py`, después de detectar que es genérico, mostrar modal:
```
Detectado dataset genérico con Event IDs: [1, 2, 3, 4, 5]

Configura los nombres de clase:
Event ID 1: [Clase_A    ]
Event ID 2: [Clase_B    ]
Event ID 3: [Clase_C    ]
...

[Procesar Dataset]
```

---

## 🎯 Implementación Paso a Paso (PLAN_MEJORAS.md PASO 4)

Ya está en el plan, pero ahora entiendes el contexto:

### PASO 4.1: Detección de formato
- Agregar `_detect_dataset_type()`
- Detectar Inner Speech vs genérico

### PASO 4.2: Extracción genérica
- Agregar `_generic_event_extraction()`
- Usar TODOS los event IDs únicos

### PASO 4.3: Modificar upload_dataset
- Bifurcar lógica según tipo
- Generar archivos correctos para ambos

### PASO 4.4: UI de configuración
- Modal para mapear Event IDs → nombres de clase
- Guardar configuración en JSON

---

## 📋 Resumen Ejecutivo

**Actualmente**: Solo funciona con Inner Speech (event IDs hardcodeados)

**Si subes otro BDF**: NO genera Events/, Labels vacío, app NO funciona

**Archivos necesarios**:
- `archivo.npy` (raw)
- `Labels/archivo.npy` (labels)
- `Events/clase[t1]{t2}.npy` (eventos segmentados)
- `dataset_metadata.json` (metadata)

**Solución**: Detectar tipo, extraer eventos genéricos, permitir configurar mapping
