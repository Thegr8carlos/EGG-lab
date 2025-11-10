# 📊 Plan: Dashboard Científico para Página de Dataset

## 🎯 Objetivo
Convertir la página "Dataset" en un dashboard científico completo donde el investigador pueda explorar visualmente todas las estadísticas, calidad de señal, y características espaciotemporales del dataset cargado.

---

## 📸 Estado Actual
- ✅ Árbol de archivos (izquierda)
- ✅ Label Color Map (arriba)
- ✅ Gráfico lineal de señal raw (centro)

**Limitaciones:**
- Solo muestra señal raw sin contexto
- No hay estadísticas descriptivas
- No hay visualización espacial (topomaps)
- No hay análisis de calidad
- No hay métricas por sesión

---

## 🚀 Plan de Mejoras

### **FASE 1: Información Adicional a Calcular y Guardar en JSON**

Ampliar `dataset_metadata.json` con nueva información calculada durante `upload_dataset()`:

#### 1.1 **Estadísticas por Sesión** (nuevo objeto `sessions`)
```json
"sessions": [
  {
    "subject": "sub-01",
    "session": "ses-01",
    "file_path": "sub-01/ses-01/eeg/archivo.vhdr",
    "duration_sec": 520.5,
    "n_events": 100,
    "events_per_class": {
      "Down": 12,
      "Left": 13,
      "Rest": 50,
      ...
    },
    "sampling_rate": 250.0,
    "n_channels": 8,
    "bad_channels": []  // Canales con excesivo ruido (opcional)
  },
  ...
]
```

#### 1.2 **Estadísticas de Señal por Canal** (nuevo objeto `channel_stats`)
```json
"channel_stats": {
  "Fz": {
    "mean": 0.00012,
    "std": 0.00045,
    "min": -0.002,
    "max": 0.0018,
    "variance": 2.025e-07,
    "rms": 0.00046
  },
  "C3": { ... },
  ...
}
```

#### 1.3 **Información Espacial de Canales** (nuevo objeto `montage`)
```json
"montage": {
  "type": "standard_1020",  // o "biosemi128", "custom", etc.
  "positions": {
    "Fz": {"x": 0.0, "y": 0.8, "z": 0.6},
    "C3": {"x": -0.6, "y": 0.0, "z": 0.6},
    ...
  },
  "has_positions": true  // false si no se pueden inferir posiciones
}
```

#### 1.4 **Análisis de Frecuencias** (nuevo objeto `frequency_bands`)
Calcular potencia promedio en bandas clásicas:
```json
"frequency_bands": {
  "delta": {"range": [0.5, 4], "mean_power": 0.00012},
  "theta": {"range": [4, 8], "mean_power": 0.00008},
  "alpha": {"range": [8, 13], "mean_power": 0.00015},
  "beta": {"range": [13, 30], "mean_power": 0.00005},
  "gamma": {"range": [30, 100], "mean_power": 0.00002}
}
```

#### 1.5 **Calidad del Dataset** (nuevo objeto `quality_metrics`)
```json
"quality_metrics": {
  "total_bad_channels": 2,
  "bad_channels_list": ["A12", "B23"],
  "mean_snr_db": 12.5,  // Signal-to-noise ratio promedio
  "n_events_total": 18300,
  "events_balance": "balanced",  // "balanced" | "imbalanced"
  "class_imbalance_ratio": 1.2  // Max/min class count ratio
}
```

#### 1.6 **Metadata Adicional** (ya existente, ampliar)
```json
"created_at": "2025-01-09T17:45:00Z",
"format": "brainvision",  // "bdf", "edf", "brainvision"
"total_files": 186,
"total_size_mb": 1234.5
```

---

### **FASE 2: Diseño de la Nueva Página de Dataset**

Reorganizar la página en **secciones colapsables** estilo dashboard científico.

#### 2.1 **Layout Propuesto**

```
┌─────────────────────────────────────────────────────────────┐
│  📂 Dataset: arabic_inner_speech                            │
├─────────────┬───────────────────────────────────────────────┤
│             │  🔍 OVERVIEW (siempre visible)               │
│   Árbol     │  ├─ 8 clases, 186 archivos, 250 Hz           │
│   Archivos  │  ├─ 25h duración, 18,300 eventos             │
│   (sidebar) │  └─ Calidad: ⭐⭐⭐⭐☆ (Buena)                │
│             ├──────────────────────────────────────────────┤
│             │  📊 SECCIÓN 1: ESTADÍSTICAS GENERALES       │
│             │  [▼ Expandido / ▶ Colapsado]                │
│             │  ├─ Gráfico de barras: eventos por clase     │
│             │  ├─ Tabla: sesiones (sub, ses, duración)     │
│             │  └─ Tabla: frecuencias de muestreo           │
│             ├──────────────────────────────────────────────┤
│             │  🧠 SECCIÓN 2: VISUALIZACIÓN ESPACIAL       │
│             │  ├─ Topomap: ubicación de electrodos         │
│             │  └─ Dropdown: seleccionar montage            │
│             ├──────────────────────────────────────────────┤
│             │  📈 SECCIÓN 3: ANÁLISIS DE SEÑAL            │
│             │  ├─ Gráfico: señal raw (ya existe)           │
│             │  ├─ Dropdown: seleccionar sesión + canal     │
│             │  ├─ Estadísticas: media, std, rango          │
│             │  └─ PSD (Power Spectral Density)             │
│             ├──────────────────────────────────────────────┤
│             │  🔥 SECCIÓN 4: HEATMAP TEMPORAL             │
│             │  ├─ Heatmap: actividad por clase en cerebro  │
│             │  ├─ Slider: tiempo (para animación)          │
│             │  └─ Dropdown: seleccionar clase              │
│             ├──────────────────────────────────────────────┤
│             │  🌈 SECCIÓN 5: ANÁLISIS DE FRECUENCIAS      │
│             │  ├─ Gráfico: potencia por banda (delta-gamma)│
│             │  └─ Topomap: banda seleccionada              │
│             ├──────────────────────────────────────────────┤
│             │  ⚠️ SECCIÓN 6: CALIDAD DEL DATASET          │
│             │  ├─ Bad channels: lista + topomap            │
│             │  ├─ SNR promedio por canal                   │
│             │  └─ Balance de clases (gráfico)              │
│             ├──────────────────────────────────────────────┤
│             │  📊 SECCIÓN 7: CORRELACIONES                │
│             │  └─ Matriz de correlación entre canales      │
└─────────────┴──────────────────────────────────────────────┘
```

---

### **FASE 3: Implementación por Secciones**

#### **PASO 3.1: Ampliar `dataset.py` para calcular nueva metadata**

**Archivo:** `src/backend/classes/dataset.py`

**Funciones a agregar:**

1. **`_calculate_session_stats(raw, events, labels_dict, file_path)`**
   - Retorna dict con estadísticas de la sesión

2. **`_calculate_channel_stats(raw_data)`**
   - Calcula mean, std, min, max, variance, RMS por canal
   - Usa `np.mean()`, `np.std()`, etc.

3. **`_infer_montage(channel_names)`**
   - Usa `mne.channels.make_standard_montage()` para inferir posiciones
   - Maneja casos donde no se puede inferir (retorna `has_positions: false`)

4. **`_calculate_frequency_bands(raw_data, sfreq)`**
   - Calcula PSD con `mne.time_frequency.psd_array()`
   - Integra potencia en bandas delta, theta, alpha, beta, gamma

5. **`_detect_bad_channels(raw_data)`**
   - Usa `mne.preprocessing.find_bad_channels_maxwell()` (si hay info espacial)
   - O detecta canales con varianza extrema (threshold basado en std)

6. **`_calculate_snr(raw_data)`**
   - Signal-to-noise ratio promedio

**Modificar `upload_dataset()`:**
- Al final del loop de archivos, llamar a estas funciones
- Acumular resultados en listas/dicts
- Guardar en `dataset_metadata.json` al final

---

#### **PASO 3.2: Crear página de Dataset mejorada**

**Archivo:** `src/app/pages/dataset.py`

**Componentes a crear:**

1. **`create_overview_panel(metadata)`**
   - Muestra resumen: clases, archivos, duración, calidad

2. **`create_general_stats_section(metadata)`**
   - Gráfico de barras: eventos por clase
   - Tabla: sesiones (sub, ses, duración, eventos)

3. **`create_spatial_section(metadata)`**
   - Topomap con posiciones de electrodos
   - Usa `mne.viz.plot_topomap()` o equivalente en Plotly

4. **`create_signal_analysis_section(metadata, selected_session, selected_channel)`**
   - Gráfico de señal raw (ya existe)
   - PSD (Power Spectral Density)
   - Tabla de estadísticas

5. **`create_heatmap_section(metadata, selected_class, time_point)`**
   - Heatmap temporal en topomap
   - Carga evento promedio de una clase
   - Anima la actividad cerebral frame por frame

6. **`create_frequency_bands_section(metadata)`**
   - Gráfico de barras: potencia por banda
   - Topomap de banda seleccionada

7. **`create_quality_section(metadata)`**
   - Bad channels: lista + topomap marcado
   - SNR por canal (gráfico de barras)
   - Balance de clases (pie chart)

8. **`create_correlation_section(metadata, raw_data)`**
   - Matriz de correlación entre canales (heatmap)
   - Usa `np.corrcoef()`

---

#### **PASO 3.3: Implementar Topomaps interactivos**

**Opciones:**

1. **MNE + Matplotlib → imagen estática**
   - Generar imagen con `mne.viz.plot_topomap()`
   - Convertir a base64, mostrar en Dash

2. **Plotly (recomendado)**
   - Crear scatter plot con posiciones 2D de electrodos
   - Interpolar valores con `scipy.interpolate.griddata()`
   - Más interactivo, permite zoom, hover

**Función helper:**
```python
def plot_topomap_plotly(channel_positions, channel_values, title):
    """
    channel_positions: dict {ch_name: (x, y)}
    channel_values: dict {ch_name: value}
    """
    # Crear scatter plot con interpolación
    # Retornar fig de Plotly
```

---

#### **PASO 3.4: Implementar Heatmap Temporal (Feature estrella)**

**Cómo funciona:**

1. **Cargar evento promedio de una clase**
   - Ej: promedio de todos los eventos "Down"
   - Shape: (n_channels, n_timepoints)

2. **Por cada frame de tiempo:**
   - Extraer valores de todos los canales en ese tiempo
   - Plotear topomap con esos valores
   - Repetir para todos los frames → animación

3. **Controles:**
   - Slider: seleccionar tiempo (0 - duración del evento)
   - Dropdown: seleccionar clase
   - Button: "▶ Play" para animar

**Función a crear:**
```python
def create_temporal_heatmap(dataset_name, class_name, time_point):
    """
    1. Cargar eventos de clase_name
    2. Calcular promedio
    3. Extraer valores en time_point
    4. Generar topomap
    """
```

---

### **FASE 4: Priorización de Features**

#### **Prioridad ALTA (Must Have)**
1. ✅ **Estadísticas generales** (eventos por clase, sesiones, duración)
2. ✅ **Topomap de ubicación de electrodos** (estático)
3. ✅ **Análisis de señal básico** (gráfico raw + estadísticas)
4. ✅ **Calidad del dataset** (bad channels, balance de clases)

#### **Prioridad MEDIA (Should Have)**
5. ✅ **Heatmap temporal** (actividad cerebral por clase)
6. ✅ **PSD (Power Spectral Density)**
7. ✅ **Análisis de bandas de frecuencia**

#### **Prioridad BAJA (Nice to Have)**
8. ⚠️ **Correlaciones entre canales** (puede ser lento con muchos canales)
9. ⚠️ **Animación temporal** (play button en heatmap)
10. ⚠️ **SNR por canal** (requiere definir ruido base)

---

## 📋 Checklist de Implementación

### **Backend (`dataset.py`)**
- [ ] Agregar `_calculate_session_stats()`
- [ ] Agregar `_calculate_channel_stats()`
- [ ] Agregar `_infer_montage()`
- [ ] Agregar `_calculate_frequency_bands()`
- [ ] Agregar `_detect_bad_channels()`
- [ ] Modificar `upload_dataset()` para guardar nueva metadata
- [ ] Probar con nieto_inner_speech y arabic_inner_speech

### **Frontend (`pages/dataset.py`)**
- [ ] Crear layout con secciones colapsables
- [ ] Implementar `create_overview_panel()`
- [ ] Implementar `create_general_stats_section()`
- [ ] Implementar `create_spatial_section()` (topomap)
- [ ] Implementar `create_signal_analysis_section()`
- [ ] Implementar `create_quality_section()`
- [ ] Implementar `create_heatmap_section()` (temporal)
- [ ] Implementar `create_frequency_bands_section()`
- [ ] Agregar callbacks para interactividad (dropdowns, sliders)

### **Utils/Helpers**
- [ ] Crear `plot_topomap_plotly()` para topomaps interactivos
- [ ] Crear `load_average_event()` para cargar evento promedio de una clase
- [ ] Crear `interpolate_topomap_data()` para interpolación espacial

---

## 🎨 Mockups de Secciones Clave

### **Topomap de Electrodos**
```
     🧠
   O   O   O    ← Fz, FCz, Cz
 O   O   O   O  ← F3, FC3, C3, CP3
   O   O   O    ← P3, Pz, P4
     O   O      ← PO7, PO8
       O        ← Oz

- Círculo = electrodo
- Hover = nombre del canal
- Color = valor (si se mapea una métrica)
```

### **Heatmap Temporal**
```
[Clase: Down ▼]  [────────────●──────] 1.2s / 3.2s  [▶ Play]

        🧠 (topomap)
     🔴   🔵   🔴    ← Activación en tiempo t
   🟠   🔵   🔵   🟠
     🟡   🟡   🟡
       🟢   🟢

Colormap: azul (bajo) → rojo (alto)
```

---

## 🔧 Tecnologías a Usar

- **MNE-Python**: Cálculo de estadísticas, PSD, montage inference
- **Plotly**: Gráficos interactivos (topomaps, PSD, barras)
- **Dash Bootstrap Components**: Layout responsivo con `dbc.Accordion` para secciones colapsables
- **NumPy/SciPy**: Cálculos de estadísticas, interpolación
- **Dash Core Components**: Sliders, dropdowns, buttons

---

## 📊 Beneficios para el Investigador

1. **Exploración Rápida**: Ver resumen completo del dataset en un vistazo
2. **Calidad de Datos**: Detectar problemas (bad channels, desbalance)
3. **Insights Espaciales**: Ver qué regiones del cerebro están activas por clase
4. **Validación**: Verificar que frecuencias de muestreo, duración, etc. son correctas
5. **Reproducibilidad**: Toda la metadata guardada en JSON para papers

---

## 🚦 Siguiente Paso

**¿Por dónde empezar?**

1. **PASO 1**: Ampliar `dataset.py` para calcular metadata de sesiones y canales
2. **PASO 2**: Implementar topomap estático (ubicación de electrodos)
3. **PASO 3**: Implementar sección de estadísticas generales
4. **PASO 4**: Implementar heatmap temporal (feature estrella)
5. **PASO 5**: Análisis de frecuencias y calidad

---

## 💡 Ideas Adicionales (Futuro)

- **Exportar reporte PDF** con todas las estadísticas
- **Comparar múltiples datasets** (lado a lado)
- **Detección automática de artefactos** (parpadeos, movimientos)
- **Sugerir preprocesamiento** basado en calidad (ej: "Aplicar notch filter a 50Hz")
