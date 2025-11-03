# Sistema Completo de Filtros y Transformadas

**Fecha de Finalización**: 2025-11-02
**Estado**: ✅ COMPLETAMENTE FUNCIONAL

---

## Resumen Ejecutivo

El sistema de filtros y transformadas de EGG-Lab está completamente implementado y funcional. Permite a los investigadores aplicar filtros de procesamiento de señales y transformadas sobre datos EEG de manera interactiva, con visualización en tiempo real de resultados.

---

## Páginas Implementadas

### 1. Página de Filtros (`/filtros`)

**Propósito**: Aplicar filtros de procesamiento de señales EEG y comparar señal original vs filtrada.

**Filtros disponibles**:
- ✅ **ICA** - Independent Component Analysis
- ✅ **WaveletsBase** - Denoising por wavelets
- ✅ **BandPass** - Filtros paso banda/alto/bajo (con auto-ajuste de filter_length)
- ✅ **Notch** - Filtro notch para eliminar ruido de línea eléctrica (50/60 Hz)

**Características**:
- Vista de dos columnas (Original azul vs Filtrada morada oscurecida)
- Navegación de canales (8 canales por página)
- Filtrado por clase de eventos (abajo, arriba, derecha, izquierda, Todas)
- Selección de canales específicos con checklist scrollable
- Botones de ayuda: "Todos", "Limpiar", "Solo EEG"
- Guardado automático en `Events/filtered/{evento}_{tipo}_{id}.npy`
- Registro en experimento actual

### 2. Página de Transformadas (`/extractores`)

**Propósito**: Aplicar transformadas de extracción de características sobre señales EEG.

**Transformadas disponibles**:
- ✅ **WaveletTransform** - Transformada wavelet con 60+ opciones de wavelets
- ✅ **FFTTransform** - Fast Fourier Transform con ventanas configurables
- ✅ **DCTTransform** - Discrete Cosine Transform con normalización opcional
- ✅ **WindowingTransform** - Ventaneo de señales con diferentes configuraciones

**Características**:
- Vista de dos columnas (Original vs Transformada con color oscurecido)
- Mismo sistema de navegación y filtrado que filtros
- Generación automática de etiquetas para eventos individuales
- Manejo de arrays 3D (concatenación de frames para visualización)
- Guardado en `Events/transformed/{evento}_{tipo}_{id}.npy`
- Guardado de etiquetas en `Events/transformed_labels/`
- Registro en experimento actual

---

## Arquitectura del Sistema

### Componente RightColumn

**Archivo**: `src/app/components/RigthComlumn.py`

**Función principal**: Generación dinámica de formularios desde schemas Pydantic.

**Tipos de ventanas soportadas**:
- `"filter"` → Filtros de señales
- `"featureExtracture"` → Transformadas/extractores
- `"clasificationModelsP300"` → Modelos para paradigma P300
- `"clasificationModelsInner"` → Modelos para paradigma Inner Speech

**Detección automática de tipos de campos**:

1. **Enums/Literals** → Dropdowns
   ```python
   method: Literal["fastica", "infomax", "picard"]
   # Genera: Dropdown con 3 opciones
   ```

2. **Union types** → Inputs numéricos/texto con validación
   ```python
   freq: Union[float, Tuple[float, float]]
   # Genera: Input text con placeholder "Ej: 30 o 1,30"
   ```

3. **Arrays** → Input text separado por comas
   ```python
   freqs: List[float]
   # Genera: Input "8, 12, 30, 45"
   ```

4. **Optional[Literal[..., None]]** → Dropdown con "None" como string
   ```python
   norm: Optional[Literal["ortho", None]]
   # Genera: Dropdown ["ortho", "None"]
   # Callback convierte "None" → Python None
   ```

**Traducción automática a español**: `NOMBRE_CAMPOS_ES` (30+ traducciones)

### FilterSchemaFactory

**Archivo**: `src/backend/classes/Filter/FilterSchemaFactory.py`

**Funciones principales**:
- `get_all_filter_schemas()` - Obtiene schemas JSON de todos los filtros
- `filterCallbackRegister(boton_id, inputs_map)` - Registra callbacks dinámicamente

**Flujo de aplicación de filtros**:
```
Usuario llena formulario
    ↓
Clic en "Aplicar" (btn-aplicar-{FilterName})
    ↓
Callback valida con Pydantic
    ↓
Genera ID autoincremental
    ↓
Aplica: Filter.apply(instance, file_path, directory_path_out)
    ↓
Guarda: Events/filtered/{evento}_{sufijo}_{id}.npy
    ↓
Actualiza: filtered-signal-store-filtros
    ↓
UI renderiza columna derecha automáticamente
```

**Preprocesamiento de inputs**:
- Conversión de strings con comas a arrays: `"1,30"` → `[1.0, 30.0]`
- Conversión de string "None" a Python `None`
- Auto-población de frecuencia de muestreo (`sp`) desde signal_data

### TransformSchemaFactory

**Archivo**: `src/backend/classes/FeatureExtracture/TransformSchemaFactory.py`

**Funciones principales**:
- `get_all_transform_schemas()` - Obtiene schemas JSON de todas las transformadas
- `TransformCallbackRegister(boton_id, inputs_map)` - Registra callbacks dinámicamente

**Flujo de aplicación de transformadas**:
```
Usuario llena formulario
    ↓
Clic en "Aplicar" (btn-aplicar-{TransformName})
    ↓
Callback valida con Pydantic
    ↓
Obtiene path del evento desde signal-store-extractores
    ↓
Genera etiquetas temporales (extrae clase del nombre del archivo)
    ↓
Aplica: Transform.apply(instance, file_path_in, directory_path_out, labels_directory, labels_out_path)
    ↓
Guarda: Events/transformed/{evento}_{sufijo}_{id}.npy
         Events/transformed_labels/{evento}_{sufijo}_{id}_labels.npy
    ↓
Carga datos transformados (manejo de arrays 3D → 2D)
    ↓
Actualiza: transformed-signal-store-extractores
    ↓
UI renderiza columna derecha automáticamente
```

**Características especiales**:
- Generación automática de etiquetas temporales para eventos individuales
- Manejo de arrays 3D: `(n_frames, frame_size, n_channels)` → `(n_channels, n_frames * frame_size)`
- Limpieza de archivos temporales después de aplicar
- Sistema de colores dinámicos con `get_class_color()`

---

## Sistema de Colores Dinámicos

**Archivo**: `src/shared/class_colors.py`

**Colores predefinidos por clase** (formato HSL):
```python
CLASS_COLORS = {
    "abajo": "hsl(0, 75%, 55%)",      # Rojo vibrante
    "arriba": "hsl(120, 70%, 50%)",   # Verde brillante
    "derecha": "hsl(210, 75%, 55%)",  # Azul cielo
    "izquierda": "hsl(45, 85%, 55%)", # Amarillo/dorado
    "target": "hsl(270, 70%, 55%)",
    "non-target": "hsl(180, 65%, 50%)",
}
```

**Función principal**:
```python
get_class_color(class_name: str, index: int = 0) -> str
```
- Retorna color HSL para una clase
- Fallback: Genera color consistente basado en hash del nombre
- Normalización automática: espacios → guiones bajos, lowercase

**Integración en plots**:
- **Columna izquierda (Original)**: Color brillante de la clase
- **Columna derecha (Procesada)**: Color oscurecido con `darkenHSL(classColor, 20)`
- **Títulos**: Borde superior con color de clase

**Función JavaScript `darkenHSL()`**:
```javascript
function darkenHSL(hslColor, amount = 20) {
  const match = hslColor.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
  const h = parseInt(match[1]);
  const s = parseInt(match[2]);
  const l = parseInt(match[3]);
  const newL = Math.max(0, l - amount);
  return `hsl(${h}, ${s}%, ${newL}%)`;
}
```

---

## Flujo de Datos Completo

### Filtros (filtros.py)

```
1. Usuario selecciona dataset → Metadata cargada
    ↓
2. Usuario selecciona archivo → Dataset.get_events_by_class()
    ↓
3. Usuario aplica filtro → filterCallbackRegister()
    ↓
4. Validación Pydantic → Filter.apply()
    ↓
5. Guardado en Events/filtered/
    ↓
6. Actualización de filtered-signal-store-filtros
    ↓
7. Clientside callback renderiza columna derecha
    ↓
8. Visualización con color oscurecido
```

### Transformadas (extractores.py)

```
1. Usuario selecciona dataset → Metadata cargada
    ↓
2. Usuario selecciona archivo → Dataset.get_events_by_class()
    ↓
3. Usuario aplica transformada → TransformCallbackRegister()
    ↓
4. Generación de etiquetas temporales
    ↓
5. Validación Pydantic → Transform.apply()
    ↓
6. Guardado en Events/transformed/ y Events/transformed_labels/
    ↓
7. Manejo de arrays 3D → 2D (si aplica)
    ↓
8. Actualización de transformed-signal-store-extractores
    ↓
9. Clientside callback renderiza columna derecha
    ↓
10. Visualización con color oscurecido
```

---

## Stores de Dash

### Filtros

```python
EVENTS_STORE_ID = "events-store-filtros"
DATA_STORE_ID = "signal-store-filtros"
FILTERED_DATA_STORE_ID = "filtered-signal-store-filtros"
CHANNEL_RANGE_STORE = "channel-range-store-filtros"
SELECTED_CLASS_STORE = "selected-class-store-filtros"
SELECTED_CHANNELS_STORE = "selected-channels-store-filtros"
```

### Transformadas

```python
EVENTS_STORE_ID = "events-store-extractores"
DATA_STORE_ID = "signal-store-extractores"
TRANSFORMED_DATA_STORE_ID = "transformed-signal-store-extractores"
CHANNEL_RANGE_STORE = "channel-range-store-extractores"
SELECTED_CLASS_STORE = "selected-class-store-extractores"
SELECTED_CHANNELS_STORE = "selected-channels-store-extractores"
```

---

## Características Avanzadas

### 1. Navegación de Canales

**Modos de visualización**:

1. **Paginación** (sin selección específica):
   - 8 canales por página
   - Botones "← Anteriores" / "Siguientes →"
   - Texto informativo: "Canales 0 - 7 de 137"

2. **Canales específicos** (con selección):
   - Checklist scrollable con todos los canales
   - Botones de ayuda: "Todos", "Limpiar", "Solo EEG"
   - Contador: "128 canales seleccionados"
   - Deshabilita navegación por páginas

**Callbacks de navegación**:
- `populate_channel_checklist()` - Llena checklist con nombres de canales
- `save_selected_channels()` - Guarda selección en store
- `update_channel_count()` - Actualiza contador
- `handle_channel_buttons()` - Maneja botones de ayuda

### 2. Filtrado por Clase

**Funcionamiento**:
- Botones por clase: "abajo", "arriba", "derecha", "izquierda"
- Botón "Todas" para mostrar eventos sin filtrar
- Selección única (un botón activo a la vez)
- Backend: `Dataset.get_events_by_class(path, class_name)`
- Callback: `select_specific_class()` y `select_all_classes()`

**Flujo**:
```
Usuario selecciona archivo → Determina sesión
    ↓
Usuario hace clic en "derecha"
    ↓
SELECTED_CLASS_STORE = "derecha"
    ↓
Callback se dispara con selected_class="derecha"
    ↓
Dataset.get_events_by_class(path, "derecha")
    ↓
Retorna primer evento de clase "derecha" en esa sesión
    ↓
Carga y muestra ese evento
```

### 3. Generación Automática de Etiquetas

**Problema**: Las transformadas requieren archivos de etiquetas, pero los eventos individuales en `Events/` no tienen etiquetas separadas (la clase está en el nombre del archivo).

**Solución** (`TransformSchemaFactory.py:232-251`):

```python
# Extraer clase del nombre del archivo
# "abajo[439.357]{441.908}.npy" → "abajo"
file_name = p_in.stem
event_class = file_name.split('[')[0].strip()

# Crear directorio temporal para etiquetas
labels_dir = p_in.parent / "temp_labels"
labels_dir.mkdir(parents=True, exist_ok=True)

# Generar array de etiquetas (todas con la misma clase)
arr_signal = np.load(str(p_in), allow_pickle=False)
n_samples = arr_signal.shape[1] if arr_signal.ndim == 2 else arr_signal.shape[0]
labels_array = np.array([event_class] * n_samples, dtype=str)

# Guardar etiquetas temporales
temp_labels_file = labels_dir / p_in.name
np.save(str(temp_labels_file), labels_array)
```

**Limpieza automática** (líneas 266-270):
```python
# Limpiar archivo temporal después de aplicar
if temp_labels_file.exists():
    temp_labels_file.unlink()
```

### 4. Manejo de Arrays 3D

**Problema**: Las transformadas ventaneadas generan arrays 3D `(n_frames, frame_size, n_channels)`, pero la visualización espera 2D `(n_channels, n_times)`.

**Solución** (`TransformSchemaFactory.py:299-310`):

```python
if arr.ndim == 3:
    # Formato: (n_frames, frame_size, n_channels)
    # Objetivo: (n_channels, n_frames * frame_size)
    n_frames, frame_size, n_channels = arr.shape

    # Paso 1: Transponer → (n_channels, n_frames, frame_size)
    arr_transposed = arr.transpose(2, 0, 1)

    # Paso 2: Concatenar frames → (n_channels, n_frames * frame_size)
    arr = arr_transposed.reshape(n_channels, n_frames * frame_size)

    print(f"Array 3D concatenado: {arr.shape} (canales x tiempo)")
```

---

## Mejoras Implementadas

### 1. Validaciones Context-Aware y Feedback Visual (2025-11-02) ✨

**Problema reportado**:
- Dropdown de wavelets mostraba 60+ opciones para filtros, pero `WaveletsBase` solo acepta 16 específicos
- Errores de validación solo aparecían en consola, usuario no tenía feedback visual
- Usuario seleccionaba opciones inválidas sin saber por qué fallaba

**Solución implementada**:

#### A. Dropdown Diferenciado para Wavelets

**Detección automática del contexto** (`RigthComlumn.py:82-90`):
```python
# Detectar si es WaveletsBase (filtro) o WaveletTransform (transformada)
is_filter = "WaveletsBase" in type

if is_filter:
    # WaveletsBase: Solo 16 wavelets válidos según el Literal del modelo
    valid_wavelets = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db8',
                    'sym2', 'sym3', 'sym4', 'sym5',
                    'coif1', 'coif2', 'coif3', 'coif5', 'haar']
    dropdown_options = [{"label": w, "value": w} for w in valid_wavelets]
else:
    # WaveletTransform: Catálogo completo de 60+ wavelets
    wavelet_families = {
        "Daubechies": [f"db{i}" for i in range(1, 39)],
        "Symlets": [f"sym{i}" for i in range(2, 21)],
        # ... más familias
    }
```

**Beneficios**:
- ✅ **Filtros** (`WaveletsBase`): Solo muestra las 16 opciones válidas
- ✅ **Transformadas** (`WaveletTransform`): Muestra catálogo completo (60+)
- ✅ Usuario no puede seleccionar opciones inválidas
- ✅ Previene errores de validación antes de enviar

**Wavelets válidos para filtros**:
| Familia | Wavelets |
|---------|----------|
| Daubechies | db1, db2, db3, db4, db5, db6, db8 |
| Symlets | sym2, sym3, sym4, sym5 |
| Coiflets | coif1, coif2, coif3, coif5 |
| Haar | haar |

#### B. Feedback Visual de Errores en Botones

**Mensajes de error visibles** (implementado en ambas Factories):

**1. FilterSchemaFactory.py (líneas 224-242)**:
```python
except ValidationError as e:
    errores = e.errors()
    # Construir mensaje de error legible
    error_fields = [err['loc'][0] for err in errores if err['loc']]
    msg_short = f"❌ Error: {', '.join(error_fields)}"
    # Retornar mensaje en el botón
    return msg_short, no_update

except ValueError as e:
    error_msg = f"❌ Error: {str(e)}"
    return error_msg, no_update

except Exception as e:
    return f"❌ Error inesperado", no_update
```

**2. TransformSchemaFactory.py (líneas 390-403)** - Misma implementación

**Tipos de mensajes de error**:

| Tipo de Error | Mensaje en Botón | Cuándo Ocurre |
|---------------|------------------|---------------|
| ValidationError | `❌ Error: wavelet, threshold` | Campos con valores inválidos |
| ValueError | `❌ Error: Nivel inválido: 6. Permitido hasta 5` | Validación de backend |
| Sin señal cargada | `❌ No hay señal cargada` | Usuario no ha cargado evento |
| Archivo no encontrado | `❌ Archivo no encontrado` | Error en procesamiento |
| Error inesperado | `❌ Error inesperado` | Excepciones no manejadas |

**Ejemplo de flujo con error**:
```
Usuario selecciona wavelet="rbio3.1" en filtro
    ↓
Hace clic en "Aplicar"
    ↓
Backend valida con Pydantic
    ↓
ValidationError: wavelet debe ser uno de [db1, db2, ..., haar]
    ↓
Botón muestra: "❌ Error: wavelet"
    ↓
Usuario ve error inmediatamente sin revisar consola
```

**Ejemplo de flujo exitoso**:
```
Usuario selecciona wavelet="db4" en filtro
    ↓
Hace clic en "Aplicar"
    ↓
Backend valida con Pydantic ✅
    ↓
Filtro se aplica correctamente
    ↓
Botón mantiene texto: "Aplicar"
    ↓
Columna derecha muestra señal filtrada
```

**Beneficios**:
- ✅ Usuario ve errores inmediatamente en la UI
- ✅ Mensajes concisos y accionables
- ✅ No necesita revisar consola
- ✅ Experiencia de usuario mejorada significativamente
- ✅ Mensajes en español (formato consistente con la app)

**Archivos modificados**:
- `RigthComlumn.py` (líneas 76-122): Dropdown context-aware
- `FilterSchemaFactory.py` (líneas 140-142, 189-201, 223-242): Feedback visual
- `TransformSchemaFactory.py` (líneas 164-166, 177-179, 283-285, 321-324, 390-403): Feedback visual

### 2. Auto-ajuste de filter_length en BandPass (2025-10-28)

**Problema**: FIR filters fallan con `filter_length` muy corto.

**Solución** (`BandPass.py:104-146`):

```python
if instance.method == "fir":
    if instance.order is not None:
        filter_length = instance.order
        if filter_length % 2 == 0:
            filter_length += 1  # MNE requiere impar
    else:
        filter_length = "auto"

    try:
        out = mne.filter.filter_data(..., filter_length=filter_length)
    except ValueError as e:
        if "too short" in str(e) and filter_length != "auto":
            # Fallback automático
            out = mne.filter.filter_data(..., filter_length="auto")
        else:
            raise
```

### 3. Fix para Dropdowns con valores `None` (2025-11-02)

**Problema**: Campos tipo `Optional[Literal["ortho", None]]` generaban error en Dash Dropdown.

**Causa**: Dash Dropdown no acepta `null` como valor válido.

**Solución**:

1. **RightColumn** (líneas 99-111):
   ```python
   dropdown_options = []
   has_none = False
   for val in enum_values:
       if val is None:
           has_none = True
       else:
           dropdown_options.append({"label": str(val), "value": val})

   if has_none:
       dropdown_options.append({"label": "None", "value": "None"})
   ```

2. **Callbacks** (FilterSchemaFactory y TransformSchemaFactory):
   ```python
   if isinstance(value, str) and value == "None":
       datos[field] = None
   ```

### 4. Preprocesamiento de Union Types

**Problema**: Campos como `freq` en BandPass pueden ser `float` O `Tuple[float, float]`.

**Solución** (ambas Factory):

```python
if isinstance(value, str) and "," in value:
    try:
        valores_separados = [float(v.strip()) for v in value.split(",")]
        datos[field] = valores_separados
    except (ValueError, AttributeError):
        datos[field] = value
```

**UI**:
```python
if has_number and has_array:
    inputType = "text"
    placeholder = "Ej: 30 (un valor) o 1,30 (dos valores separados por coma)"
```

### 5. Frecuencia de Muestreo Automática

**Problema**: El campo `sp` es requerido pero el usuario no debería ingresarlo manualmente.

**Solución** (ambas Factory):

```python
# Obtener sp del signal_data si no viene del formulario
if "sp" not in datos or datos.get("sp") is None:
    sfreq = signal_data.get("sfreq", 1024.0)
    datos["sp"] = float(sfreq)
    print(f"📊 Usando frecuencia de muestreo: {sfreq} Hz")
```

---

## Convenciones de Código

### Nomenclatura de IDs

**Formularios**:
```python
id=f"{type}-{field_name}"
# Ejemplos:
# - "ICA-sp"
# - "BandPass-freq"
# - "WaveletTransform-wavelet"
```

**Botones**:
```python
id=f"btn-aplicar-{type}"
# Ejemplos:
# - "btn-aplicar-ICA"
# - "btn-aplicar-WaveletTransform"
```

### Nomenclatura de Archivos Procesados

**Filtros**:
```
{evento}_{sufijo}_{id}.npy

Ejemplos:
- abajo[439.357]{441.908}_ica_0.npy
- abajo[439.357]{441.908}_bandpass_1.npy
- abajo[439.357]{441.908}_wav_2.npy
- abajo[439.357]{441.908}_notch_3.npy
```

**Transformadas**:
```
{evento}_{sufijo}_{id}.npy
{evento}_{sufijo}_{id}_labels.npy

Ejemplos:
- abajo[439.357]{441.908}_wavelet_0.npy
- abajo[439.357]{441.908}_wavelet_0_labels.npy
- abajo[439.357]{441.908}_fft_1.npy
- abajo[439.357]{441.908}_fft_1_labels.npy
```

### Mapeo de Sufijos

**Filtros**:
```python
filter_suffixes = {
    'ICA': 'ica',
    'WaveletsBase': 'wav',
    'BandPass': 'bandpass',
    'Notch': 'notch'
}
```

**Transformadas**:
```python
transform_suffixes = {
    "WaveletTransform": "wavelet",
    "FFTTransform": "fft",
    "DCTTransform": "dct",
    "WindowingTransform": "window"
}
```

---

## Testing y Validación

### Tests Completados ✅

**Filtros**:
- ✅ Aplicar ICA con canales específicos
- ✅ Aplicar Wavelets con diferentes wavelets
- ✅ Aplicar BandPass con auto-ajuste de filter_length
- ✅ Aplicar Notch en 50 Hz y 60 Hz
- ✅ Filtrado por clase funciona correctamente
- ✅ Navegación de canales (paginación y selección específica)
- ✅ Visualización en columna derecha con color oscurecido

**Transformadas**:
- ✅ Aplicar WaveletTransform con dropdowns de wavelets
- ✅ Aplicar FFTTransform con ventanas configurables
- ✅ Aplicar DCTTransform con normalización "ortho" y "None"
- ✅ Generación automática de etiquetas temporales
- ✅ Manejo de arrays 3D → 2D
- ✅ Limpieza de archivos temporales
- ✅ Visualización en columna derecha con color oscurecido

**RightColumn**:
- ✅ Dropdowns con valores `None` funcionan correctamente
- ✅ Conversión automática "None" (string) → None (Python)
- ✅ Preprocesamiento de arrays desde strings con comas
- ✅ Traducción a español de campos técnicos
- ✅ Validación con Pydantic de todos los campos
- ✅ Dropdown context-aware para wavelets (16 opciones en filtros, 60+ en transformadas)
- ✅ Feedback visual de errores en botones
- ✅ Mensajes de error legibles y accionables

---

## Documentación Actualizada

Los siguientes archivos de documentación han sido actualizados:

1. **`.claude/FILTROS_Y_TRANSFORMADAS_SISTEMA_COMPLETO.md`** (este archivo - Última actualización: 2025-11-02)
   - Nueva sección: "Validaciones Context-Aware y Feedback Visual"
   - Documentación completa del dropdown diferenciado de wavelets
   - Sistema de feedback visual de errores en botones
   - Tabla de tipos de errores y mensajes
   - Ejemplos de flujos con y sin errores
   - Tests actualizados con nuevas características

2. **`.claude/components/RightColumn.md`** (Última actualización: 2025-11-02)
   - Nueva sección: "Dropdown Context-Aware para Wavelets"
   - Sección "Cambios Recientes" con fix de Dropdowns `None`
   - Flujo completo de conversión `None` → "None" → `None`
   - Documentación de detección automática filtro vs transformada

3. **`.claude/context.md`**
   - Nueva sección "Sistema Completo de Filtros y Transformadas"
   - Lista de características funcionando
   - Estado actual: ✅ COMPLETAMENTE FUNCIONAL

---

## Próximos Pasos Sugeridos

### Integración con Modelos de Clasificación

Ahora que filtros y transformadas están completos, el siguiente paso natural es integrarlos con los modelos de clasificación:

1. **Pipeline completo**: Dataset → Filtros → Transformadas → Modelo → Evaluación
2. **Configuración de experimentos**: Guardar configuraciones completas de preprocesamiento
3. **Comparación de configuraciones**: Comparar diferentes combinaciones de filtros/transformadas
4. **Optimización de hiperparámetros**: Búsqueda automática de mejores configuraciones

### Mejoras de UX

1. **Tooltips con descripciones**: Mostrar ayuda contextual en campos del formulario
2. **Validación en frontend**: Validar rangos y tipos antes de enviar al backend
3. **Componente especial para arrays**: Editor visual para listas de valores
4. **Guardar/cargar configuraciones**: Presets de filtros/transformadas

### Performance

1. **Caching de resultados**: Evitar recomputar transformadas iguales
2. **Procesamiento en batch**: Aplicar filtros/transformadas a múltiples eventos
3. **Optimización de visualización**: Mejoras en renderizado de plots

---

## Conclusión

El sistema de filtros y transformadas de EGG-Lab está **completamente funcional** y listo para ser usado en investigación. Todas las características clave han sido implementadas, probadas y documentadas. El sistema es robusto, extensible y mantiene una arquitectura limpia basada en generación dinámica desde schemas Pydantic.

**Estado final**: ✅ PRODUCCIÓN
