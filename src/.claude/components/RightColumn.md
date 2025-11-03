# Componente: RightColumn

**Archivo**: `src/app/components/RigthComlumn.py`
**Última actualización**: 2025-11-02

---

## Propósito

Componente que genera dinámicamente paneles de configuración con formularios basados en schemas Pydantic. Usado para configurar filtros, transformadas y modelos de clasificación en las páginas de la aplicación.

---

## Función Principal

```python
def get_rightColumn(window: str) -> html.Div
```

### Parámetros
- **window** (str): Tipo de ventana que determina qué formularios generar
  - `"filter"` → Filtros de señales
  - `"featureExtracture"` → Transformadas/extractores
  - `"clasificationModelsP300"` → Modelos para paradigma P300
  - `"clasificationModelsInner"` → Modelos para paradigma Inner Speech

### Retorno
- **html.Div**: Contenedor con título + tarjetas de configuración

---

## Arquitectura

### 1. Obtención de Schemas

Por cada tipo de ventana, se obtienen los schemas de las clases Pydantic correspondientes:

```python
if window == "filter":
    all_schemas = FilterSchemaFactory.get_all_filter_schemas()
    title = "Filtros"
elif window == "featureExtracture":
    all_schemas = TransformSchemaFactory.get_all_transform_schemas()
    title = "Extractores de características"
elif window == "clasificationModelsP300":
    all_schemas = _tag_classifier_schemas("_p300")
    title = "Modelos de clasificación"
elif window == "clasificationModelsInner":
    all_schemas = _tag_classifier_schemas("_inner")
    title = "Modelos de clasificación"
```

**Nota**: Los modelos de clasificación se "etiquetan" con sufijos `_p300` o `_inner` para diferenciarlos en la UI y los callbacks.

### 2. Generación de Tarjetas

Para cada schema, se llama a `build_configuration_ui(schema)`:

```python
cards = []
for filter_type, schema in all_schemas.items():
    cards.append(build_configuration_ui(schema))
```

### 3. Layout Final

```python
html.Div([
    html.H2(title, className="right-panel-title"),
    *cards
], className="right-panel-container")
```

---

## Construcción de Formularios: `build_configuration_ui()`

### Flujo de Generación

```
Schema Pydantic
    ↓
JSON Schema (via model_json_schema())
    ↓
Analizar cada campo (properties)
    ↓
Detectar tipo del campo
    ↓
Generar componente Dash apropiado
    ↓
Agregar botón "Aplicar"
    ↓
Envolver en Card
```

### Tipos de Campos Soportados

#### 1. Enums / Literals
**Detección**:
```python
# Directo
if "enum" in field_info:
    enum_values = field_info["enum"]

# Desde anyOf (Literal de Pydantic)
elif "anyOf" in field_info:
    consts = [x.get("const") for x in field_info["anyOf"] if "const" in x]
    if consts:
        enum_values = [c for c in consts if c is not None]
```

**Componente generado**:
```python
dcc.Dropdown(
    id=f"{type}-{field_name}",
    options=[{"label": str(val), "value": val} for val in enum_values],
    placeholder=f"Selecciona valor",
    style={"flex": "1", "color": "black", "fontSize": "14px"}
)
```

**Ejemplo**: `phase` → Dropdown con opciones ["zero", "zero-double", "minimum"]

#### 2. anyOf (Union types)
**Detección**:
```python
elif "anyOf" in field_info:
    posibles_tipos = {x.get("type") for x in field_info["anyOf"] if "type" in x}
```

**Lógica**:
- Si uno de los tipos es `number` o `integer` → Input numérico
- Extrae `minimum`, `maximum`, `default` si existen
- Caso contrario → Input de texto

**Componente generado**:
```python
dbc.Input(
    type="number",  # o "text"
    id=f"{type}-{field_name}",
    placeholder=f"Ingresa {showName}",
    min=...,  # si aplica
    max=...,  # si aplica
    value=...,  # si hay default
    style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
)
```

**Ejemplo**: `freq` → Input numérico con min/max

#### 3. Arrays
**Detección**:
```python
elif field_info.get("type") == "array":
```

**Componente generado**:
```python
dbc.Input(
    type="text",
    id=f"{type}-{field_name}",
    placeholder=f"Ingresa lista separada por comas",
    style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
)
```

**Nota**: El usuario debe ingresar valores separados por comas. El callback debe parsear este string.

**Ejemplo**: `freqs` → "8, 12, 30, 45"

#### 4. Tipos Simples
**Detección**:
```python
else:
    tipo_dash = {
        "number": "number",
        "integer": "number",
        "string": "text",
    }.get(field_info.get("type", "string"), "text")
```

**Componente generado**:
```python
dbc.Input(
    type=tipo_dash,
    id=f"{type}-{field_name}",
    placeholder=f"Ingresa {showName}",
    style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
)
```

### Nomenclatura de IDs

```python
id=f"{type}-{field_name}"
```

**Ejemplos**:
- `ICA-sp` → Frecuencia de muestreo de ICA
- `BandPass-freq` → Frecuencia de corte de BandPass
- `WaveletsBase-wavelet` → Tipo de wavelet

**ID del botón**:
```python
id=f"btn-aplicar-{type}"
```

**Ejemplos**:
- `btn-aplicar-ICA`
- `btn-aplicar-BandPass`

---

## Traducción de Campos: `NOMBRE_CAMPOS_ES`

Diccionario que mapea nombres técnicos (nombres de atributos en Pydantic) a nombres legibles en español:

```python
NOMBRE_CAMPOS_ES = {
    # General
    "sp": "Frecuencia de Muestreo",
    "epochs": "Épocas",
    "batch_size": "Tamaño de lote",

    # Filters and Feature Extractors
    "numeroComponentes": "Número de componentes",
    "method": "Método",
    "random_state": "Semilla aleatoria",
    "max_iter": "Iteraciones máximas",
    "wavelet": "Tipo de Wavelet",
    "level": "Nivel de descomposición",
    "mode": "Modo de borde",
    "threshold": "Umbral",
    "filter_type": "Tipo de filtro",
    "freq": "Frecuencia de corte",
    "freqs": "Frecuencias",
    "order": "Orden del filtro",
    "phase": "Tipo de fase",
    "fir_window": "Ventana FIR",
    "quality": "Calidad",
    "window": "Ventana de análisis",
    "nfft": "Tamaño de FFT",
    "overlap": "Solapamiento",
    "type": "Tipo de DCT",
    "norm": "Normalización",
    "axis": "Eje",

    # Classifiers
    "hidden_size": "Tamaño oculto",
    "num_layers": "Número de capas",
    "bidirectional": "Bidireccional",
    "dropout": "Dropout",
    "learning_rate": "Tasa de aprendizaje",
    "kernel": "Kernel",
    "C": "Coeficiente C",
    "gamma": "Gamma",
    "n_estimators": "Número de árboles",
    "max_depth": "Profundidad máxima",
    "criterion": "Criterio",
    "num_filters": "Número de filtros",
    "kernel_size": "Tamaño del kernel",
    "pool_size": "Tamaño de pooling"
}
```

**Uso**:
```python
showName = NOMBRE_CAMPOS_ES.get(field_name, field_info.get("title", field_name))
```

**Fallback**: Si no hay traducción, usa `field_info.get("title")` o el nombre del campo directamente.

---

## Registro de Callbacks

Los callbacks se registran automáticamente al final del archivo `RigthComlumn.py`:

### Estructura General

```python
for grupo in generar_mapa_validacion_inputs(schemas):
    for boton_id, inputs_map in grupo.items():
        CallbackRegister(boton_id, inputs_map)
```

**Helper**: `generar_mapa_validacion_inputs()` (de `backend.helpers.mapaValidacion`)
- Lee todos los schemas
- Genera mapas de `{boton_id: {input_id: field_name}}`
- Retorna lista de diccionarios listos para registro

### Callbacks Registrados

#### 1. Transforms (Extractores)
```python
for grupo in generar_mapa_validacion_inputs(TransformSchemaFactory.get_all_transform_schemas()):
    for boton_id, inputs_map in grupo.items():
        TransformCallbackRegister(boton_id, inputs_map)
```

**Archivos afectados**: `extractores.py`

#### 2. Filters (Filtros)
```python
for grupo in generar_mapa_validacion_inputs(FilterSchemaFactory.get_all_filter_schemas()):
    for boton_id, inputs_map in grupo.items():
        filterCallbackRegister(boton_id, inputs_map)
```

**Archivos afectados**: `filtros.py`

#### 3. Classifiers P300
```python
for grupo in generar_mapa_validacion_inputs(_tag_classifier_schemas("_p300")):
    for boton_id, inputs_map in grupo.items():
        ClassifierCallbackRegister(boton_id, inputs_map)
```

**Archivos afectados**: `modelado_p300.py`

#### 4. Classifiers Inner Speech
```python
for grupo in generar_mapa_validacion_inputs(_tag_classifier_schemas("_inner")):
    for boton_id, inputs_map in grupo.items():
        ClassifierCallbackRegister(boton_id, inputs_map)
```

**Archivos afectados**: `modelado_inner_speech.py`

### Helper: `_tag_classifier_schemas()`

```python
def _tag_classifier_schemas(type_suffix: str):
    schemas = ClassifierSchemaFactory.get_all_classifier_schemas()
    for s in schemas.values():
        base = s.get("title", s["type"])
        s["title"] = f"{base}{type_suffix}"  # p.ej., "LSTM_p300" o "LSTM_inner"
    return schemas
```

**Por qué**: Los modelos de clasificación son los mismos (LSTM, CNN, SVM, etc.), pero se usan en diferentes paradigmas. El sufijo permite:
1. IDs únicos en la UI (`btn-aplicar-LSTM_p300` vs `btn-aplicar-LSTM_inner`)
2. Callbacks separados por paradigma
3. Evitar colisiones de IDs entre páginas

---

## Estilos CSS

### Clases usadas

```css
.right-panel-container {
    /* Contenedor principal */
}

.right-panel-title {
    /* Título "Filtros", "Extractores de características", etc. */
}

.right-panel-card {
    /* Card individual por filtro/transformada */
}

.right-panel-card-header {
    /* Header de cada card (nombre del filtro) */
}

.input-field-group {
    /* Grupo de label + input */
}
```

### Estilos inline importantes

**Label**:
```python
style={
    "minWidth": "140px",
    "color": "white",
    "fontSize": "13px"
}
```

**Input**:
```python
style={
    "flex": "1",
    "fontSize": "15px",
    "height": "42px",
    "padding": "8px 12px"
}
```

**Dropdown**:
```python
style={
    "flex": "1",
    "color": "black",  # Texto negro para legibilidad
    "fontSize": "14px"
}
```

**Botón Aplicar**:
```python
style={
    "fontSize": "15px",
    "height": "42px",
    "fontWeight": "600"
}
```

---

## Ejemplo Completo: ICA

### Schema Pydantic (entrada)

```python
class ICA(Filter):
    sp: float  # Frecuencia de muestreo
    numeroComponentes: int = 10
    method: Literal["fastica", "infomax", "picard"] = "fastica"
    random_state: Optional[int] = None
    max_iter: int = 200
```

### Schema JSON generado

```json
{
  "title": "ICA",
  "type": "object",
  "properties": {
    "sp": {
      "type": "number",
      "title": "Sp"
    },
    "numeroComponentes": {
      "type": "integer",
      "default": 10,
      "title": "Numerocomponentes"
    },
    "method": {
      "anyOf": [
        {"const": "fastica"},
        {"const": "infomax"},
        {"const": "picard"}
      ],
      "default": "fastica"
    },
    "random_state": {
      "anyOf": [
        {"type": "integer"},
        {"type": "null"}
      ],
      "default": null
    },
    "max_iter": {
      "type": "integer",
      "default": 200
    }
  }
}
```

### UI Generada

```
┌─────────────────────────────────────────┐
│ ICA                                     │  ← Card Header
├─────────────────────────────────────────┤
│ Frecuencia de Muestreo  [________]      │  ← Input numérico
│ Número de componentes   [___10___]      │  ← Input numérico (default=10)
│ Método                  [▼ fastica]     │  ← Dropdown (fastica/infomax/picard)
│ Semilla aleatoria       [________]      │  ← Input numérico (opcional)
│ Iteraciones máximas     [___200__]      │  ← Input numérico (default=200)
│                                         │
│              [ Aplicar ]                 │  ← Botón (id="btn-aplicar-ICA")
└─────────────────────────────────────────┘
```

### IDs Generados

- `ICA-sp`
- `ICA-numeroComponentes`
- `ICA-method`
- `ICA-random_state`
- `ICA-max_iter`
- `btn-aplicar-ICA`

---

## Flujo de Datos al Aplicar

```
Usuario llena formulario
    ↓
Hace clic en "Aplicar" (btn-aplicar-ICA)
    ↓
Callback registrado se ejecuta (filterCallbackRegister)
    ↓
Recolecta valores de todos los States (ICA-sp, ICA-numeroComponentes, etc.)
    ↓
Construye diccionario de parámetros
    ↓
Valida con Pydantic: ICA(**parametros)
    ↓
Aplica filtro al archivo cargado
    ↓
Guarda resultado en /filtered/
    ↓
Actualiza store (filtered-signal-store-filtros)
    ↓
UI se actualiza automáticamente (columna derecha)
```

---

## Ventajas del Diseño

### 1. Generación Dinámica
- ✅ No hay código hardcodeado por filtro/transformada
- ✅ Agregar nuevo filtro = solo crear clase Pydantic
- ✅ Modificar parámetros = solo editar clase Pydantic

### 2. Validación Automática
- ✅ Pydantic valida tipos, rangos, opciones
- ✅ Mensajes de error claros
- ✅ Defaults aplicados automáticamente

### 3. Internacionalización (i18n)
- ✅ Diccionario centralizado `NOMBRE_CAMPOS_ES`
- ✅ Fácil agregar otros idiomas
- ✅ Fallback a nombres técnicos si no hay traducción

### 4. Mantenibilidad
- ✅ Un solo lugar para cambiar estilos
- ✅ Un solo lugar para cambiar lógica de detección de tipos
- ✅ Callbacks registrados automáticamente

### 5. Consistencia
- ✅ Todas las páginas usan el mismo componente
- ✅ UI uniforme en toda la app
- ✅ Comportamiento predecible

---

## Limitaciones Actuales

### 1. Arrays como strings
- **Problema**: Arrays se ingresan como texto separado por comas
- **Impacto**: Usuario puede cometer errores de formato
- **Solución futura**: Componente multi-input o JSON editor

### 2. Sin validación en frontend
- **Problema**: Validación solo en backend (Pydantic)
- **Impacto**: Usuario ve errores después de hacer clic
- **Solución futura**: Validación en clientside callback

### 3. Defaults no siempre visibles
- **Problema**: Si el campo tiene default, puede no mostrarse en el input
- **Impacto**: Usuario no sabe qué valor se usará
- **Solución futura**: Usar `placeholder` o `value` para mostrar default

### 4. Descripciones no mostradas
- **Problema**: Los schemas tienen `description` pero no se muestran
- **Impacto**: Usuario no sabe qué hace cada parámetro
- **Solución futura**: Agregar tooltips o ayuda contextual

---

## Cambios Recientes

### 2025-11-02: Dropdown Context-Aware para Wavelets ✨

**Problema**: El campo `wavelet` mostraba 60+ opciones para todos los casos, pero `WaveletsBase` (filtro) solo acepta 16 wavelets específicos mientras que `WaveletTransform` (transformada) acepta el catálogo completo.

**Solución implementada** (líneas 76-122):

#### Detección Automática del Contexto

```python
# Detectar si es WaveletsBase (filtro) o WaveletTransform (transformada)
is_filter = "WaveletsBase" in type
```

El sistema detecta automáticamente si está generando el formulario para un filtro o una transformada basándose en el nombre del tipo.

#### Lógica Diferenciada

**Para Filtros (WaveletsBase)**:
```python
if is_filter:
    # Solo 16 wavelets válidos según el Literal del modelo
    valid_wavelets = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db8',
                    'sym2', 'sym3', 'sym4', 'sym5',
                    'coif1', 'coif2', 'coif3', 'coif5', 'haar']
    dropdown_options = [{"label": w, "value": w} for w in valid_wavelets]
```

**Para Transformadas (WaveletTransform)**:
```python
else:
    # Catálogo completo de wavelets (60+)
    wavelet_families = {
        "Daubechies": [f"db{i}" for i in range(1, 39)],
        "Symlets": [f"sym{i}" for i in range(2, 21)],
        "Coiflets": [f"coif{i}" for i in range(1, 18)],
        "Biorthogonal": [...],
        "Reverse biorthogonal": [...],
        "Discrete Meyer": ["dmey"],
        "Haar": ["haar"],
    }
```

#### Wavelets Válidos para Filtros

| Familia | Wavelets Disponibles |
|---------|---------------------|
| Daubechies | db1, db2, db3, db4, db5, db6, db8 |
| Symlets | sym2, sym3, sym4, sym5 |
| Coiflets | coif1, coif2, coif3, coif5 |
| Haar | haar |

**Total**: 16 wavelets

#### Comparación

| Contexto | Opciones Mostradas | Propósito |
|----------|-------------------|-----------|
| **Filtros** (`WaveletsBase`) | 16 wavelets | Denoising de señales (limitado por validación Pydantic) |
| **Transformadas** (`WaveletTransform`) | 60+ wavelets | Extracción de características (catálogo completo) |

#### Beneficios

- ✅ Usuario no puede seleccionar opciones inválidas
- ✅ Previene errores de validación antes de enviar
- ✅ Dropdown más limpio y relevante para cada contexto
- ✅ Consistente con las restricciones del backend (Pydantic Literal)
- ✅ Mejora experiencia de usuario significativamente

#### Ejemplo de Error Previo

**Antes**:
```
Usuario abre página de filtros
→ Dropdown muestra todas las 60+ wavelets
→ Selecciona "rbio3.1" (biorthogonal)
→ Hace clic en "Aplicar"
→ ValidationError en consola (no visible para usuario)
```

**Ahora**:
```
Usuario abre página de filtros
→ Dropdown muestra solo 16 wavelets válidos
→ "rbio3.1" no está disponible
→ Solo puede seleccionar wavelets válidos
→ Aplicación funciona correctamente
```

**Archivos relacionados**:
- `WaveletsBase.py` (línea 10): Define `Literal` con 16 wavelets válidos
- `WaveletTransform.py`: Acepta cualquier wavelet del catálogo de PyWavelets

### 2025-11-02: Fix para Dropdowns con valores `None` ✅

**Problema**: Campos tipo `Optional[Literal["ortho", None]]` generaban error en Dash Dropdown:
```
Invalid argument `options` passed into Dropdown with ID "DCTTransform-norm".
Expected one of type [object].
Value provided: [{"label": "ortho", "value": "ortho"}, {"label": "None", "value": null}]
```

**Causa**: Dash Dropdown no acepta `null` como valor válido en las opciones.

**Solución implementada** (líneas 87-122):

1. **Detección de valores `None`**:
   ```python
   # Línea 92: NO filtrar valores None prematuramente
   enum_values = consts  # Mantener None en la lista
   ```

2. **Conversión None → String "None"**:
   ```python
   # Líneas 99-111: Procesar valores None
   dropdown_options = []
   has_none = False
   for val in enum_values:
       if val is None:
           has_none = True
       else:
           dropdown_options.append({"label": str(val), "value": val})

   # Si había None, agregar como string
   if has_none:
       dropdown_options.append({"label": "None", "value": "None"})
   ```

3. **Integración en callbacks**:
   - `FilterSchemaFactory.py` (líneas 159-161): Convierte "None" → None
   - `TransformSchemaFactory.py` (líneas 195-197): Convierte "None" → None

**Flujo completo**:
```
Pydantic: norm: Optional[Literal["ortho", None]]
    ↓
JSON Schema: anyOf: [{const: "ortho"}, {type: "null"}]
    ↓
RightColumn: enum_values = ["ortho", None]
    ↓
Dropdown: options = [
    {"label": "ortho", "value": "ortho"},
    {"label": "None", "value": "None"}  ← String válido
]
    ↓
Usuario selecciona: "None" (string)
    ↓
Callback: value == "None" → convierte a None (Python)
    ↓
Pydantic valida: norm = None ✅
```

**Beneficios**:
- ✅ Dropdowns con `Optional[Literal[..., None]]` funcionan correctamente
- ✅ Usuario puede ver y seleccionar explícitamente "None"
- ✅ Conversión automática en callbacks mantiene tipos correctos
- ✅ Solución genérica aplicable a cualquier campo con este patrón

**Archivos modificados**:
- `RigthComlumn.py` (líneas 87-122)
- `FilterSchemaFactory.py` (líneas 159-161)
- `TransformSchemaFactory.py` (líneas 195-197)

---

### 2025-11-02: Feedback Visual de Errores en Botones ✅

**Problema**: Los errores de validación solo aparecían en la consola del navegador. El usuario hacía clic en "Aplicar" y no recibía feedback visual de qué había salido mal.

**Solución implementada** (FilterSchemaFactory.py y TransformSchemaFactory.py):

#### Mensajes de Error en Botones

El texto del botón ahora cambia para mostrar mensajes de error descriptivos:

**Tipos de errores manejados**:

| Tipo de Error | Ejemplo de Mensaje | Cuándo Ocurre |
|---------------|-------------------|---------------|
| ValidationError | `❌ Error: wavelet, threshold` | Campos con valores inválidos según Pydantic |
| ValueError | `❌ Error: Nivel inválido: 6. Permitido hasta 5` | Validaciones específicas del backend |
| Sin señal cargada | `❌ No hay señal cargada` | Usuario no ha seleccionado evento |
| Archivo no encontrado | `❌ Archivo no encontrado` | Error al guardar/cargar archivos procesados |
| Error inesperado | `❌ Error inesperado` | Excepciones no manejadas |

#### Implementación

**FilterSchemaFactory.py** (líneas 224-242):
```python
except ValidationError as e:
    errores = e.errors()
    error_fields = [err['loc'][0] for err in errores if err['loc']]
    msg_short = f"❌ Error: {', '.join(error_fields)}"
    return msg_short, no_update  # Actualiza el botón con el mensaje

except ValueError as e:
    error_msg = f"❌ Error: {str(e)}"
    return error_msg, no_update

except Exception as e:
    return f"❌ Error inesperado", no_update
```

**TransformSchemaFactory.py** (líneas 390-403) - Misma implementación

#### Flujo de Usuario

**Antes**:
```
Usuario llena formulario con valor inválido
→ Hace clic en "Aplicar"
→ Error solo en consola (F12)
→ Usuario no sabe qué pasó
→ Tiene que revisar la consola
```

**Ahora**:
```
Usuario llena formulario con valor inválido
→ Hace clic en "Aplicar"
→ Botón muestra: "❌ Error: campo1, campo2"
→ Usuario ve inmediatamente qué está mal
→ Puede corregir y reintentar
```

#### Beneficios

- ✅ Feedback inmediato sin revisar consola
- ✅ Mensajes concisos y accionables
- ✅ Indica exactamente qué campos tienen error
- ✅ Consistente con el estilo de la app (español)
- ✅ Mejor experiencia de usuario

#### Ejemplo Real

**Caso 1: ValidationError**
```python
# Usuario ingresa level=6 pero la señal solo permite level=5
Botón muestra: "❌ Error: level"
Consola: "Nivel inválido: 6. Permitido hasta 5 para longitud 2612."
```

**Caso 2: ValueError**
```python
# Usuario selecciona wavelet inválido (antes del dropdown context-aware)
Botón muestra: "❌ Error: wavelet"
Consola: "Input should be 'db1', 'db2', ..., 'haar'"
```

**Caso 3: Sin señal**
```python
# Usuario intenta aplicar filtro sin cargar evento
Botón muestra: "❌ No hay señal cargada"
```

**Archivos modificados**:
- `FilterSchemaFactory.py` (líneas 140-142, 189-201, 223-242)
- `TransformSchemaFactory.py` (líneas 164-166, 177-179, 283-285, 321-324, 390-403)

---

## Próximas Mejoras

### Implementadas ✅
- ✅ **Dropdown context-aware** - Opciones diferenciadas según contexto (filtro vs transformada)
- ✅ **Feedback visual de errores** - Mensajes en botones cuando algo falla
- ✅ **Dropdowns con valores None** - Conversión automática None ↔ "None"

### Pendientes
- [ ] Agregar tooltips con descripciones de campos
- [ ] Validación en frontend (clientside) para feedback instantáneo
- [ ] Componente especial para arrays con editor visual
- [ ] Mostrar valores default en placeholders
- [ ] Agregar botón "Reset" para limpiar formulario
- [ ] Permitir guardar/cargar configuraciones predefinidas (presets)
- [ ] Mostrar unidades de medida (Hz, ms, etc.) junto a labels
- [ ] Agregar validación visual en tiempo real (borde rojo si inválido)
- [ ] Soporte para campos anidados (objetos dentro de objetos)
- [ ] Editor JSON para configuración avanzada

---

## Referencias

### Archivos relacionados
- `backend/classes/Filter/FilterSchemaFactory.py` - Factory de filtros
- `backend/classes/FeatureExtracture/TransformSchemaFactory.py` - Factory de transformadas
- `backend/classes/ClasificationModel/ClassifierSchemaFactory.py` - Factory de clasificadores
- `backend/helpers/mapaValidacion.py` - Helper para generar mapas de inputs
- `app/pages/filtros.py` - Página que usa filtros
- `app/pages/extractores.py` - Página que usa transformadas
- `assets/palette.css` - Variables CSS

### Callbacks relacionados
- `filterCallbackRegister()` - Filtros (FilterSchemaFactory.py:99-223)
- `TransformCallbackRegister()` - Transformadas
- `ClassifierCallbackRegister()` - Clasificadores

### Componentes similares
- `SideBar.py` - Navegación lateral
- `PlayGround.py` - Visualización central
