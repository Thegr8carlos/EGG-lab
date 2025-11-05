# Sistema Interactivo de Construcción de Arquitecturas de Redes Neuronales

## 📋 Descripción General

Sistema completo e interactivo para que el usuario pueda construir arquitecturas de redes neuronales de manera visual y validada. Permite agregar capas dinámicamente, configurarlas paso a paso, y visualizar la arquitectura completa en tiempo real.

---

## 🎨 Características Principales

### 1. **Layout Optimizado**
- **Columna izquierda**: 85% del espacio - Área de configuración
- **Barra derecha**: Pegada completamente a la derecha (`position: fixed`)
- Diseño responsivo con scroll independiente

### 2. **Visualización de Arquitectura**
La arquitectura se visualiza con:
- **Input Layer** (fijo) - Nodo azul con icono al inicio
- **Capas configurables** - Nodos numerados con colores por tipo
- **Output Layer** (fijo) - Nodo verde con icono al final
- **Flechas conectoras** entre todas las capas
- **Placeholder** cuando no hay capas: "Agrega capas aquí"

### 3. **Construcción Dinámica**
- El usuario puede **agregar ilimitadas capas**
- Botones con iconos para cada tipo de capa
- **Validación en tiempo real** de reglas de negocio
- **Mensajes de confirmación/error** visuales
- **Descripciones contextuales** de cada tipo de capa

### 4. **Navegación por Pasos**
- **Solo se muestra la configuración de la capa actual**
- Indicador de paso: `Paso X/Y: [Nombre de Capa]`
- Navegación con:
  - Botones "Anterior" / "Siguiente"
  - Click directo en los nodos numerados
- **Botón "Eliminar Capa"** para remover la capa actual

### 5. **Editor Matricial para CNN** 🆕
- **Definición visual de kernels** con matrices editables
- **Múltiples filtros** por capa convolucional
- Selección de tamaño de kernel (3×3, 5×5, 7×7)
- Editor para 3 kernels por filtro (R, G, B)
- Configuración de stride, padding y activación por filtro

---

## 📚 Descripción de Capas

Cada capa muestra una descripción breve cuando se selecciona:

### **LSTMLayer**
> **Procesa secuencias temporales capturando dependencias a largo plazo.**
>
> Las LSTM (Long Short-Term Memory) son ideales para datos secuenciales como señales EEG. Pueden recordar información importante durante largos períodos y olvidar la irrelevante mediante sus puertas de entrada, salida y olvido.

### **GRULayer**
> **Versión simplificada de LSTM, procesa secuencias de forma más eficiente.**
>
> Las GRU (Gated Recurrent Units) son más rápidas que LSTM con solo 2 puertas (reset y update). Funcionan bien para secuencias donde las dependencias no son extremadamente largas.

### **DenseLayer**
> **Capa completamente conectada que aprende representaciones no lineales.**
>
> Cada neurona está conectada a todas las neuronas de la capa anterior. Es la capa más común para clasificación y aprendizaje de patrones complejos después de la extracción de características.

### **ConvolutionLayer**
> **Extrae características espaciales usando filtros deslizantes (kernels).**
>
> Aplica múltiples filtros sobre la entrada para detectar patrones locales como bordes, texturas o formas. Cada filtro aprende a detectar un tipo específico de característica en diferentes posiciones de la imagen/señal.

### **PoolingLayer**
> **Reduce dimensionalidad preservando características importantes.**
>
> Max Pooling toma el valor máximo en cada región, manteniendo las características más prominentes. Avg Pooling promedia los valores. Ambos reducen el tamaño espacial y el costo computacional.

### **Dropout**
> **Regularización: desactiva neuronas aleatoriamente para evitar overfitting.**
>
> Durante el entrenamiento, apaga aleatoriamente un porcentaje de neuronas. Esto previene que la red dependa demasiado de neuronas específicas y mejora la generalización.

### **BatchNorm**
> **Normaliza las activaciones para entrenamiento más estable y rápido.**
>
> Normaliza las salidas de cada capa para tener media 0 y varianza 1. Acelera el entrenamiento, permite tasas de aprendizaje más altas y actúa como regularización.

### **Flatten**
> **Convierte matrices multidimensionales en un vector 1D.**
>
> Transforma la salida de capas convolucionales/pooling (matrices 2D/3D) en un vector plano que puede alimentar capas densas. Esencial para la transición de CNN a clasificador denso.

---

## 🎯 Iconos y Colores por Tipo de Capa

| Tipo de Capa | Icono | Color | Código |
|--------------|-------|-------|--------|
| Input Layer | `fa-sign-in-alt` | Azul | #4A90E2 |
| LSTM | `fa-project-diagram` | Naranja | #F5A623 |
| GRU | `fa-circle-notch` | Morado | #BD10E0 |
| Dense | `fa-layer-group` | Turquesa | #50E3C2 |
| Convolución | `fa-th` | Verde | #7ED321 |
| Pooling | `fa-compress-arrows-alt` | Rojo | #D0021B |
| Flatten | `fa-align-justify` | Morado oscuro | #9013FE |
| Dropout | `fa-random` | Rojo claro | #FF6B6B |
| BatchNorm | `fa-balance-scale` | Verde claro | #95E1D3 |
| Output Layer | `fa-flag-checkered` | Verde oscuro | #417505 |

---

## 🛡️ Validaciones de Reglas de Negocio

### **Reglas para LSTM**

#### Primera Capa
- ✅ **Debe ser LSTMLayer**
- ❌ No puede ser Dense, Dropout, etc.

#### Secuencia de Capas
- Después de **LSTMLayer**:
  - ✅ Otra LSTMLayer
  - ✅ DenseLayer
  - ✅ Dropout
  - ❌ Nada más

- Después de **DenseLayer**:
  - ✅ Otra DenseLayer
  - ✅ Dropout
  - ❌ **No se puede agregar LSTMLayer**

#### Requisitos Generales
- ✅ Al menos una capa LSTM en la arquitectura
- ❌ No puede terminar con Dropout

---

### **Reglas para GRU**

#### Primera Capa
- ✅ **Debe ser GRULayer**
- ❌ No puede ser Dense, Dropout, etc.

#### Secuencia de Capas
- Después de **GRULayer**:
  - ✅ Otra GRULayer
  - ✅ DenseLayer
  - ✅ Dropout
  - ❌ Nada más

- Después de **DenseLayer**:
  - ✅ Otra DenseLayer
  - ✅ Dropout
  - ❌ **No se puede agregar GRULayer**

#### Requisitos Generales
- ✅ Al menos una capa GRU en la arquitectura
- ❌ No puede terminar con Dropout

---

### **Reglas para CNN**

#### Primera Capa
- ✅ **Debe ser ConvolutionLayer**
- ❌ No puede ser Dense, Pooling, etc.

#### Secuencia de Capas
- **PoolingLayer** solo después de:
  - ✅ ConvolutionLayer
  - ✅ Otra PoolingLayer
  - ❌ Cualquier otra capa

- **Flatten** debe ir:
  - ✅ Después de ConvolutionLayer o PoolingLayer
  - ❌ Antes de capas convolucionales
  - ⚠️ **Requerido antes de capas Dense**

- Después de **Flatten**:
  - ✅ DenseLayer
  - ✅ Dropout
  - ❌ ConvolutionLayer
  - ❌ PoolingLayer

#### Requisitos Generales
- ✅ Al menos una capa Convolucional
- ✅ Si hay Dense, **debe haber Flatten antes**
- ❌ No puede terminar con Dropout

---

### **Reglas para SVNN (Red Neuronal Simple)**

#### Primera Capa
- ✅ Puede ser DenseLayer
- ✅ Puede ser Dropout
- ✅ Puede ser BatchNorm

#### Secuencia de Capas
- Más flexible, pero:
  - ❌ No Dropout consecutivos
  - ❌ No BatchNorm consecutivos

#### Requisitos Generales
- ✅ Al menos una capa Dense
- ❌ No puede terminar con Dropout o BatchNorm

---

### **Reglas Generales para Todos los Modelos**

| Regla | Descripción | Ejemplo Inválido |
|-------|-------------|------------------|
| **No Dropout consecutivos** | No se puede agregar Dropout después de otro Dropout | `Dense → Dropout → Dropout` ❌ |
| **No BatchNorm consecutivos** | No se puede agregar BatchNorm después de otro BatchNorm | `Dense → BatchNorm → BatchNorm` ❌ |
| **Capa final válida** | No puede terminar con Dropout o BatchNorm | `LSTM → Dense → Dropout` ❌ |
| **Arquitectura no vacía** | Debe tener al menos una capa | `Input → Output` ❌ |

---

## 💬 Sistema de Mensajes de Validación

### Alertas Visuales
- **Posición**: Esquina superior derecha
- **Duración**: 4 segundos (auto-dismiss)
- **Dismissable**: El usuario puede cerrarlas manualmente

### Tipos de Mensajes

#### ✅ Éxito (Verde)
```
✓ Capa LSTM agregada
✓ Capa Densa agregada
✓ Capa Convolucional agregada
```

#### ❌ Error (Rojo)
```
⚠ La primera capa debe ser LSTM
⚠ No puedes agregar Dropout después de otro Dropout
⚠ Pooling debe ir después de una capa Convolucional
⚠ No puedes agregar Capa LSTM después de capas Densas
⚠ Después de Flatten solo puedes agregar capas Densas o Dropout
```

---

## 📦 Capas Disponibles por Modelo

### **LSTM**
- 🔶 LSTMLayer
- 🔷 DenseLayer
- 🎲 Dropout

### **GRU**
- 🔮 GRULayer
- 🔷 DenseLayer
- 🎲 Dropout

### **CNN**
- 🟩 ConvolutionLayer
- 🔴 PoolingLayer
- 📋 Flatten
- 🔷 DenseLayer
- 🎲 Dropout

### **SVNN**
- 🔷 DenseLayer
- 🎲 Dropout
- ⚖️ BatchNorm

---

## 🎮 Ejemplos de Flujo de Usuario

### Ejemplo 1: Construcción de LSTM Válida

**Pasos del usuario:**
1. Click en "Agregar Capa LSTM"
   - ✅ `Input → [1: LSTM] → Output`
2. Click en "Agregar Dropout"
   - ✅ `Input → [1: LSTM] → [2: Dropout] → Output`
3. Click en "Agregar Capa Densa"
   - ✅ `Input → [1: LSTM] → [2: Dropout] → [3: Dense] → Output`
4. Click en "Agregar Capa Densa"
   - ✅ `Input → [1: LSTM] → [2: Dropout] → [3: Dense] → [4: Dense] → Output`

**Resultado:** ✅ Arquitectura válida y lista para entrenar

---

### Ejemplo 2: LSTM - Errores Comunes

**Intento 1: Empezar con Dense**
```
Click "Agregar Capa Densa"
❌ Error: "La primera capa debe ser LSTM"
```

**Intento 2: Dropout consecutivo**
```
Input → [1: LSTM] → [2: Dropout]
Click "Agregar Dropout"
❌ Error: "No puedes agregar Dropout después de otro Dropout"
```

**Intento 3: LSTM después de Dense**
```
Input → [1: LSTM] → [2: Dense]
Click "Agregar Capa LSTM"
❌ Error: "No puedes agregar Capa LSTM después de capas Densas"
```

---

### Ejemplo 3: Construcción de CNN Válida

**Pasos del usuario:**
1. Click en "Agregar Capa Convolucional"
   - ✅ `Input → [1: Conv] → Output`
2. Click en "Agregar Pooling"
   - ✅ `Input → [1: Conv] → [2: Pooling] → Output`
3. Click en "Agregar Capa Convolucional"
   - ✅ `Input → [1: Conv] → [2: Pooling] → [3: Conv] → Output`
4. Click en "Agregar Flatten"
   - ✅ `Input → [1: Conv] → [2: Pooling] → [3: Conv] → [4: Flatten] → Output`
5. Click en "Agregar Capa Densa"
   - ✅ `Input → [1: Conv] → [2: Pooling] → [3: Conv] → [4: Flatten] → [5: Dense] → Output`

**Resultado:** ✅ CNN válida con feature extraction y clasificación

---

### Ejemplo 4: CNN - Errores Comunes

**Error 1: Dense sin Flatten**
```
Input → [1: Conv] → [2: Pooling]
Click "Agregar Capa Densa"
(Se agrega pero al validar arquitectura completa)
❌ Error: "Las CNNs deben tener una capa Flatten antes de las capas Densas"
```

**Error 2: Pooling en lugar incorrecto**
```
Input → [1: Conv] → [2: Flatten]
Click "Agregar Pooling"
❌ Error: "Pooling debe ir después de una capa Convolucional"
```

**Error 3: Convolución después de Flatten**
```
Input → [1: Conv] → [2: Flatten]
Click "Agregar Capa Convolucional"
❌ Error: "Después de Flatten solo puedes agregar capas Densas o Dropout"
```

---

## 🔧 Funcionalidades Adicionales

### Navegación
- **Botones Anterior/Siguiente**: Navegar secuencialmente entre capas
- **Click en nodos**: Saltar directamente a cualquier capa
- **Indicador de paso**: Muestra `Paso X/Y: [Nombre de Capa]`
- **Botones deshabilitados**:
  - "Anterior" en la primera capa
  - "Siguiente" en la última capa

### Gestión de Capas
- **Botón "Eliminar Capa"**: Elimina la capa actual
- **Auto-ajuste de paso**: Si eliminas una capa, el paso se ajusta automáticamente
- **Actualización en tiempo real**: La visualización se actualiza inmediatamente

### Configuración de Capas
- **Formulario dinámico**: Generado automáticamente desde schemas JSON
- **Validación de tipos**: Números, enums, booleanos, arrays, strings
- **Valores por defecto**: Pre-cargados desde el schema
- **Inputs específicos**:
  - Dropdowns para enums
  - Number inputs con min/max para números
  - Checkboxes para booleanos
  - Text inputs para strings/arrays

---

## 🎨 Editor Matricial de Kernels CNN

### Características del Editor

#### 1. **Múltiples Filtros**
- Agrega tantos filtros como necesites
- Cada filtro se visualiza en una card separada
- Botón "Agregar Filtro" para crear nuevos
- Botón de eliminar en cada filtro

#### 2. **Definición de Kernels**
Cada filtro contiene **3 kernels** (canales R, G, B):
- Editor de matriz visual para cada kernel
- Valores editables celda por celda
- Inicialización automática en 0.0

#### 3. **Tamaños Disponibles**
- **3×3**: Kernel pequeño, rápido, bueno para detalles finos
- **5×5**: Kernel mediano, balance entre detalle y contexto
- **7×7**: Kernel grande, captura patrones amplios

#### 4. **Parámetros por Filtro**
- **Stride**: Desplazamiento del filtro (vertical × horizontal)
- **Padding**:
  - `Same`: Mantiene dimensiones de salida
  - `Valid`: Sin padding, reduce dimensiones
- **Activación**: ReLU, Tanh, Sigmoid, Linear

### Ejemplo Visual del Editor

```
┌─────────────────────────────────────────────────┐
│ Filtro 1 de 3                          [🗑️]    │
├─────────────────────────────────────────────────┤
│ Tamaño: [3×3 ▼]                                 │
│                                                 │
│ ┌──────────┬──────────┬──────────┐             │
│ │ Kernel R │ Kernel G │ Kernel B │             │
│ ├──────────┼──────────┼──────────┤             │
│ │ 0  0  0  │ 0  0  0  │ 0  0  0  │             │
│ │ 0  1  0  │ 0  0  0  │ 0  0  0  │             │
│ │ 0  0  0  │ 0  0  0  │ 0  0  0  │             │
│ └──────────┴──────────┴──────────┘             │
│                                                 │
│ Stride: [1] × [1]                               │
│ Padding: [Same ▼]                               │
│ Activación: [ReLU ▼]                            │
└─────────────────────────────────────────────────┘

[➕ Agregar Filtro]  Total: 1 filtro(s)
```

---

## 📁 Estructura de Archivos

### Archivos Principales

```
src/
├── app/
│   ├── components/
│   │   ├── interactive_architecture_builder.py  # Sistema interactivo completo
│   │   ├── cnn_kernel_editor.py                 # Editor matricial de kernels (NUEVO)
│   │   ├── model_config_cards.py                # Integración con sistema existente
│   │   └── model_cards.py                       # Cards de selección (legacy)
│   └── pages/
│       └── modelado_p300.py                     # Página principal con layout
├── backend/
│   └── classes/
│       └── ClasificationModel/
│           └── ClassifierSchemaFactory.py       # Genera schemas JSON
└── schemas.json                                  # Schemas de todos los modelos
```

---

## 🚀 Uso del Sistema

### Para el Usuario Final

1. **Seleccionar modelo** de la barra derecha
2. **Ver pantalla de bienvenida** con Input → [vacío] → Output
3. **Agregar capas** usando los botones con iconos
   - Ver descripción de cada tipo de capa
4. **Configurar cada capa**:
   - Navegar con botones o clicks en nodos
   - Llenar formulario (estándar o matricial para CNN)
   - Eliminar si es necesario
5. **Para capas convolucionales**:
   - Agregar múltiples filtros
   - Definir kernels matricialmente
   - Configurar stride, padding y activación
6. **Validación automática** al agregar cada capa
7. **Probar configuración** cuando esté lista

### Para Desarrolladores

#### Agregar un nuevo tipo de capa

```python
# 1. Agregar color en LAYER_COLORS
LAYER_COLORS = {
    ...
    "new_layer": "#HEXCOLOR"
}

# 2. Agregar icono en LAYER_ICONS
LAYER_ICONS = {
    ...
    "NewLayer": "fa-icon-name"
}

# 3. Agregar nombre amigable en LAYER_NAMES
LAYER_NAMES = {
    ...
    "NewLayer": "Nueva Capa"
}

# 4. Agregar a modelos en AVAILABLE_LAYERS
AVAILABLE_LAYERS = {
    "ModelType": [..., "NewLayer"]
}

# 5. Agregar validaciones en validate_layer_addition()
```

#### Agregar validaciones personalizadas

```python
def validate_layer_addition(new_layer_type, current_layers, model_type):
    # ... código existente ...

    # Nueva regla personalizada
    if model_type == "MiModelo":
        if new_layer_type == "MiCapa" and len(current_layers) > 10:
            return False, "No puedes tener más de 10 capas en MiModelo"

    return True, ""
```

---

## 🎯 Estados del Sistema

### Stores de Dash

| Store | Propósito | Tipo |
|-------|-----------|------|
| `architecture-layers` | Lista de capas agregadas | `List[Dict]` |
| `current-step` | Índice de la capa actual | `int` |
| `model-type` | Tipo de modelo seleccionado | `string` |
| `validation-trigger` | Trigger para mensajes | `Dict` |

### Estructura de una Capa

```python
{
    "type": "LSTMLayer",  # Tipo de capa
    "config": {           # Configuración (se llena en formulario)
        "hidden_size": 128,
        "dropout": 0.2,
        "bidirectional": True,
        # ... más campos según el tipo
    }
}
```

---

## 🔄 Callbacks Implementados

| Callback | Trigger | Acción |
|----------|---------|--------|
| `add_layer` | Click en botón "Agregar [Capa]" | Valida y agrega capa |
| `update_visualization` | Cambio en layers o step | Actualiza visualización |
| `update_step_indicator` | Cambio en step o layers | Actualiza header |
| `show_add_buttons` | Carga del modelo | Muestra botones disponibles |
| `show_current_step_form` | Cambio en step o layers | Muestra formulario |
| `show_navigation` | Cambio en layers o step | Muestra botones nav |
| `navigate_steps` | Click en nav o nodos | Cambia de paso |
| `delete_current_layer` | Click en "Eliminar" | Elimina capa actual |
| `show_validation_message` | Cambio en validation-trigger | Muestra alerta |

---

## 📝 Notas Técnicas

### Limitaciones Conocidas
- Los nodos de Input y Output son visuales únicamente (no configurables)
- La validación de arquitectura completa se ejecuta al hacer click en "Probar Configuración"
- Las capas Dropout y BatchNorm no pueden ser la última capa

### Consideraciones de Performance
- Los callbacks usan `prevent_initial_call=True` para evitar ejecuciones innecesarias
- La visualización se actualiza solo cuando hay cambios reales
- Los mensajes de validación se auto-destruyen después de 4 segundos

### Compatibilidad
- Redes neuronales (LSTM, GRU, CNN, SVNN): Sistema interactivo
- Modelos clásicos (SVM, RandomForest): Formulario simple tradicional

---

## 🎨 Personalización de Estilos

Los estilos utilizan las mismas clases que `RightColumn.py`:
- `.right-panel-card` - Cards de modelos
- `.right-panel-card-header` - Headers de cards
- `.input-field-group` - Grupos de inputs
- `.right-panel-title` - Título de sección
- `.right-panel-container` - Contenedor de cards

---

## ✅ Testing Recomendado

### Casos de Prueba - LSTM

- [ ] Primera capa debe ser LSTM
- [ ] Puede agregar varias capas LSTM seguidas
- [ ] Puede agregar Dense después de LSTM
- [ ] Puede agregar Dropout después de LSTM
- [ ] No puede agregar LSTM después de Dense
- [ ] No puede agregar Dropout consecutivo
- [ ] No puede terminar con Dropout

### Casos de Prueba - CNN

- [ ] Primera capa debe ser Conv
- [ ] Puede agregar Pooling después de Conv
- [ ] Puede agregar múltiples Conv-Pooling
- [ ] Debe agregar Flatten antes de Dense
- [ ] No puede agregar Dense sin Flatten
- [ ] No puede agregar Conv después de Flatten
- [ ] Flatten solo después de Conv/Pooling

### Casos de Prueba - Navegación

- [ ] Click en nodo cambia paso actual
- [ ] Botones Anterior/Siguiente funcionan
- [ ] Eliminar capa actualiza visualización
- [ ] Eliminar última capa ajusta paso
- [ ] Header muestra paso correcto

---

## 📚 Referencias

- **Font Awesome Icons**: https://fontawesome.com/icons
- **Dash Bootstrap Components**: https://dash-bootstrap-components.opensource.faculty.ai/
- **Pydantic Schemas**: Usados para validación automática de configuración

---

## 🤝 Contribuciones Futuras

### Mejoras Sugeridas
- [ ] Drag & drop para reordenar capas
- [ ] Duplicar capas existentes
- [ ] Templates de arquitecturas pre-definidas
- [ ] Exportar/importar configuración JSON
- [ ] Preview de número de parámetros
- [ ] Validación de dimensiones automática
- [ ] Sugerencias de capas según contexto
- [ ] Visualización de kernels como imágenes (heatmaps)
- [ ] Kernels pre-definidos (Sobel, Laplacian, etc.)

---

## 🔄 Changelog

### v1.1 - 2025-02-11
- ✅ Agregadas descripciones contextuales de cada capa
- ✅ Editor matricial interactivo para kernels de CNN
- ✅ Soporte para múltiples filtros en capas convolucionales
- ✅ Selector de tamaño de kernel (3×3, 5×5, 7×7)
- ✅ Configuración visual de matrices R, G, B por filtro
- ✅ Parámetros individuales por filtro (stride, padding, activation)

### v1.0 - 2025-02-11
- ✅ Sistema interactivo base
- ✅ Validaciones de reglas de negocio
- ✅ Visualización con nodos Input/Output fijos
- ✅ Navegación por pasos
- ✅ Iconos y colores por tipo de capa

---

**Versión**: 1.1
**Fecha**: 2025-02-11
**Estado**: ✅ Producción
