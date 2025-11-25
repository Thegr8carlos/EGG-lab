# Sistema de Re-etiquetado Adaptativo

## 🎯 Pipeline de Procesamiento

**Flujo General:**
- Señal cruda → Filtros → Transformadas → Re-etiquetado → Entrenamiento

**Componentes:**
- **Filtros**: BandPass, Notch, ICA
- **Transformadas**: FFT, Wavelet, DCT, Windowing

---

## 🏷️ Estrategias de Re-etiquetado

### 1️⃣ P300 - Detección de Eventos (Binario)

**Objetivo:** Detectar si hay intención de movimiento

**Mapeo:**
- `0` = **NonTarget** (rest, none)
- `1` = **Target** (abajo, arriba, derecha, izquierda, etc.)

**Ejemplo:**
```
Entrada:  ["rest", "abajo", "arriba", "rest", "derecha"]
Salida:   [   0,      1,       1,       0,        1    ]
```

**Uso:** Primera etapa del sistema (filtro inicial)

---

### 2️⃣ Inner Speech - Clasificación Multiclase

**Objetivo:** Identificar qué pensamiento/palabra mental

**Mapeo:**
- Orden alfabético desde `0`
- **Excluye:** rest, none, unlabeled

**Ejemplo:**
```
Clases:   [abajo, arriba, derecha, izquierda, rest]

Filtrado: [abajo, arriba, derecha, izquierda]  ← rest excluido

Mapeo:    abajo(0), arriba(1), derecha(2), izquierda(3)
```

**Uso:** Segunda etapa (clasificación de acción)

---

### 3️⃣ Genérico - Multiclase Simple

**Objetivo:** Clasificación estándar de todas las clases

**Mapeo:**
- Orden alfabético desde `0`
- **Incluye** todas las clases (incluso rest)

**Uso:** Experimentos genéricos sin arquitectura dual

---

## 🔄 Proceso de Ventaneo

**¿Por qué ventanear?**
- Crea múltiples ejemplos de entrenamiento
- Permite capturar patrones temporales

**Parámetros:**
- `frame_length`: Tamaño de ventana (muestras)
- `overlap`: Solapamiento (0.0 - 1.0)
- `hop`: Salto entre ventanas = `frame_length × (1 - overlap)`

**Asignación de etiquetas:**
- Voto mayoritario por ventana
- Ejemplo: Si 80% de samples son "abajo" → etiqueta = "abajo"

---

## 📊 Arquitectura Dual (P300 + Inner Speech)

**Sistema de 2 Etapas:**

1. **P300 Model** (Binario):
   - Input: Señal EEG
   - Output: ¿Hay evento? (0/1)
   - Threshold: Solo procesa si probabilidad > umbral

2. **Inner Speech Model** (Multiclase):
   - Input: Señal EEG (solo si P300 detectó evento)
   - Output: Clase específica (abajo, arriba, derecha, izquierda)

**Ventajas:**
- ✅ Reduce falsos positivos
- ✅ Más eficiente computacionalmente
- ✅ Mejor accuracy en clasificación final

---

## 🔢 Salida de Transformadas

**Formato estandarizado:**
```
Shape: (n_frames, n_features, n_channels)

n_frames:   Número de ventanas
n_features: Dimensión de features (ej: frecuencias en FFT)
n_channels: Canales EEG (ej: 64, 137)
```

**Archivos generados:**
- `{stem}_fft_{id}.npy`: Datos transformados
- `{stem}_fft_{id}_labels.npy`: Etiquetas numéricas
- `{stem}_fft_{id}_mapping.json`: Mapeo id→clase

---

## ⚙️ Configuración en JSON

**Ejemplo para P300:**
```json
{
  "transform_type": "FFT",
  "window": "hamming",
  "frame_length": 2000,
  "overlap": 0.5,
  "model_type": "p300"
}
```

**Ejemplo para Inner Speech:**
```json
{
  "transform_type": "FFT",
  "window": "hamming",
  "frame_length": 2000,
  "overlap": 0.5,
  "model_type": "inner",
  "all_classes": ["abajo", "arriba", "derecha", "izquierda"]
}
```

---

## 🎓 Reproducibilidad Científica

**Trazabilidad completa:**
- Cada transformada guarda su configuración
- Experimento registra todo el pipeline
- Mapeo de clases incluido en resultados

**Metadatos guardados:**
- Input shape original
- Output shape estandarizado
- Semántica de ejes (tiempo, frecuencias, canales)
- Configuración de ventaneo

---

## 💡 Puntos Clave

✅ **Adaptativo**: Mismo código, diferentes estrategias según `model_type`

✅ **Flexible**: Soporta P300, Inner Speech, y clasificación genérica

✅ **Reproducible**: Todo se serializa en JSON

✅ **Eficiente**: Vectorización con NumPy

✅ **Robusto**: Validación con Pydantic
