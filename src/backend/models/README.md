# Guía de Persistencia de Modelos

## 📁 Estructura de Directorios

```
src/backend/models/
├── p300/              # Modelos para experimentos P300
│   ├── lstm_20251109_143022.pkl
│   ├── svm_20251109_143030.pkl
│   └── svnn_20251109_143045.pkl
├── inner/             # Modelos para Inner Speech
│   ├── lstm_20251109_144000.pkl
│   └── svm_20251109_144100.pkl
└── README.md          # Esta guía
```

## 🔄 Ciclo de Vida del Modelo

### 1️⃣ **Entrenamiento con Auto-Guardado**

```python
from backend.classes.ClasificationModel.LSTM import LSTMNet, LSTMLayer, SequenceEncoder, DenseLayer, ActivationFunction

# Construir arquitectura
lstm_config = LSTMNet(
    encoder=SequenceEncoder(
        input_feature_dim=64,
        layers=[LSTMLayer(hidden_size=128, bidirectional=True)]
    ),
    fc_layers=[DenseLayer(units=64)],
    classification=DenseLayer(units=5, activation=ActivationFunction(kind="softmax"))
)

# Opción A: train() con auto-guardado (legacy API + persistencia)
metrics = LSTMNet.train(
    lstm_config,
    xTest=["data/test_01.npy"],
    yTest=["data/test_labels_01.npy"],
    xTrain=["data/train_01.npy", "data/train_02.npy"],
    yTrain=["data/train_labels_01.npy", "data/train_labels_02.npy"],
    epochs=50,
    batch_size=32,
    model_label="p300"  # 👈 Auto-guarda en: src/backend/models/p300/lstm_TIMESTAMP.pkl
)

# Opción B: fit() con auto-guardado (nueva API completa)
result = LSTMNet.fit(
    lstm_config,
    xTest=["data/test_01.npy"],
    yTest=["data/test_labels_01.npy"],
    xTrain=["data/train_01.npy"],
    yTrain=["data/train_labels_01.npy"],
    epochs=50,
    batch_size=32,
    model_label="inner"  # 👈 Auto-guarda en: src/backend/models/inner/lstm_TIMESTAMP.pkl
)

# Acceder a métricas y modelo
print(f"Accuracy: {result.metrics.accuracy:.3f}")
print(f"Training time: {result.training_seconds:.2f}s")
print(f"Loss curve: {result.history['loss']}")
```

### 2️⃣ **Guardado Manual (Sin Auto-Guardado)**

```python
# Si NO pasas model_label, debes guardar manualmente
result = LSTMNet.fit(lstm_config, xTest, yTest, xTrain, yTrain)

# Opción A: Ruta generada automáticamente
save_path = LSTMNet._generate_model_path("p300")  # src/backend/models/p300/lstm_TIMESTAMP.pkl
lstm_config.save(save_path)

# Opción B: Ruta personalizada
lstm_config.save("custom/path/my_lstm_model.pkl")

# Opción C: Guardar con metadata adicional (recomendado)
experiment_id = "exp_2024_p300_v3"
save_path = f"src/backend/models/p300/{experiment_id}_lstm.pkl"
lstm_config.save(save_path)
```

### 3️⃣ **Carga del Modelo**

```python
# Cargar desde ruta específica
lstm_model = LSTMNet.load("src/backend/models/p300/lstm_20251109_143022.pkl")

# ✅ El modelo está listo para inferencia inmediatamente
predictions = LSTMNet.query(
    lstm_model,
    sequences=[seq1, seq2, seq3],  # Lista de arrays (T, F)
    return_logits=False
)

# Con probabilidades
preds, probs = LSTMNet.query(lstm_model, sequences, return_logits=True)
```

### 4️⃣ **Gestión en Streamlit (Session State)**

```python
import streamlit as st
from backend.classes.ClasificationModel.SVM import SVM

# ==================== PÁGINA DE ENTRENAMIENTO ====================
st.title("Entrenamiento de Modelo")

# Configurar modelo
svm_config = SVM(kernel="rbf", C=1.0, probability=True)

if st.button("Entrenar y Guardar"):
    # Entrenar con auto-guardado
    result = SVM.fit(
        svm_config,
        xTest=xTest_paths,
        yTest=yTest_paths,
        xTrain=xTrain_paths,
        yTrain=yTrain_paths,
        model_label="p300"  # Auto-guarda
    )
    
    # Guardar instancia en session_state para uso inmediato
    st.session_state['trained_model'] = svm_config
    st.session_state['model_path'] = SVM._generate_model_path("p300")
    
    st.success(f"✅ Modelo entrenado y guardado")
    st.write(f"📊 Accuracy: {result.metrics.accuracy:.3f}")
    st.write(f"💾 Guardado en: {st.session_state['model_path']}")

# ==================== PÁGINA DE INFERENCIA ====================
st.title("Inferencia")

# Opción 1: Usar modelo de session_state (si está en misma sesión)
if 'trained_model' in st.session_state:
    model = st.session_state['trained_model']
    st.info("🔥 Usando modelo de sesión activa")
else:
    # Opción 2: Cargar desde disco (nueva sesión o recarga)
    import glob
    available_models = glob.glob("src/backend/models/p300/*.pkl")
    
    if available_models:
        model_path = st.selectbox("Selecciona modelo guardado", available_models)
        model = SVM.load(model_path)
        st.info(f"📂 Modelo cargado desde: {model_path}")
    else:
        st.error("❌ No hay modelos guardados. Entrena uno primero.")
        st.stop()

# Realizar inferencia
if st.button("Predecir"):
    predictions = SVM.query(model, x_new_paths)
    st.write(f"Predicciones: {predictions}")
```

### 5️⃣ **Patrón Recomendado: Hybrid (Session + Disco)**

```python
# En módulo compartido (e.g., src/backend/model_manager.py)
import streamlit as st
from pathlib import Path
import glob

class ModelManager:
    """Gestor centralizado de modelos con fallback automático."""
    
    @staticmethod
    def get_or_load_model(model_type: str, label: str):
        """
        Intenta obtener modelo de session_state, si no existe carga el más reciente.
        
        Args:
            model_type: "lstm", "svm", "svnn"
            label: "p300", "inner", etc.
        
        Returns:
            Instancia del modelo lista para query()
        """
        key = f"model_{model_type}_{label}"
        
        # 1. Buscar en session_state (rápido)
        if key in st.session_state:
            return st.session_state[key]
        
        # 2. Cargar último modelo guardado (fallback)
        pattern = f"src/backend/models/{label}/{model_type}_*.pkl"
        models = sorted(glob.glob(pattern), reverse=True)  # Más reciente primero
        
        if models:
            latest = models[0]
            
            # Cargar según tipo
            if model_type == "lstm":
                from backend.classes.ClasificationModel.LSTM import LSTMNet
                model = LSTMNet.load(latest)
            elif model_type == "svm":
                from backend.classes.ClasificationModel.SVM import SVM
                model = SVM.load(latest)
            elif model_type == "svnn":
                from backend.classes.ClasificationModel.SVNN import SVNN
                model = SVNN.load(latest)
            
            # Cachear en session_state
            st.session_state[key] = model
            return model
        
        return None
    
    @staticmethod
    def save_and_cache(model, model_type: str, label: str):
        """Guarda a disco Y cachea en session_state."""
        # Generar ruta según tipo
        if model_type == "lstm":
            from backend.classes.ClasificationModel.LSTM import LSTMNet
            path = LSTMNet._generate_model_path(label)
        elif model_type == "svm":
            from backend.classes.ClasificationModel.SVM import SVM
            path = SVM._generate_model_path(label)
        elif model_type == "svnn":
            from backend.classes.ClasificationModel.SVNN import SVNN
            path = SVNN._generate_model_path(label)
        
        # Guardar
        model.save(path)
        
        # Cachear
        key = f"model_{model_type}_{label}"
        st.session_state[key] = model
        
        return path

# USO:
from backend.model_manager import ModelManager

# Al entrenar
result = SVM.fit(svm_config, xTest, yTest, xTrain, yTrain)
saved_path = ModelManager.save_and_cache(svm_config, "svm", "p300")
st.success(f"Guardado en: {saved_path}")

# Al inferir (en cualquier página)
model = ModelManager.get_or_load_model("svm", "p300")
if model:
    predictions = SVM.query(model, new_data)
else:
    st.error("No hay modelo disponible")
```

## 🎯 Mejores Prácticas

### ✅ **DO: Usar model_label para experimentos estándar**
```python
# Auto-organiza por tipo de experimento
result = LSTMNet.fit(..., model_label="p300")  # ✅
result = SVM.fit(..., model_label="inner")     # ✅
```

### ✅ **DO: Combinar session_state + disco para producción**
```python
# Rápido en misma sesión, persistente entre sesiones
ModelManager.save_and_cache(model, "lstm", "p300")
model = ModelManager.get_or_load_model("lstm", "p300")
```

### ✅ **DO: Usar rutas relativas al proyecto**
```python
# Portabilidad entre máquinas
model.save("src/backend/models/p300/final_model.pkl")  # ✅
```

### ❌ **DON'T: Hardcodear rutas absolutas**
```python
# Rompe en otras máquinas
model.save("C:/Users/hugus/models/model.pkl")  # ❌
```

### ❌ **DON'T: Olvidar manejar ausencia de modelos**
```python
# Puede crashear
model = SVM.load("models/nonexistent.pkl")  # ❌

# Mejor:
try:
    model = SVM.load(path)
except FileNotFoundError:
    st.error("Modelo no encontrado")
```

## 📊 Comparación de Estrategias

| Estrategia | Velocidad | Persistencia | Complejidad |
|------------|-----------|--------------|-------------|
| **Solo Session State** | ⚡⚡⚡ Rápida | ❌ Se pierde al cerrar | 🟢 Baja |
| **Solo Disco** | 🐢 Lenta | ✅ Permanente | 🟢 Baja |
| **Hybrid (Recomendado)** | ⚡⚡ Rápida | ✅ Permanente | 🟡 Media |
| **ModelManager** | ⚡⚡ Rápida | ✅ Permanente | 🔴 Alta |

## 🔍 Troubleshooting

### Problema: "Modelo no entrenado: usa fit() antes de query()"
**Solución**: El modelo cargado no tiene `_tf_model/_svc_model/_keras_model` poblado.
```python
# Verificar antes de query
if hasattr(model, '_svc_model') and model._svc_model is not None:
    predictions = SVM.query(model, data)
else:
    st.error("Modelo no tiene estado entrenado")
```

### Problema: "FileNotFoundError al cargar modelo"
**Solución**: Verificar que el archivo existe y la ruta es correcta.
```python
from pathlib import Path

model_path = "src/backend/models/p300/lstm_20251109_143022.pkl"
if Path(model_path).exists():
    model = LSTMNet.load(model_path)
else:
    st.error(f"Archivo no encontrado: {model_path}")
```

### Problema: "pickle.UnpicklingError" o incompatibilidad de versiones
**Solución**: Los modelos guardados con pickle dependen de las versiones de librerías.
```python
# Registrar versiones al guardar (en hyperparams)
import tensorflow as tf
import sklearn

result = TrainResult(
    ...,
    hyperparams={
        "tf_version": tf.__version__,
        "sklearn_version": sklearn.__version__,
        ...
    }
)
```

## 📚 Ejemplos Completos por Modelo

### LSTM
```python
from backend.classes.ClasificationModel.LSTM import LSTMNet

# Entrenar y auto-guardar
result = LSTMNet.fit(lstm_config, xTest, yTest, xTrain, yTrain, 
                     model_label="p300", epochs=50)

# Cargar y usar
model = LSTMNet.load("src/backend/models/p300/lstm_TIMESTAMP.pkl")
predictions = LSTMNet.query(model, sequences)
```

### SVM
```python
from backend.classes.ClasificationModel.SVM import SVM

# Entrenar y auto-guardar
result = SVM.fit(svm_config, xTest, yTest, xTrain, yTrain, 
                 model_label="inner")

# Cargar y usar
model = SVM.load("src/backend/models/inner/svm_TIMESTAMP.pkl")
predictions = SVM.query(model, x_paths)
```

### SVNN
```python
from backend.classes.ClasificationModel.SVNN import SVNN

# Entrenar y auto-guardar
result = SVNN.fit(svnn_config, xTest, yTest, xTrain, yTrain, 
                  model_label="p300")

# Cargar y usar
model = SVNN.load("src/backend/models/p300/svnn_TIMESTAMP.pkl")
predictions = SVNN.query(model, x_paths)
```

## 🚀 Resumen Rápido

```python
# 1. ENTRENAR con auto-guardado
result = Model.fit(..., model_label="p300")  # Guarda automáticamente

# 2. USAR inmediatamente (mismo objeto)
predictions = Model.query(model_instance, data)

# 3. CARGAR después (otra sesión)
model = Model.load("src/backend/models/p300/model_TIMESTAMP.pkl")
predictions = Model.query(model, data)

# 4. GESTIONAR con ModelManager (producción)
ModelManager.save_and_cache(model, "svm", "p300")
model = ModelManager.get_or_load_model("svm", "p300")
```
