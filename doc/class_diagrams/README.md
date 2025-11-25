# Diagramas de Clases - EGG-lab Backend

Esta carpeta contiene diagramas de clases en formato PlantUML que documentan la arquitectura del backend del sistema EGG-lab.

## Diagramas Disponibles

### architecture_general.puml
**Arquitectura General del Sistema**
- Vista de alto nivel de la arquitectura completa
- Componentes principales:
  - **Frontend**: Páginas Dash (Cargar Datos, Filtros, Extractores, Modelado, Simulación)
  - **Backend**: Clases Core (Dataset, Experiment, Filter, Transform, Classifier)
  - **Storage**: Estructura de almacenamiento (Data/, Aux/, Models/, Experiments/)
  - **Pipeline**: Flujo de procesamiento (Carga → Filtros → Transformaciones → Entrenamiento → Evaluación → Simulación)
  - **Azure Cloud**: Integración con Azure ML (FastAPI Server + Azure ML Workspace)
- Relaciones entre componentes
- Dual-mode: Ejecución local y en la nube
- Notas explicativas de cada sección

### azure_architecture.puml
**Arquitectura Azure ML - Ejecución de Pipelines en la Nube**
- Flujo detallado de ejecución en Azure
- **FastAPI Server**:
  - REST API local con identidad Azure
  - Endpoints: /upload-folder, /upload-dataset, /upload-expr
  - No requiere autenticación del usuario
- **Azure ML Workspace**:
  - Data Assets: Datasets registrados como recursos
  - Pipeline Nodes: Nodos de procesamiento (Filter → Transform → Training)
  - Outputs: preprocessed_signal, training_data_path, final_output_path
  - Compute Cluster: Infraestructura escalable con GPU/CPU
  - MLflow Tracking: Registro de experimentos y métricas
- Pipeline como DAG (Grafo Dirigido Acíclico)
- Comunicación entre nodos vía Data Assets temporales
- Escalabilidad: Soporte para múltiples pipelines concurrentes

### dataset_class.puml
**Clase Dataset - Gestión de Datos EEG**
- Funciones principales:
  - Carga de archivos EEG (BDF/EDF/BrainVision)
  - Extracción de eventos con timestamps
  - Mapeo y selección de canales
  - Creación de conjuntos train/test
  - Cálculo y guardado de metadatos
- Soporta múltiples datasets:
  - BNCI2014-001 (64 canales)
  - BNCI2014-004 (3 canales)
  - Nieto Inner Speech (137 canales)
  - Datasets genéricos
- Métodos estáticos para procesamiento
- Estructura de metadatos completa

### experiment_class.puml
**Clase Experiment - Sistema de Gestión de Experimentos**
- Orquesta todo el pipeline de ML
- Gestiona:
  - Referencia al dataset
  - Filtros de preprocesamiento
  - Transformaciones (extracción de características)
  - Clasificadores (P300 + Inner Speech)
  - Métricas de evaluación
- Métodos estáticos para:
  - Crear experimentos
  - Guardar/cargar configuraciones
  - Agregar componentes al pipeline
- Serialización en JSON para reproducibilidad

### filter_class.puml
**Clases Filter - Preprocesamiento de Señales**
- Clase base abstracta `Filter`
- Implementaciones:
  - **BandPass**: Filtrado de frecuencias (ej: 1-30 Hz)
  - **Notch**: Eliminación de frecuencias específicas (ej: 60 Hz)
  - **ICA**: Análisis de Componentes Independientes (reducción dimensional)
  - **WaveletsBase**: Filtrado basado en wavelets
- Flujo de aplicación:
  1. Cargar datos (.npy)
  2. Validar parámetros
  3. Aplicar filtro
  4. Guardar resultado
- Usa MNE, PyWavelets, SciPy

### transform_class.puml
**Clases Transform - Extracción de Características**
- Clase base abstracta `Transform`
- Implementaciones:
  - **FFTTransform**: Transformada rápida de Fourier (espectrogramas)
  - **DCTTransform**: Transformada discreta del coseno
  - **WaveletTransform**: Transformada wavelet
  - **WindowingTransform**: Ventaneo simple
- Funcionalidades:
  - Ventaneo con overlap configurable
  - Re-etiquetado según model_type (P300/Inner Speech)
  - Asignación de etiquetas por voto mayoritario
  - Salida estandarizada: (n_frames, features, canales)
- Sistema de re-etiquetado:
  - **P300 (binario)**: 0=NonTarget, 1=Target
  - **Inner Speech (multiclase)**: 0,1,2,... (alfabético)

### relabeling_flow.puml
**Flujo de Re-etiquetado según Tipo de Modelo**
- Diagrama de flujo que explica el proceso completo de re-etiquetado
- Incluye:
  - Ventaneo con overlap y voto mayoritario
  - Decisión según `model_type` (p300/inner/generic)
  - **P300**: Etiquetado binario (0=NonTarget, 1=Target)
  - **Inner Speech**: Multiclase con filtrado de rest/none/unlabeled
  - **Genérico**: Multiclase simple con todas las clases
  - Generación de valid_mask para filtrar datos
  - Salida: etiquetas numéricas + mapeo + máscara
- **Uso**: Documentación de respaldo para entender el re-etiquetado

### filters_parameters.puml
**Filtros - Parámetros y Funcionalidad**
- Documentación detallada de cada filtro con parámetros y qué hace
- **BandPass**: Filtrado de frecuencias (lowpass/highpass/bandpass)
  - Uso común: 0.5-40 Hz para EEG
  - Métodos: FIR (zero-phase) vs IIR (eficiente)
- **Notch**: Eliminación de frecuencias específicas
  - Uso típico: 60 Hz (ruido eléctrico)
  - Quality factor (Q) configurable
- **ICA**: Descomposición en componentes independientes
  - Reducción dimensional (137→30 canales típico)
  - Métodos: fastica, picard, infomax
- Incluye notas sobre pipeline típico y cuándo usar cada filtro

### transforms_parameters.puml
**Transformadas - Parámetros y Funcionalidad**
- Documentación detallada de cada transformada
- **FFTTransform**: Espectro de frecuencias (espectrogramas)
  - Ventajas: Captura bandas EEG, interpretable
  - Desventajas: Pierde información temporal precisa
- **WaveletTransform**: Tiempo + frecuencia simultáneos
  - Ventajas: Denoising, capta transientes
  - Wavelets: db4, coif5, sym8
- **WindowingTransform**: Ventaneo simple sin procesamiento
  - Ventajas: Rápido, señal cruda
  - Para modelos que aprenden features (CNN, LSTM)
- **DCTTransform**: Compresión eficiente
- Incluye comparativa de ventajas/desventajas

### models_parameters.puml
**Modelos - Parámetros y Funcionalidad**
- Documentación completa de todos los modelos
- **Deep Learning**:
  - **CNN**: Patrones espaciales, convoluciones 2D
  - **LSTM**: Dependencias temporales, memoria largo plazo
  - **GRU**: Similar a LSTM, más rápido
  - **SVNN**: Red feedforward simple
- **Machine Learning**:
  - **SVM**: Hiperplano óptimo, kernel trick
  - **RandomForest**: Ensemble de árboles
- Incluye guía de elección según tamaño de dataset y tipo de features
- Parámetros de entrenamiento: epochs, batch_size, learning_rate, optimizer

### classifier_class.puml
**Clases Classifier - Modelos de Clasificación**
- Clase base abstracta `Classifier`
- **Modelos Deep Learning**:
  - **CNN**: Convolutional Neural Network
    - Kernels personalizados 2D
    - Pooling (Max/Avg)
    - Capas densas finales
  - **LSTM**: Long Short-Term Memory
    - Múltiples capas apiladas
    - Opción bidireccional
    - Pooling strategies: last/mean/max/attn
  - **GRU**: Gated Recurrent Unit
    - Similar a LSTM, más eficiente
  - **SVNN**: Simple Vanilla Neural Network
    - Red feedforward clásica
- **Modelos Machine Learning**:
  - **SVM**: Support Vector Machine
    - Kernels: linear, rbf, poly, sigmoid
    - Regularización C
  - **RandomForest**: Ensemble de árboles
    - N estimadores configurables
    - Balance de clases
- Componentes auxiliares:
  - Kernel, Pool, DenseLayer
  - LSTMLayer, GRULayer
  - TrainResult, EvaluationMetrics
- Métodos comunes:
  - fit(): Entrenar modelo
  - predict(): Inferencia
  - evaluate(): Métricas de evaluación
  - save_model() / load_model()

## Generación de Diagramas

Para generar imágenes SVG/PNG a partir de los archivos `.puml`:

```bash
# Instalar PlantUML
brew install plantuml  # macOS
apt-get install plantuml  # Ubuntu/Debian

# Generar SVG
plantuml -tsvg *.puml

# Generar PNG
plantuml -tpng *.puml
```

O usar el servidor online: http://www.plantuml.com/plantuml/

## Convenciones de Diseño

- **Clases abstractas**: Usamos `abstract class` para clases base
- **Métodos estáticos**: Marcados con `{static}`
- **Atributos privados/protegidos**: Prefijados con `-` o `#`
- **Herencias**: Flecha sólida `<|--`
- **Composiciones**: Rombo relleno `*--`
- **Dependencias**: Flecha punteada `..>`
- **Notas**: Bloques explicativos con `note`

## Estructura del Backend

```
backend/
├── classes/
│   ├── dataset.py              → Dataset
│   ├── Experiment.py           → Experiment
│   ├── Filter/                 → BandPass, Notch, ICA, etc.
│   ├── FeatureExtracture/      → FFT, DCT, Wavelet, etc.
│   └── ClasificationModel/     → CNN, LSTM, SVM, etc.
└── helpers/
    ├── model_storage.py        → Guardar/cargar modelos
    ├── simulation_utils.py     → Utilidades de simulación
    └── simulation_engine.py    → Motor de simulación
```

## Flujo Típico

1. **Dataset**: Carga datos EEG → Genera eventos → Calcula metadata
2. **Filter**: Aplica filtros de preprocesamiento (Notch, BandPass)
3. **Transform**: Extrae características (FFT, Wavelet)
4. **Classifier**: Entrena modelo (CNN, LSTM, SVM)
5. **Experiment**: Orquesta todo y guarda configuración
6. **Simulation**: Usa modelo entrenado para predicción en tiempo real
