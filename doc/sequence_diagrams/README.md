# Diagramas de Secuencia - EGG-lab

Esta carpeta contiene diagramas de secuencia en formato PlantUML que documentan los flujos principales del sistema EGG-lab.

## Diagramas Disponibles

### 01_carga_datos.puml
**Flujo de Carga de Datos EEG**
- Proceso de carga de datasets desde archivos .bdf/.edf/.vhdr
- Detección automática de tipo de dataset (Inner Speech vs Genérico)
- Extracción de eventos con timestamps
- Cálculo de metadata completo (frecuencias, canales, estadísticas)
- Sincronización automática con Azure Blob Storage para respaldo en la nube

### 02_aplicar_transformaciones.puml
**Flujo de Aplicación de Transformaciones**
- Selección y configuración de transformaciones (FFT, Wavelet, DCT)
- Validación de esquemas con Pydantic
- Aplicación de transformaciones sobre datos (ya filtrados)
- Re-etiquetado según model_type (P300/Inner Speech)
- Visualización de resultados

### 03_entrenamiento_modelo.puml
**Flujo de Entrenamiento de Modelo (Local / Nube)**
- Selección de arquitectura (CNN, LSTM, GRU, SVM, Random Forest, SVNN)
- Configuración de hiperparámetros
- División estratificada train/test
- **Dos opciones de entrenamiento:**
  - **Local**: Entrenamiento en máquina local con GPU/CPU
  - **Nube**: Entrenamiento en Azure ML con compute clusters
- Seguimiento de métricas en tiempo real
- Evaluación y guardado de resultados

### 04_simulacion_tiempo_real.puml
**Flujo de Simulación en Tiempo Real (P300 + Inner Speech)**
- Carga de modelos P300 e Inner Speech entrenados
- Simulación con ventana deslizante (hop_size configurable)
- Pipeline de inferencia en dos etapas:
  1. P300: Detección de eventos
  2. Inner Speech: Clasificación de acción mental (solo si se detectó evento)
- Cálculo de métricas globales (accuracy, confusion matrix)

### 05_pipeline_transformaciones.puml
**Flujo de Pipeline de Transformaciones (Encadenamiento)**
- Construcción de pipelines con múltiples transformaciones
- Aplicación secuencial: salida de Transform1 → entrada de Transform2
- Registro de configuraciones intermedias
- Trazabilidad de experimentos

### 06_gestion_experimentos.puml
**Flujo de Gestión de Experimentos**
- Creación de experimentos con IDs únicos
- Registro de transformaciones y configuraciones
- Guardado de modelos con metadata completa
- Consulta de historial de experimentos
- Estructura de archivos en `Experiments/`

### 07_flujo_completo_e2e.puml
**Flujo Completo End-to-End**
- Visión general del ciclo completo: Datos → Filtros → Transformaciones → Entrenamiento → Predicción
- 6 fases principales:
  1. Carga de datos
  2. Aplicación de filtros (Notch, Bandpass)
  3. Aplicación de transformaciones (FFT, Wavelet)
  4. Entrenamiento del modelo
  5. Evaluación (aplicando pipeline completo)
  6. Guardado para producción

### 08_integracion_azure.puml
**Flujo de Integración con Azure (Respaldo y Entrenamiento en la Nube)**
- Sincronización de datos con Azure Blob Storage
- Configuración de workspace de Azure ML
- Entrenamiento distribuido en compute clusters
- Descarga y registro de modelos entrenados en la nube
- 4 fases principales:
  1. Respaldo de datos en Blob Storage
  2. Configuración de entrenamiento en Azure ML
  3. Ejecución de job de entrenamiento
  4. Descarga y registro del modelo

### 09_visualizacion_dataset.puml
**Flujo de Visualización y Análisis de Dataset**
- Navegación por pestañas de análisis del dataset
- Visualización de estadísticas (media, std, distribución de clases)
- Generación de topomaps cerebrales (mapas espaciales de actividad)
- Visualización de heatmaps (actividad temporal por canal)
- Gráficos de señal cruda multicanal
- Selección interactiva de clases y rangos temporales

### 10_aplicar_filtros.puml
**Flujo de Aplicación de Filtros**
- Selección de tipo de filtro (Notch, Bandpass, Highpass, Lowpass)
- Configuración de parámetros (frecuencias, orden, ventana)
- Diseño de filtro (cálculo de coeficientes)
- Aplicación con scipy.signal.filtfilt (zero-phase)
- Pipeline de filtros encadenados (ej: Notch 60Hz → Bandpass 0.5-40Hz)
- Registro en experimento para reproducibilidad

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

## Estilo de Diagramas

Los diagramas siguen un estilo simple y claro:
- Actor (Usuario) representado con figura de persona
- Participantes (componentes del sistema) como cajas
- Flechas con etiquetas descriptivas
- Activaciones para mostrar procesamiento
- Loops y alternativas para lógica condicional
