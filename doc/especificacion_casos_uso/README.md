# Especificación de Casos de Uso - EGG-lab

Esta carpeta contiene especificaciones detalladas de los casos de uso del sistema EGG-lab en formato tabla PlantUML.

## Archivos Disponibles

### cargar_datos_spec.puml
**Especificaciones para la página "Cargar Datos"**

- **CU-1**: Procesar dataset pendiente
- **CU-2**: Seleccionar dataset procesado
- **CU-3**: Limpiar datos de procesamiento

### filtros_spec.puml
**Especificaciones para la página "Filtros"**

- **CU-1**: Aplicar filtro BandPass
- **CU-2**: Aplicar filtro Notch
- **CU-3**: Aplicar ICA (reducción dimensional)

### modelado_p300_spec.puml
**Especificaciones para la página "Modelado P300"**

- **CU-1**: Configurar modelo P300
- **CU-2**: Entrenar modelo P300 (local)
- **CU-3**: Entrenar modelo P300 (Azure ML)

### modelado_inner_speech_spec.puml
**Especificaciones para la página "Modelado Inner Speech"**

- **CU-1**: Configurar modelo Inner Speech
- **CU-2**: Entrenar modelo Inner Speech
- **CU-3**: Evaluar modelo con métricas

### simulation_spec.puml
**Especificaciones para la página "Simulación"**

- **CU-1**: Configurar simulación en tiempo real
- **CU-2**: Ejecutar simulación dual-stage (P300 + Inner)
- **CU-3**: Visualizar resultados de simulación

### dataset_spec.puml
**Especificaciones para la página "Vista Dataset"**

- **CU-1**: Visualizar estadísticas del dataset
- **CU-2**: Visualizar topomaps cerebrales
- **CU-3**: Visualizar heatmap temporal
- **CU-4**: Visualizar señal cruda multicanal

## Formato de Especificación

Cada caso de uso incluye:

| Campo | Descripción |
|-------|-------------|
| **Caso de uso** | Identificador y título del caso de uso |
| **Actor** | Usuario que interactúa con el sistema |
| **Propósito** | Objetivo que busca alcanzar el caso de uso |
| **Tipo** | Primario o Secundario |
| **Entradas** | Datos que el usuario proporciona |
| **Salidas** | Resultados que el sistema produce |
| **Precondiciones** | Condiciones que deben cumplirse antes |
| **Postcondiciones** | Estado del sistema después de ejecutar |
| **Errores** | Situaciones de error que pueden ocurrir |

## Generación de Diagramas

Para generar imágenes PNG/SVG a partir de los archivos `.puml`:

```bash
# Generar PNG
plantuml -tpng *.puml

# Generar SVG
plantuml -tsvg *.puml
```

O usar el servidor online: http://www.plantuml.com/plantuml/

## Convenciones

- **Casos primarios**: Flujos principales del usuario
- **Casos secundarios**: Funcionalidades auxiliares o avanzadas
- **Errores**: Situaciones excepcionales y cómo manejarlas
- **Precondiciones**: Estados requeridos antes de ejecutar
- **Postcondiciones**: Estados garantizados después de ejecutar

## Estructura del Sistema

```
EGG-lab/
├── Cargar Datos      → Procesar datasets EEG
├── Filtros           → Preprocesamiento (BandPass, Notch, ICA)
├── Extractores       → Transformadas (FFT, Wavelet, DCT, Windowing)
├── Modelado P300     → Entrenar detector de eventos
├── Modelado Inner    → Entrenar clasificador multiclase
├── Simulación        → Predicción en tiempo real (dual-stage)
└── Vista Dataset     → Análisis exploratorio (stats, topomaps, heatmaps)
```

## Flujo Típico de Trabajo

1. **Cargar Datos**: Procesar dataset EEG → Generar metadatos
2. **Vista Dataset**: Explorar estadísticas y visualizaciones
3. **Filtros**: Aplicar preprocesamiento (Notch, BandPass, ICA)
4. **Extractores**: Aplicar transformadas (FFT, Wavelet)
5. **Modelado P300**: Entrenar detector de eventos
6. **Modelado Inner Speech**: Entrenar clasificador de acciones
7. **Simulación**: Probar sistema completo en tiempo real

## Casos de Uso por Prioridad

### Alta Prioridad (Core Functionality)
- CU: Procesar dataset pendiente
- CU: Aplicar filtro BandPass
- CU: Entrenar modelo P300 (local)
- CU: Entrenar modelo Inner Speech
- CU: Ejecutar simulación dual-stage

### Media Prioridad (Features)
- CU: Visualizar estadísticas del dataset
- CU: Aplicar filtro Notch
- CU: Aplicar ICA
- CU: Configurar simulación en tiempo real

### Baja Prioridad (Nice-to-have)
- CU: Entrenar modelo P300 (Azure ML)
- CU: Limpiar datos de procesamiento
- CU: Visualizar topomaps cerebrales
- CU: Visualizar heatmap temporal

## Notas

- Todas las especificaciones siguen el mismo formato estándar
- Los casos de uso están alineados con los diagramas UML existentes
- Las precondiciones y postcondiciones aseguran flujo correcto
- Los errores documentados ayudan a implementar manejo robusto
