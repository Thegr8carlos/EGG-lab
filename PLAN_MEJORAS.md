# 📋 PLAN DE MEJORAS - Sistema EEG Lab

**Fecha**: 2025-11-09
**Problemas identificados**: 4 áreas críticas

---

## 🎯 PROBLEMA 1: Error en Pipeline (ICA + Wavelets)

### Diagnóstico
- **Error**: `FileNotFoundError: 'temp_step_1_input.npy'`
- **Causa raíz**: El pipeline crea archivos intermedios pero no los enlaza correctamente entre pasos
- **Ubicación**: `backend/classes/Experiment.py` líneas 850-900
- **Impacto**: El pipeline no funciona cuando se combinan filtros + transformadas

### Solución: 3 pasos incrementales

#### PASO 1.1: Arreglar flujo de archivos intermedios en pipeline
**Objetivo**: Asegurar que cada paso del pipeline encuentre el archivo de entrada correcto

**Cambios**:
- En `Experiment.py`, método `apply_model_pipeline()`:
  - Después de aplicar cada filtro, copiar explícitamente el output como input del siguiente paso
  - Renombrar archivos intermedios con patrón consistente
  - Agregar validación de existencia de archivos antes de cada paso

**Testing esperado**:
```bash
# Probar aplicar ICA solo → Debe funcionar
# Ver logs: ✅ ICA aplicado correctamente
# Verificar que se crea el archivo en intermediates/
```

**Qué soluciona**: Previene el error de archivo no encontrado
**Cómo lo hace**: Crea enlaces explícitos entre outputs/inputs de pasos

---

#### PASO 1.2: Mejorar logging y manejo de errores del pipeline
**Objetivo**: Detectar y reportar claramente cuándo falla un paso

**Cambios**:
- Agregar try-catch específico por cada paso del pipeline
- Logging detallado de:
  - Archivo de entrada de cada paso
  - Archivo de salida generado
  - Si el archivo existe después de aplicar
- Si un paso falla, retornar resultado parcial en lugar de crash completo

**Testing esperado**:
```bash
# Aplicar ICA + Wavelets
# Ver logs detallados:
# 📍 Fase 1: Aplicando ICA
#   → Entrada: temp_step_0_input.npy ✅
#   → Salida: temp_step_0_output/... ✅
# 📍 Fase 2: Aplicando Wavelets
#   → Entrada: temp_step_1_input.npy ✅
#   → Salida: temp_output/... ✅
```

**Qué soluciona**: Visibilidad completa de qué paso falla
**Cómo lo hace**: Logging granular + manejo de errores robusto

---

#### PASO 1.3: Validar pipeline completo con cache
**Objetivo**: Asegurar que el sistema de cache no interfiere con el pipeline

**Cambios**:
- Verificar que el hash del pipeline incluye TODOS los pasos (filtros + transforms)
- Si el cache es inválido, forzar recalcular pipeline completo
- Agregar flag `force_recalculate` para testing

**Testing esperado**:
```bash
# Aplicar ICA + Wavelets por primera vez → Calcular todo
# Aplicar ICA + Wavelets segunda vez → Usar cache ⚡
# Cambiar parámetro de ICA → Invalidar cache, recalcular
```

**Qué soluciona**: Asegura que el pipeline siempre se aplica completamente
**Cómo lo hace**: Sistema de cache con invalidación correcta

---

## 🎯 PROBLEMA 2: Reorganizar Página Modelado P300

### Diagnóstico
- **Situación actual**: Step indicator está arriba, controles de navegación/clase están en `create_navigation_controls()`
- **Objetivo**: Mover controles de navegación de canales y botones de clase al slide de "Metadata"
- **Ubicación**: `app/pages/modelado_p300.py` líneas 86-500

### Solución: 2 pasos incrementales

#### PASO 2.1: Crear componente separado para controles de metadata
**Objetivo**: Separar controles de navegación/clase en un componente reutilizable

**Cambios**:
- Crear función `create_metadata_controls(meta)` que retorna:
  - Información del dataset (nombre, clases, canales, frecuencia)
  - Chips de clases (clickeables para seleccionar)
  - Navegación de canales (← Anteriores | Siguientes →)
- Los chips de clase seleccionada se agrandan (transform: scale(1.1))

**Testing esperado**:
```bash
# Ejecutar app, ir a /p300
# Ver que los controles aparecen en el lugar correcto
# Clic en chip de clase → Se agranda y cambia visualización
# Navegación de canales funciona igual que antes
```

**Qué soluciona**: Separa lógica de metadata en componente reutilizable
**Cómo lo hace**: Extrae UI a función independiente
**No rompe**: Los callbacks existentes siguen funcionando (mismos IDs)

---

#### PASO 2.2: Integrar controles en slide de metadata
**Objetivo**: Colocar el componente en el slide correcto

**Cambios**:
- Modificar layout de slide "Metadata" para incluir `create_metadata_controls()`
- Remover controles duplicados del layout principal
- Mantener todos los IDs exactamente iguales para no romper callbacks

**Testing esperado**:
```bash
# Ejecutar app, ir a /p300
# Cambiar entre slides → Ver controles en slide "Metadata"
# Todos los callbacks funcionan: filtro de clase, navegación, etc.
```

**Qué soluciona**: UI más organizada y lógica
**Cómo lo hace**: Mueve componentes sin cambiar IDs
**No rompe**: Callbacks existentes funcionan sin modificación

---

## 🎯 PROBLEMA 3: Visualización de Wavelets en Filtros

### Diagnóstico
- **Síntoma**: En filtros.py, la transformada wavelet se ve como una línea plana
- **Causa probable**: Los datos de wavelet son 3D (ventanas) pero el plot espera 2D
- **Ubicación**: `app/pages/filtros.py`, clientside callback de render

### Solución: 2 pasos incrementales

#### PASO 3.1: Detectar datos 3D en callback clientside
**Objetivo**: Identificar cuándo la transformada es 3D y ajustar visualización

**Cambios**:
- En clientside callback de `filtros.py`:
  - Detectar si `filteredData.matrix` es 3D (tiene .ndim == 3 o shape[0] es array)
  - Si es 3D, convertir a 2D para visualización:
    - Opción A: Mostrar solo la primera ventana
    - Opción B: Promediar todas las ventanas
    - Opción C: Aplanar (concatenar ventanas)

**Testing esperado**:
```bash
# Aplicar solo ICA → Se ve bien (2D)
# Aplicar solo Wavelets → Se ve bien (3D → 2D convertido)
# Aplicar ICA + Wavelets → Se ve bien
```

**Qué soluciona**: Visualización correcta de datos venteados
**Cómo lo hace**: Detección de dimensionalidad + conversión

---

#### PASO 3.2: Agregar indicador visual de datos 3D
**Objetivo**: Informar al usuario que está viendo datos venteados

**Cambios**:
- Si los datos son 3D, agregar anotación en el plot:
  - "Datos venteados (mostrando ventana 1 de N)"
  - Posición: esquina superior derecha
- Color diferente para datos 3D vs 2D

**Testing esperado**:
```bash
# Aplicar Wavelets → Ver anotación "Datos venteados (ventana 1 de 100)"
# Aplicar ICA → No ver anotación (datos 2D normales)
```

**Qué soluciona**: Claridad sobre qué tipo de datos se visualizan
**Cómo lo hace**: Anotación condicional en plot

---

## 🎯 PROBLEMA 4: Load Dataset Genérico

### Diagnóstico
- **Situación actual**: `Dataset.load_dataset()` está hardcodeado para Inner Speech
  - Usa IDs de eventos específicos (31=arriba, 32=abajo, etc.)
  - Función `_inner_speech_cues()` busca eventos específicos (15, 22, 16)
- **Objetivo**: Hacer que funcione con cualquier dataset BDF/EDF

### Solución: 4 pasos incrementales

#### PASO 4.1: Crear sistema de detección de formato
**Objetivo**: Detectar automáticamente si el dataset usa el formato Inner Speech o es genérico

**Cambios**:
- Crear función `_detect_dataset_format(events)`:
  - Busca event IDs específicos de Inner Speech (15, 22, 31-34)
  - Si encuentra → retorna "inner_speech"
  - Si no → retorna "generic"
- Agregar parámetro `format="auto"` a `load_dataset()`

**Testing esperado**:
```bash
# Cargar dataset Inner Speech → Detecta "inner_speech" ✅
# Cargar otro dataset BDF → Detecta "generic" ✅
```

**Qué soluciona**: Identifica automáticamente el tipo de dataset
**Cómo lo hace**: Inspección de event IDs

---

#### PASO 4.2: Implementar extracción genérica de eventos
**Objetivo**: Extraer eventos sin asumir IDs específicos

**Cambios**:
- Crear función `_generic_event_extraction(raw, events)`:
  - Lista TODOS los event IDs únicos
  - Asigna labels genéricos: "Evento_1", "Evento_2", etc.
  - Permite al usuario mapear IDs → nombres después
- Guardar mapping en `dataset_metadata.json`:
  ```json
  {
    "event_id_mapping": {
      "31": "arriba",  // Inner Speech
      "1": "Evento_1"  // Genérico
    }
  }
  ```

**Testing esperado**:
```bash
# Cargar dataset genérico
# Ver en metadata: event_id_mapping con todos los IDs encontrados
# Poder editar mapping manualmente si es necesario
```

**Qué soluciona**: Extrae eventos de cualquier dataset
**Cómo lo hace**: Usa todos los IDs únicos encontrados

---

#### PASO 4.3: Ajustar generación de archivos auxiliares
**Objetivo**: Generar Events, Labels, Raw para cualquier dataset

**Cambios**:
- Modificar `load_dataset()`:
  - Si format="inner_speech" → Usa lógica actual
  - Si format="generic":
    - Extrae epochs para cada event ID único
    - Crea carpeta `Events/` con archivos por ID
    - Genera `labels.npy` con IDs en lugar de nombres
    - Permite configurar parámetros de epoching (tmin, tmax)

**Testing esperado**:
```bash
# Cargar dataset genérico con 5 event IDs
# Ver carpeta Events/ con 5 subcarpetas
# Ver labels.npy con valores correspondientes
# Metadata JSON generado correctamente
```

**Qué soluciona**: Genera estructura completa para datasets custom
**Cómo lo hace**: Extracción genérica de epochs + metad

ata

---

#### PASO 4.4: Crear UI para configurar datasets custom
**Objetivo**: Permitir al usuario configurar parámetros de carga

**Cambios**:
- En página `cargar_datos.py`, agregar modal/formulario:
  - Parámetros de epoching (tmin, tmax, baseline)
  - Mapping manual de event IDs → nombres de clase
  - Selección de canales a incluir
- Guardar configuración en JSON para reutilizar

**Testing esperado**:
```bash
# Subir dataset custom
# Abrir modal de configuración
# Ajustar tmin=-0.2, tmax=1.0
# Mapear ID 1 → "Clase_A", ID 2 → "Clase_B"
# Generar dataset → Funciona con configuración custom
```

**Qué soluciona**: Flexibilidad total para datasets custom
**Cómo lo hace**: UI interactiva + configuración persistente

---

## 📊 ORDEN DE EJECUCIÓN RECOMENDADO

### Semana 1: Pipeline + P300
1. PASO 1.1 - Arreglar flujo archivos pipeline
2. PASO 1.2 - Logging y errores pipeline
3. PASO 1.3 - Validar cache pipeline
4. Probar ICA + Wavelets → Debe funcionar ✅

### Semana 2: UI + Visualización
5. PASO 2.1 - Componente controles metadata
6. PASO 2.2 - Integrar en slide
7. PASO 3.1 - Detectar datos 3D
8. PASO 3.2 - Indicador visual 3D
9. Probar todas las transformadas → Visualizan bien ✅

### Semana 3: Load Dataset Genérico
10. PASO 4.1 - Detección de formato
11. PASO 4.2 - Extracción genérica
12. PASO 4.3 - Generación archivos
13. PASO 4.4 - UI configuración
14. Probar con dataset custom → Funciona ✅

---

## ✅ CRITERIOS DE ÉXITO POR PROBLEMA

### Problema 1 (Pipeline)
- ✅ ICA + Wavelets funciona sin errores
- ✅ Logs muestran cada paso claramente
- ✅ Cache funciona correctamente

### Problema 2 (P300 UI)
- ✅ Controles en slide correcto
- ✅ Callbacks funcionan sin cambios
- ✅ Chips de clase se agrandan al seleccionar

### Problema 3 (Visualización)
- ✅ Wavelets se visualizan correctamente
- ✅ Datos 3D convierten a 2D
- ✅ Indicador visual claro

### Problema 4 (Load Custom)
- ✅ Detecta formato automáticamente
- ✅ Extrae eventos de cualquier dataset
- ✅ Genera estructura completa
- ✅ UI permite configurar parámetros

---

## 🚨 NO ROMPER (Validaciones)

Después de cada paso, verificar:
- ✅ Pipeline existente sigue funcionando
- ✅ Historial de filtros/transforms se guarda
- ✅ Callbacks no cambian de comportamiento
- ✅ Stores mantienen mismos IDs
- ✅ Cache sigue siendo válido

---

**Siguiente acción**: Empezar con PASO 1.1 - Arreglar flujo archivos pipeline
