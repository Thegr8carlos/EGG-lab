# 🐛 BUG DETECTADO: Conflicto de Callbacks en P300 Step Indicator

## 🎯 Resumen del Problema

**Síntoma**: Al navegar a la página `/modelado_p300`, la página se muestra correctamente por unos segundos, pero luego se hace un update automático que vacía el contenido de metadata.

**Causa raíz**: Conflicto entre dos callbacks que intentan controlar el mismo elemento (`STEP_INDICATOR_ID`).

---

## 📊 Evidencia Visual

### Estado Inicial (Correcto) - prueba1.png
- Metadata del Dataset: ✅ Muestra 4 clases, dataset, frecuencia, canales
- Configuración: ✅ Muestra controles y slider de canales

### Estado Final (Incorrecto) - prueba2.png
- Metadata del Dataset: ❌ "Sin metadata de clases", todos los valores en "---"
- Configuración: ❌ "Por definir" en gris

---

## 🔍 Análisis Técnico

### Cambio que Causó el Bug

En un intento de optimizar el espacio, se movió el step indicator desde la parte superior de la página hacia dentro del componente `create_navigation_controls()`:

```python
# CAMBIO PROBLEMÁTICO
def create_navigation_controls(meta: dict):
    return html.Div([
        # ❌ Step indicator movido aquí
        html.Div(
            id=STEP_INDICATOR_ID,
            children=_step_indicator("transform", False),
            ...
        ),
        # ... resto de controles
    ])
```

Este componente se pasa como `navigation_controls` a `get_playGround()`, que lo coloca dentro de la tarjeta "Configuración".

### Conflicto de Callbacks

**Callback 1: `update_playground_desc`** (línea 691)
```python
@callback(
    Output(PG_WRAPPER_P300, "children"),
    Input("selected-dataset", "data")
)
def update_playground_desc(selected_dataset):
    # Crea TODO el playground incluyendo navigation_controls
    # que contiene el STEP_INDICATOR_ID
    nav_controls = create_navigation_controls(meta)
    return get_playGround(..., navigation_controls=nav_controls)
```

**Callback 2: `update_step_indicator`** (línea 759)
```python
@callback(
    Output(STEP_INDICATOR_ID, 'children'),
    [Input(CURRENT_STEP_STORE_ID, 'data'),
     Input('has-transform-p300', 'data')]
)
def update_step_indicator(current_step, has_transform):
    # Intenta actualizar SOLO el step indicator
    return _step_indicator(current_step or "transform", has_transform)
```

### ¿Por Qué Causa el Bug?

1. **Navegas a P300** → Layout inicial se renderiza
2. **Callback 1 se ejecuta** → Crea playground completo con metadata y step indicator → Se ve BIEN (prueba1)
3. **Callback 2 se ejecuta** (unos segundos después cuando `has-transform-p300` se inicializa)
4. **Dash detecta conflicto**: El Callback 2 intenta actualizar `STEP_INDICATOR_ID`, pero ese elemento está DENTRO del output del Callback 1 (`PG_WRAPPER_P300`)
5. **Dash regenera el playground** para resolver el conflicto, pero lo hace con datos vacíos → Se ve MAL (prueba2)

### Regla de Dash Violada

**Regla**: Un elemento NO puede ser actualizado por DOS callbacks diferentes si uno de ellos es padre del otro.

En este caso:
- Callback 1 actualiza `PG_WRAPPER_P300` (padre)
- Callback 2 actualiza `STEP_INDICATOR_ID` (hijo de `PG_WRAPPER_P300`)

Esto crea un **conflicto de outputs anidados**.

---

## ✅ Solución Implementada

**Revertir el cambio**: Mover el step indicator de vuelta a la parte superior de la página.

### Cambios Realizados:

**1. Remover step indicator de `create_navigation_controls()`** (línea 231-236)
```python
def create_navigation_controls(meta: dict):
    """Crea los controles de navegación de canales y filtrado por clase"""
    # ❌ REMOVIDO: Step indicator

    return html.Div([
        # Navegación de canales
        # ... resto de controles
    ])
```

**2. Restaurar step indicator en layout principal** (línea 541-551)
```python
html.Div([
    # ✅ Step Indicator en la parte superior (fuera del playground)
    html.Div(
        id=STEP_INDICATOR_ID,
        children=_step_indicator("transform", False),
        style={
            "width": "100%",
            "padding": "10px 20px",
            "boxSizing": "border-box",
            "flexShrink": "0"
        }
    ),

    # ... resto del layout
```

---

## 🧪 Verificación

**Prueba**:
1. Navegar a `/modelado_p300`
2. Esperar 5-10 segundos
3. Verificar que la metadata NO se vacía

**✅ Resultado esperado**:
- Metadata del Dataset: Sigue mostrando las clases y datos correctos
- Step indicator: Funciona correctamente arriba
- NO hay updates automáticos que vacíen el contenido

---

## 📝 Lecciones Aprendidas

### 1. Callbacks Anidados en Dash
**Nunca** tener dos callbacks donde:
- Callback A actualiza `Output(parent, "children")`
- Callback B actualiza `Output(child_inside_parent, "children")`

Esto siempre causa conflictos.

### 2. Separación de Responsabilidades
Cada elemento debe ser controlado por UN SOLO callback. Si necesitas actualizarlo desde múltiples fuentes, usa un callback con múltiples inputs.

### 3. Testing de Callbacks
Siempre probar:
- Navegación entre páginas
- Esperar unos segundos para ver si hay callbacks asíncronos
- Verificar que no haya updates inesperados

---

## 🔄 Alternativas Consideradas

### Opción 1: Eliminar el callback `update_step_indicator`
**Pros**: Elimina el conflicto
**Contras**: El step indicator no se actualizaría cuando `has-transform-p300` cambie sin que se recargue el playground

### Opción 2: Usar un componente completamente separado
**Pros**: Más modular
**Contras**: Más complejidad, duplicación de código

### Opción 3: Combinar ambos callbacks en uno solo
**Pros**: Elimina el conflicto, mantiene funcionalidad
**Contras**: Callback más complejo, se ejecuta más frecuentemente

**Decisión**: Opción implementada (revertir) es la más simple y segura.

---

## 📊 Estado Final

- ✅ **PASO 3.1 y 3.2**: Visualización de Wavelets FUNCIONA correctamente
- ❌ **Mover Step Indicator**: REVERTIDO (causaba bug)
- ✅ Bug resuelto, página funciona correctamente

---

**Fecha**: 2025-11-09
**Versión**: 1.0
