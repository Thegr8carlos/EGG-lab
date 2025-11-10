# ⚠️ Step Indicator NO se puede mover

## Problema
El step indicator ocupa espacio arriba y el usuario quiere moverlo al slide de "Metadata del Dataset".

## Por qué NO es posible

**Arquitectura de Dash**:
- El playground (incluyendo metadata slide) se regenera cuando cambia `selected-dataset`
- El step indicator se actualiza cuando cambia `current_step` o `has_transform`
- Si el step indicator está DENTRO del playground → conflicto de callbacks

**Conflicto**:
1. Callback A actualiza `PG_WRAPPER_P300` (todo el playground)
2. Callback B actualiza `STEP_INDICATOR_ID` (dentro del playground)
3. Dash regenera el playground con datos vacíos para resolver el conflicto

## Intentos realizados
- ✅ PASO 3: Wavelets funcionando en filtros
- ❌ Mover step indicator: REVERTIDO (causa conflicto de callbacks)

## Solución actual
Step indicator permanece arriba. Es la única forma de evitar conflictos de callbacks en Dash.

**Archivos**: `modelado_p300.py` (restaurado a versión estable)
