"""
Componente reutilizable para entrenamiento local de modelos.
Incluye selección de subsets, creación de nuevos subsets y entrenamiento.
"""

from dash import html, dcc, callback, Input, Output, State, no_update, MATCH, ALL
import dash_bootstrap_components as dbc
from pathlib import Path
import json
from typing import Optional, Dict, Any, List


def create_local_training_section(model_identifier: str) -> html.Div:
    """
    Crea la sección de entrenamiento local con selector de subsets y creación.

    Args:
        model_identifier: Identificador único del modelo (ej: "LSTM", "GRU", "CNN")

    Returns:
        Componente Dash con selector, creación y entrenamiento
    """
    return html.Div([
        # Store para el subset seleccionado
        dcc.Store(id={"type": "local-selected-subset", "model": model_identifier}, storage_type='session'),

        # Store para lista de subsets disponibles
        dcc.Store(id={"type": "local-subsets-list", "model": model_identifier}),

        # Store para estado del entrenamiento local
        dcc.Store(id={"type": "local-training-status", "model": model_identifier}),

        # Store para métricas resultantes
        dcc.Store(id={"type": "local-training-metrics", "model": model_identifier}),
        
        # Store para estado de validación de la configuración del modelo
        dcc.Store(id={"type": "model-validation-status", "model": model_identifier}, data=False),

        # Título de la sección
        html.H5([
            html.I(className="fas fa-laptop-code me-2"),
            "Entrenamiento Local"
        ], className="mb-3", style={"color": "white", "fontWeight": "600"}),
        
        # Mensaje de advertencia si no está validado
        html.Div(
            id={"type": "local-training-disabled-msg", "model": model_identifier},
            className="mb-3"
        ),

        # Contenedor principal (se deshabilita si no hay validación)
        html.Div([
            # Selector de subset existente
            html.Div([
            html.Label([
                html.I(className="fas fa-database me-2", style={"fontSize": "12px"}),
                "Seleccionar Subset"
            ], style={"color": "rgba(255,255,255,0.9)", "fontSize": "13px", "fontWeight": "500"}),
            
            dcc.Dropdown(
                id={"type": "local-subset-selector", "model": model_identifier},
                placeholder="Cargando subsets disponibles...",
                style={
                    "backgroundColor": "#FFFFFF",
                    "color": "#000000",
                    "border": "1px solid color-mix(in srgb, var(--color-4) 40%, transparent)",
                    "borderRadius": "8px",
                    "fontSize": "14px"
                },
                className="mb-2"
            ),
            
            html.Small(
                "Solo se muestran subsets compatibles con el experimento actual",
                className="text-muted d-block",
                style={"fontSize": "11px", "marginTop": "4px"}
            )
        ], className="mb-3"),

        # Área de información del subset seleccionado
        html.Div(
            id={"type": "local-subset-info", "model": model_identifier},
            className="mb-3"
        ),

        # Divisor
        html.Hr(style={"margin": "20px 0", "borderTop": "1px solid rgba(255,255,255,0.15)"}),

        # Sección de creación de nuevo subset
        html.Div([
            html.Div([
                html.H6([
                    html.I(className="fas fa-plus-circle me-2"),
                    "Crear Nuevo Subset"
                ], className="mb-3", style={"color": "white", "fontSize": "15px", "fontWeight": "600"}),
                
                # Botón para expandir/colapsar
                dbc.Button(
                    [
                        html.I(className="fas fa-chevron-down me-2"),
                        "Mostrar opciones"
                    ],
                    id={"type": "toggle-subset-creation", "model": model_identifier},
                    color="link",
                    size="sm",
                    className="p-0",
                    style={"color": "var(--color-5)", "textDecoration": "none", "fontSize": "13px"}
                )
            ], className="d-flex justify-content-between align-items-center mb-2"),
            
            # Formulario de creación (colapsable)
            dbc.Collapse(
                html.Div([
                    # Porcentaje
                    html.Div([
                        html.Label("Porcentaje del dataset", style={"fontSize": "13px", "color": "white"}),
                        dbc.Input(
                            id={"type": "subset-percentage", "model": model_identifier},
                            type="number",
                            min=1,
                            max=100,
                            value=10,
                            step=1,
                            style={"fontSize": "14px", "height": "38px"}
                        ),
                        html.Small("Cantidad de eventos a usar (1-100%)", className="text-muted", style={"fontSize": "11px"})
                    ], className="mb-3"),
                    
                    # Train/Test split
                    html.Div([
                        html.Label("División train/test (%)", style={"fontSize": "13px", "color": "white"}),
                        dbc.Input(
                            id={"type": "subset-train-split", "model": model_identifier},
                            type="number",
                            min=1,
                            max=99,
                            value=80,
                            step=5,
                            style={"fontSize": "14px", "height": "38px"}
                        ),
                        html.Small("Porcentaje para entrenamiento (resto para test)", className="text-muted", style={"fontSize": "11px"})
                    ], className="mb-3"),
                    
                    # Seed
                    html.Div([
                        html.Label("Semilla aleatoria", style={"fontSize": "13px", "color": "white"}),
                        dbc.Input(
                            id={"type": "subset-seed", "model": model_identifier},
                            type="number",
                            value=42,
                            step=1,
                            style={"fontSize": "14px", "height": "38px"}
                        ),
                        html.Small("Para reproducibilidad", className="text-muted", style={"fontSize": "11px"})
                    ], className="mb-3"),
                    
                    # Botón de creación
                    dbc.Button(
                        [
                            html.I(className="fas fa-magic me-2"),
                            "Generar Subset"
                        ],
                        id={"type": "btn-create-subset", "model": model_identifier},
                        color="info",
                        className="w-100",
                        style={"fontSize": "14px", "height": "42px", "fontWeight": "600"}
                    ),
                    
                    # Área de resultado de creación
                    html.Div(
                        id={"type": "subset-creation-result", "model": model_identifier},
                        className="mt-3"
                    )
                ], className="p-3", style={
                    "backgroundColor": "rgba(0,0,0,0.2)",
                    "borderRadius": "8px",
                    "border": "1px solid rgba(255,255,255,0.1)"
                }),
                id={"type": "collapse-subset-creation", "model": model_identifier},
                is_open=False
            )
        ], className="mb-3"),

        # Divisor
        html.Hr(style={"margin": "20px 0", "borderTop": "1px solid rgba(255,255,255,0.15)"}),

        # Botón de entrenamiento local
        html.Div([
            dbc.Button(
                [
                    html.I(className="fas fa-play-circle me-2"),
                    "Entrenar Localmente"
                ],
                id={"type": "btn-local-training", "model": model_identifier},
                color="success",
                size="lg",
                disabled=True,
                className="w-100",
                style={
                    "fontSize": "16px",
                    "fontWeight": "600",
                    "height": "50px",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"
                }
            ),
            html.Small(
                "Selecciona un subset para habilitar el entrenamiento",
                id={"type": "local-training-hint", "model": model_identifier},
                className="text-muted mt-2 d-block text-center",
                style={"fontSize": "11px"}
            )
        ], className="mb-4"),

        # Área de estado de entrenamiento
        html.Div(
            id={"type": "local-training-status-display", "model": model_identifier},
            className="mt-3"
        ),

        # Área de visualización de métricas
        dcc.Loading(
            id={"type": "local-metrics-loading", "model": model_identifier},
            type="circle",
            fullscreen=False,
            color="#28a745",
            children=[
                html.Div(
                    id={"type": "local-metrics-display", "model": model_identifier},
                    className="mt-4"
                )
            ]
        )
        ], id={"type": "local-training-content", "model": model_identifier})
        
    ], className="local-training-section", style={
        "backgroundColor": "rgba(0,0,0,0.15)",
        "padding": "20px",
        "borderRadius": "12px",
        "border": "1px solid rgba(255,255,255,0.1)"
    })


def _load_available_subsets(dataset_name: str, experiment_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Carga subsets disponibles y filtra los compatibles con el experimento actual.

    Args:
        dataset_name: Nombre del dataset
        experiment_config: Configuración del experimento actual (filtros, transforms)

    Returns:
        Lista de subsets compatibles con metadata
    """
    aux_path = Path("Aux") / dataset_name / "generated_datasets"
    
    if not aux_path.exists():
        return []
    
    subsets = []
    
    # Iterar sobre subdirectorios (timestamps)
    for subset_dir in sorted(aux_path.iterdir(), reverse=True):
        if not subset_dir.is_dir():
            continue
        
        metadata_file = subset_dir / "metadata.json"
        if not metadata_file.exists():
            continue
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Verificar compatibilidad con experimento actual
            # (comparar filtros y transforms)
            is_compatible = _check_compatibility(metadata, experiment_config)
            
            if is_compatible:
                subsets.append({
                    "path": str(subset_dir),
                    "timestamp": metadata.get("timestamp", "unknown"),
                    "n_train": metadata.get("n_train_events", 0),
                    "n_test": metadata.get("n_test_events", 0),
                    "classes": metadata.get("classes", []),
                    "metadata": metadata
                })
        
        except Exception as e:
            print(f"Error cargando metadata de {subset_dir}: {e}")
            continue
    
    return subsets


def _check_compatibility(subset_metadata: Dict[str, Any], experiment_config: Dict[str, Any]) -> bool:
    """
    Verifica si un subset es compatible con el experimento actual.

    Args:
        subset_metadata: Metadata del subset
        experiment_config: Configuración del experimento actual

    Returns:
        True si son compatibles
    """
    # Extraer configuración del snapshot del subset
    snapshot = subset_metadata.get("experiment_snapshot", {})
    
    # Comparar filtros (lista de diccionarios)
    subset_filters = snapshot.get("filters", [])
    current_filters = experiment_config.get("filters", [])
    
    # Comparar transforms (diccionario)
    subset_transform = snapshot.get("transform", {})
    current_transform = experiment_config.get("transform", {})
    
    # Compatibilidad: mismo número de filtros y misma transformación
    filters_match = len(subset_filters) == len(current_filters)
    
    # Para transforms, comparar nombres de las claves
    subset_transform_name = list(subset_transform.keys())[0] if subset_transform else None
    current_transform_name = list(current_transform.keys())[0] if current_transform else None
    
    transform_match = subset_transform_name == current_transform_name
    
    return filters_match and transform_match


def _format_class_distribution(class_distribution: Dict[str, Dict[str, Any]]) -> html.Div:
    """
    Formatea la distribución de clases para mostrar en la UI.

    Args:
        class_distribution: Diccionario con distribución por clase
            {
                "clase_name": {
                    "total_selected": int,
                    "train": int,
                    "test": int,
                    "train_pct": float,
                    "test_pct": float
                }
            }

    Returns:
        Componente Dash con distribución de clases
    """
    if not class_distribution:
        return None

    # Crear items de distribución en grid 2 columnas
    distribution_items = []
    for class_name, stats in sorted(class_distribution.items()):
        distribution_items.append(
            html.Div([
                html.Div(
                    class_name,
                    style={
                        "fontSize": "11px",
                        "fontWeight": "600",
                        "color": "var(--color-5)",
                        "marginBottom": "2px"
                    }
                ),
                html.Div(
                    f"Train: {stats['train']} ({stats['train_pct']:.1f}%) | Test: {stats['test']} ({stats['test_pct']:.1f}%)",
                    style={
                        "fontSize": "10px",
                        "opacity": "0.8",
                        "color": "rgba(255,255,255,0.9)"
                    }
                )
            ], style={
                "padding": "6px 8px",
                "backgroundColor": "rgba(255,255,255,0.05)",
                "borderRadius": "6px",
                "border": "1px solid rgba(255,255,255,0.1)"
            })
        )

    return html.Div([
        html.Small("Distribución por clase (train/test):", className="text-muted", style={"fontSize": "11px"}),
        html.Div(
            distribution_items,
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "8px",
                "marginTop": "6px"
            }
        )
    ], className="mb-3")


def _format_subset_info(metadata: Dict[str, Any]) -> html.Div:
    """
    Formatea la información de un subset para mostrar en la UI.

    Args:
        metadata: Metadata del subset

    Returns:
        Componente Dash con información del subset
    """
    snapshot = metadata.get("experiment_snapshot", {})
    
    # Información del pipeline
    filters_count = len(snapshot.get("filters", []))
    transform_dict = snapshot.get("transform", {})
    transform_name = list(transform_dict.keys())[0] if transform_dict else "Ninguna"
    
    # Obtener información de shapes y clases si está disponible
    classes = metadata.get("classes", [])
    train_shape = metadata.get("train_shape", [])
    test_shape = metadata.get("test_shape", [])
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="fas fa-info-circle me-2", style={"color": "var(--color-5)"}),
                html.Strong("Información del Subset", style={"fontSize": "14px"})
            ], className="mb-3"),
            
            # Stats principales
            html.Div([
                html.Div([
                    html.Small("Train", className="text-muted", style={"fontSize": "11px"}),
                    html.Div(str(metadata.get("n_train_events", 0)), style={"fontSize": "20px", "fontWeight": "700", "color": "var(--color-5)"})
                ], className="text-center", style={"flex": "1"}),
                
                html.Div([
                    html.Small("Test", className="text-muted", style={"fontSize": "11px"}),
                    html.Div(str(metadata.get("n_test_events", 0)), style={"fontSize": "20px", "fontWeight": "700", "color": "var(--color-5)"})
                ], className="text-center", style={"flex": "1"}),
                
                html.Div([
                    html.Small("Clases", className="text-muted", style={"fontSize": "11px"}),
                    html.Div(str(len(classes)), style={"fontSize": "20px", "fontWeight": "700", "color": "var(--color-5)"})
                ], className="text-center", style={"flex": "1"})
            ], className="d-flex mb-3", style={"gap": "15px"}),
            
            # Clases disponibles
            html.Div([
                html.Small("Clases disponibles:", className="text-muted", style={"fontSize": "11px"}),
                html.Div(
                    ", ".join(classes[:4]) + ("..." if len(classes) > 4 else ""),
                    style={"fontSize": "12px", "color": "rgba(255,255,255,0.9)", "marginTop": "2px"}
                )
            ], className="mb-3") if classes else None,

            # Distribución de clases por split
            _format_class_distribution(metadata.get("class_distribution", {})),

            # Pipeline info
            html.Div([
                html.Div([
                    html.I(className="fas fa-filter me-2", style={"fontSize": "11px"}),
                    html.Small(f"Filtros: {filters_count}", style={"fontSize": "12px"})
                ], className="mb-1"),
                
                html.Div([
                    html.I(className="fas fa-wand-magic-sparkles me-2", style={"fontSize": "11px"}),
                    html.Small(f"Transform: {transform_name}", style={"fontSize": "12px"})
                ], className="mb-1"),
                
                html.Div([
                    html.I(className="fas fa-shapes me-2", style={"fontSize": "11px"}),
                    html.Small(f"Train shape: {train_shape}" if train_shape else "Shape: N/A", style={"fontSize": "12px"})
                ])
            ], style={"color": "rgba(255,255,255,0.8)"})
        ])
    ], style={
        "backgroundColor": "rgba(0,0,0,0.3)",
        "border": "1px solid rgba(255,255,255,0.15)",
        "borderRadius": "8px"
    })


# ============================================
# CALLBACKS
# ============================================

@callback(
    Output({"type": "collapse-subset-creation", "model": MATCH}, "is_open"),
    Output({"type": "toggle-subset-creation", "model": MATCH}, "children"),
    Input({"type": "toggle-subset-creation", "model": MATCH}, "n_clicks"),
    State({"type": "collapse-subset-creation", "model": MATCH}, "is_open"),
    prevent_initial_call=True
)
def toggle_subset_creation_form(n_clicks, is_open):
    """Toggle del formulario de creación de subset."""
    if not n_clicks:
        return no_update, no_update
    
    new_state = not is_open
    
    if new_state:
        button_content = [
            html.I(className="fas fa-chevron-up me-2"),
            "Ocultar opciones"
        ]
    else:
        button_content = [
            html.I(className="fas fa-chevron-down me-2"),
            "Mostrar opciones"
        ]
    
    return new_state, button_content


@callback(
    Output({"type": "local-training-disabled-msg", "model": MATCH}, "children"),
    Output({"type": "local-training-content", "model": MATCH}, "style"),
    Input({"type": "model-validation-status", "model": MATCH}, "data"),
    prevent_initial_call=False
)
def control_local_training_availability(is_validated):
    """
    Controla la disponibilidad de la sección de entrenamiento local.
    Solo se habilita si la configuración del modelo ha sido probada exitosamente.
    """
    if not is_validated:
        # Mostrar mensaje de advertencia y deshabilitar contenido
        warning_msg = dbc.Alert([
            html.I(className="fas fa-exclamation-circle me-2"),
            html.Div([
                html.Strong("Configuración no validada"),
                html.Br(),
                html.Small("Debes probar la configuración del modelo antes de entrenar localmente.", 
                          style={"fontSize": "12px"}),
                html.Br(),
                html.Small("Usa el botón 'Probar Configuración' en la sección superior.",
                          style={"fontSize": "12px", "opacity": "0.8"})
            ])
        ], color="warning", className="mb-3", style={"fontSize": "14px"})
        
        content_style = {
            "pointerEvents": "none",
            "opacity": "0.4",
            "filter": "grayscale(50%)"
        }
        
        return warning_msg, content_style
    else:
        # Habilitar contenido
        return None, {}


@callback(
    Output({"type": "local-subsets-list", "model": MATCH}, "data"),
    Output({"type": "local-subset-selector", "model": MATCH}, "options"),
    Output({"type": "local-subset-selector", "model": MATCH}, "placeholder"),
    Input({"type": "local-subset-selector", "model": MATCH}, "id"),
    State("selected-dataset", "data"),
    prevent_initial_call=False
)
def load_available_subsets(selector_id, selected_dataset):
    """
    Carga la lista de subsets disponibles compatibles con el experimento actual.
    """
    if not selected_dataset:
        return [], [], "No hay dataset seleccionado"
    
    try:
        # Cargar configuración del experimento actual
        from backend.classes.Experiment import Experiment
        
        experiment = Experiment._load_latest_experiment()
        
        experiment_config = {
            "filters": experiment.filters or [],
            "transform": {}
        }
        
        # Obtener transform según el tipo de modelo
        # (aquí simplificamos, puedes ajustar según necesites)
        if experiment.transform and len(experiment.transform) > 0:
            last_transform = experiment.transform[-1]
            for k, v in last_transform.items():
                if k not in ["id", "dimensionality_change"]:
                    experiment_config["transform"][k] = v
                    break
        
        # Cargar subsets
        subsets = _load_available_subsets(selected_dataset, experiment_config)
        
        if not subsets:
            return [], [], "No hay subsets compatibles disponibles"
        
        # Crear opciones para el dropdown
        options = [
            {
                "label": f"{s['timestamp']} - Train: {s['n_train']}, Test: {s['n_test']}",
                "value": s["path"]
            }
            for s in subsets
        ]
        
        return subsets, options, "Selecciona un subset"
    
    except Exception as e:
        print(f"Error cargando subsets: {e}")
        return [], [], "Error cargando subsets"


@callback(
    Output({"type": "local-subset-info", "model": MATCH}, "children"),
    Output({"type": "btn-local-training", "model": MATCH}, "disabled"),
    Output({"type": "local-training-hint", "model": MATCH}, "children"),
    Input({"type": "local-subset-selector", "model": MATCH}, "value"),
    State({"type": "local-subsets-list", "model": MATCH}, "data"),
    prevent_initial_call=True
)
def display_subset_info(selected_path, subsets_list):
    """
    Muestra información del subset seleccionado.
    """
    if not selected_path or not subsets_list:
        return (
            html.Div(),
            True,
            "Selecciona un subset para habilitar el entrenamiento"
        )
    
    # Buscar el subset seleccionado
    selected_subset = None
    for subset in subsets_list:
        if subset["path"] == selected_path:
            selected_subset = subset
            break
    
    if not selected_subset:
        return (
            dbc.Alert("Subset no encontrado", color="warning", className="mb-0"),
            True,
            "Subset no válido"
        )
    
    # Formatear información
    info_card = _format_subset_info(selected_subset["metadata"])
    
    return (
        info_card,
        False,
        "Listo para entrenar con el subset seleccionado"
    )


@callback(
    Output({"type": "subset-creation-result", "model": MATCH}, "children"),
    Output({"type": "local-subsets-list", "model": MATCH}, "data", allow_duplicate=True),
    Output({"type": "local-subset-selector", "model": MATCH}, "options", allow_duplicate=True),
    Input({"type": "btn-create-subset", "model": MATCH}, "n_clicks"),
    State({"type": "subset-percentage", "model": MATCH}, "value"),
    State({"type": "subset-train-split", "model": MATCH}, "value"),
    State({"type": "subset-seed", "model": MATCH}, "value"),
    State("selected-dataset", "data"),
    State({"type": "local-subsets-list", "model": MATCH}, "data"),
    State({"type": "classifier-type-store", "model": MATCH}, "data"),  # ✅ Agregar classifier_type
    prevent_initial_call=True
)
def create_new_subset(n_clicks, percentage, train_split, seed, selected_dataset, current_subsets, classifier_type):
    """
    Crea un nuevo subset del dataset.
    """
    if not n_clicks:
        return no_update, no_update, no_update
    
    if not selected_dataset:
        return (
            dbc.Alert(
                "No hay dataset seleccionado",
                color="danger",
                dismissable=True
            ),
            no_update,
            
            no_update
        )
    
    try:
        from backend.classes.dataset import create_subset_dataset

        # Verificar que classifier_type esté definido
        if not classifier_type:
            return (
                dbc.Alert(
                    "Error: No se pudo determinar el tipo de clasificador. Recarga la página.",
                    color="danger",
                    dismissable=True
                ),
                no_update,
                no_update
            )

        # Determinar model_type según classifier_type
        model_type = "p300" if classifier_type == "P300" else "inner"

        # Crear subset
        result = create_subset_dataset(
            dataset_name=selected_dataset,
            percentage=percentage,
            train_split=train_split,
            seed=seed,
            materialize=False,
            model_type=model_type  # ✅ Pasar model_type para filtrado y balanceo de clases
        )
        
        if result["status"] != 200:
            return (
                dbc.Alert(
                    f"Error: {result['message']}",
                    color="danger",
                    dismissable=True
                ),
                no_update,
                no_update
            )
        
        # Recargar lista de subsets
        from backend.classes.Experiment import Experiment
        experiment = Experiment._load_latest_experiment()
        experiment_config = {
            "filters": experiment.filters or [],
            "transform": {}
        }
        
        if experiment.transform and len(experiment.transform) > 0:
            last_transform = experiment.transform[-1]
            for k, v in last_transform.items():
                if k not in ["id", "dimensionality_change"]:
                    experiment_config["transform"][k] = v
                    break
        
        updated_subsets = _load_available_subsets(selected_dataset, experiment_config)
        
        options = [
            {
                "label": f"{s['timestamp']} - Train: {s['n_train']}, Test: {s['n_test']}",
                "value": s["path"]
            }
            for s in updated_subsets
        ]
        
        return (
            dbc.Alert(
                [
                    html.I(className="fas fa-check-circle me-2"),
                    f"Subset creado exitosamente: {result['subset_dir']}"
                ],
                color="success",
                dismissable=True,
                duration=5000
            ),
            updated_subsets,
            options
        )
    
    except Exception as e:
        return (
            dbc.Alert(
                f"Error creando subset: {str(e)}",
                color="danger",
                dismissable=True
            ),
            no_update,
            no_update
        )


@callback(
    Output({"type": "local-training-status-display", "model": MATCH}, "children"),
    Output({"type": "local-metrics-display", "model": MATCH}, "children"),
    Input({"type": "btn-local-training", "model": MATCH}, "n_clicks"),
    State({"type": "local-subset-selector", "model": MATCH}, "value"),
    State({"type": "local-subsets-list", "model": MATCH}, "data"),
    State({"type": "classifier-type-store", "model": MATCH}, "data"),  # ✅ Acceder al classifier_type
    prevent_initial_call=True
)
def run_local_training(n_clicks, selected_path, subsets_list, classifier_type):
    """
    Ejecuta el entrenamiento local con el subset seleccionado.
    
    Workflow:
    1. Cargar paths desde train_manifest.json y test_manifest.json
    2. Obtener configuración del modelo desde el experimento
    3. Instanciar el modelo
    4. Entrenar usando los paths cargados
    5. Mostrar métricas
    """
    if not n_clicks or not selected_path:
        return no_update, no_update
    
    # Buscar metadata del subset
    selected_subset = None
    for subset in subsets_list:
        if subset["path"] == selected_path:
            selected_subset = subset
            break
    
    if not selected_subset:
        return (
            dbc.Alert(
                [
                    html.I(className="fas fa-times-circle me-2"),
                    "Subset no encontrado"
                ],
                color="danger"
            ),
            no_update
        )
    
    try:
        from pathlib import Path
        import json
        from backend.classes.Experiment import Experiment
        from backend.classes.ClasificationModel.ClassifierSchemaFactory import ClassifierSchemaFactory
        
        subset_dir = Path(selected_path)
        metadata = selected_subset["metadata"]
        
        # ========== PASO 1: CARGAR PATHS DESDE MANIFESTS ==========
        train_manifest_path = subset_dir / "train_manifest.json"
        test_manifest_path = subset_dir / "test_manifest.json"
        
        if not train_manifest_path.exists() or not test_manifest_path.exists():
            return (
                dbc.Alert(
                    [
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "Error: Manifests no encontrados en el subset"
                    ],
                    color="danger"
                ),
                no_update
            )
        
        with open(train_manifest_path, 'r') as f:
            train_manifest = json.load(f)
        
        with open(test_manifest_path, 'r') as f:
            test_manifest = json.load(f)
        
        # Extraer paths de eventos desde manifests
        # Los manifests son listas directas, no dicts con key "events"
        event_paths_train = [event["path"] for event in train_manifest]
        event_paths_test = [event["path"] for event in test_manifest]
        event_labels_train = [event["class"] for event in train_manifest]
        event_labels_test = [event["class"] for event in test_manifest]
        
        print(f"[LocalTraining] Cargados {len(event_paths_train)} eventos train, {len(event_paths_test)} eventos test")
        print(f"[LocalTraining] Aplicando pipeline de transformaciones...")

        # ========== PASO 1.5: APLICAR PIPELINE A EVENTOS ==========
        # Los eventos en el manifest son paths a archivos .npy originales
        # Necesitamos aplicar el pipeline (filters + transforms) para obtener las ventanas

        # Determinar model_type para el pipeline
        # IMPORTANTE: Usar el classifier_type del modelo, NO el nombre del dataset
        # Esto asegura que el pipeline use el mismo esquema de etiquetado que el modelo espera

        # Usar classifier_type inyectado desde el Store del componente
        # ❌ NO usar fallback - si falta classifier_type, es un error de configuración
        if not classifier_type:
            return dbc.Alert(
                "Error: No se pudo determinar el tipo de clasificador para el entrenamiento. Recarga la página.",
                color="danger",
                dismissable=True
            ), no_update

        # Mapear classifier_type a model_type para pipeline
        model_type_for_pipeline = "p300" if classifier_type == "P300" else "inner"

        # ✅ Determinar todas las clases disponibles para mapeo consistente
        all_classes_available = sorted(set(event_labels_train + event_labels_test))

        dataset_name = metadata.get("dataset_name", "")

        print(f"\n{'='*70}")
        print(f"[LocalTraining] CONFIGURACIÓN DE MODELO")
        print(f"{'='*70}")
        print(f"  Dataset: {dataset_name}")
        print(f"  classifier_type: {classifier_type}")
        print(f"  model_type_for_pipeline: {model_type_for_pipeline}")
        print(f"  Clases disponibles: {all_classes_available}")
        print(f"{'='*70}\n")

        def apply_pipeline_to_events(event_paths, labels, split_name):
            """Aplica el pipeline completo a una lista de eventos y retorna paths a ventanas."""
            import numpy as np
            from pathlib import Path
            
            processed_paths = []
            processed_labels = []
            
            # Las etiquetas ya vienen re-etiquetadas del pipeline según model_type
            # (binario 0/1 para P300, multiclase 1/2/3... para inner)
            # NO necesitamos crear un mapeo manual aquí
            print(f"   [{split_name}] Usando etiquetas re-etiquetadas del pipeline (model_type={model_type_for_pipeline})")

            for i, (event_path, label) in enumerate(zip(event_paths, labels)):
                try:
                    if i % 10 == 0:
                        print(f"   [{split_name}] Procesando {i+1}/{len(event_paths)}...", end='\r')

                    # Aplicar pipeline completo del experimento
                    result = Experiment.apply_history_pipeline(
                        file_path=event_path,
                        force_recalculate=False,  # Usar caché si existe
                        save_intermediates=False,
                        verbose=False,
                        model_type=model_type_for_pipeline,
                        all_classes=all_classes_available  # ← Pasar clases para mapeo consistente
                    )

                    # El resultado contiene cache_path (ventana transformada) y labels_path
                    window_path = result["cache_path"]
                    labels_path = result.get("labels_path")

                    # IMPORTANTE: Las labels de la transformación DEBEN existir
                    # Verificar que el pipeline generó labels correctamente
                    if not labels_path or not Path(labels_path).exists():
                        raise FileNotFoundError(
                            f"El pipeline no generó labels para el evento {event_path}. "
                            f"Esperado: {labels_path}. "
                            f"Verifica que la transformación esté configurada correctamente."
                        )

                    # ✅ USAR DIRECTAMENTE LAS ETIQUETAS RE-ETIQUETADAS DE LA TRANSFORMACIÓN
                    # Las transformadas ya aplicaron relabel_for_model() y guardaron etiquetas numéricas
                    # No necesitamos re-mapear nada aquí, solo usar lo que generó el pipeline
                    # labels_path ya apunta a las etiquetas re-etiquetadas en formato numérico

                    # SOLO agregar a las listas si TODO fue exitoso
                    processed_paths.append(window_path)
                    processed_labels.append(labels_path)  # ✅ Usar labels_path directamente

                except Exception as e:
                    print(f"\n   ⚠️ [{split_name}] Error procesando evento {i+1}/{len(event_paths)}: {Path(event_path).name}")
                    print(f"       Error: {e}")
                    print(f"       Este evento será OMITIDO del entrenamiento")
                    continue

            print(f"\n   [{split_name}] Completado: {len(processed_paths)}/{len(event_paths)} eventos procesados exitosamente")
            print(f"   [{split_name}] Eventos omitidos: {len(event_paths) - len(processed_paths)}")
            print(f"   [{split_name}] Verificación: len(X)={len(processed_paths)}, len(y)={len(processed_labels)}")

            # Validar que ambas listas tienen la misma longitud
            if len(processed_paths) != len(processed_labels):
                raise RuntimeError(
                    f"[{split_name}] Error interno: desbalance entre datos y labels. "
                    f"X={len(processed_paths)}, y={len(processed_labels)}. "
                    f"Esto NO debería ocurrir. Contacta al desarrollador."
                )

            return processed_paths, processed_labels
        
        # Aplicar pipeline a train y test
        xTrain, yTrain = apply_pipeline_to_events(event_paths_train, event_labels_train, "TRAIN")
        xTest, yTest = apply_pipeline_to_events(event_paths_test, event_labels_test, "TEST")
        
        if len(xTrain) == 0 or len(xTest) == 0:
            return (
                dbc.Alert(
                    [
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "Error: El pipeline no pudo procesar los eventos del subset"
                    ],
                    color="danger"
                ),
                no_update
            )

        print(f"\n{'='*70}")
        print(f"[LocalTraining] RESUMEN DE DATOS PREPARADOS")
        print(f"{'='*70}")
        print(f"  TRAIN: {len(xTrain)} archivos de datos")
        print(f"  TRAIN: {len(yTrain)} archivos de labels")
        print(f"  TEST:  {len(xTest)} archivos de datos")
        print(f"  TEST:  {len(yTest)} archivos de labels")
        print(f"{'='*70}\n")

        
        # ========== PASO 2: OBTENER CONFIGURACIÓN DEL MODELO ==========
        experiment = Experiment._load_latest_experiment()
        
        print(f"[LocalTraining] DEBUG - Experimento cargado ID: {experiment.id}")
        print(f"[LocalTraining] DEBUG - P300Classifier: {experiment.P300Classifier}")
        print(f"[LocalTraining] DEBUG - innerSpeachClassifier: {experiment.innerSpeachClassifier}")
        
        # Determinar qué clasificador está configurado (P300 o InnerSpeech)
        model_instance = None
        model_name = None
        
        if experiment.P300Classifier and isinstance(experiment.P300Classifier, dict):
            print(f"[LocalTraining] DEBUG - P300Classifier keys: {list(experiment.P300Classifier.keys())}")
            for name, config in experiment.P300Classifier.items():
                print(f"[LocalTraining] DEBUG - Procesando modelo: {name}")
                print(f"[LocalTraining] DEBUG - Config type: {type(config)}")
                model_name = name
                # Reconstruir instancia del modelo
                classifier_class = ClassifierSchemaFactory.available_classifiers.get(name)
                print(f"[LocalTraining] DEBUG - Classifier class found: {classifier_class}")
                if classifier_class:
                    # Filtrar transform anidado si existe
                    config_clean = {k: v for k, v in config.items() if k != "transform"}
                    print(f"[LocalTraining] DEBUG - Config clean keys: {list(config_clean.keys())}")
                    try:
                        model_instance = classifier_class(**config_clean)
                        print(f"[LocalTraining] DEBUG - Instancia creada exitosamente")
                    except Exception as e:
                        print(f"[LocalTraining] DEBUG - Error creando instancia: {e}")
                break
        
        elif experiment.innerSpeachClassifier and isinstance(experiment.innerSpeachClassifier, dict):
            print(f"[LocalTraining] DEBUG - innerSpeachClassifier keys: {list(experiment.innerSpeachClassifier.keys())}")
            for name, config in experiment.innerSpeachClassifier.items():
                model_name = name
                classifier_class = ClassifierSchemaFactory.available_classifiers.get(name)
                if classifier_class:
                    config_clean = {k: v for k, v in config.items() if k != "transform"}
                    try:
                        model_instance = classifier_class(**config_clean)
                    except Exception as e:
                        print(f"[LocalTraining] DEBUG - Error creando instancia: {e}")
                break
        
        if not model_instance:
            print(f"[LocalTraining] DEBUG - No se pudo crear instancia del modelo")
            return (
                dbc.Alert(
                    [
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "Error: No hay modelo configurado en el experimento actual"
                    ],
                    color="danger"
                ),
                no_update
            )
        
        print(f"[LocalTraining] Modelo: {model_name}")
        
        # ========== PASO 3: ENTRENAR MODELO ==========
        status_training = dbc.Alert(
            [
                html.I(className="fas fa-spinner fa-spin me-2"),
                f"Entrenando {model_name}..."
            ],
            color="info"
        )
        
        # Obtener clase del modelo para llamar a train()
        classifier_class = ClassifierSchemaFactory.available_classifiers.get(model_name)
        
        # Determinar si el modelo acepta hiperparámetros de entrenamiento
        # Modelos basados en redes neuronales: LSTM, GRU, CNN, Transformer
        # Modelos clásicos (SVM-based): SVNN, RandomForest, etc.
        neural_network_models = ['LSTM', 'GRU', 'GRUNet', 'CNN', 'Transformer', 'EEGNet', 'DeepConvNet']
        is_neural_network = model_name in neural_network_models
        
        # Llamar a train() con parámetros apropiados según el tipo de modelo
        try:
            print(f"\n{'='*70}")
            print(f"[LocalTraining] LLAMANDO A {model_name}.train()")
            print(f"{'='*70}")
            print(f"  Parámetros:")
            print(f"    - xTrain: lista con {len(xTrain)} elementos")
            print(f"    - yTrain: lista con {len(yTrain)} elementos")
            print(f"    - xTest: lista con {len(xTest)} elementos")
            print(f"    - yTest: lista con {len(yTest)} elementos")
            if len(xTrain) > 0:
                print(f"\n  Ejemplo xTrain[0]: {xTrain[0]}")
            if len(yTrain) > 0:
                print(f"  Ejemplo yTrain[0]: {yTrain[0]}")
            print(f"{'='*70}\n")

            # Entrenar SIN guardar automáticamente (sin model_label)
            if is_neural_network:
                # Extraer hiperparámetros para redes neuronales
                epochs = getattr(model_instance, 'epochs', 10)
                batch_size = getattr(model_instance, 'batch_size', 32)
                lr = getattr(model_instance, 'learning_rate', 0.001)

                print(f"[LocalTraining] Entrenando red neuronal con epochs={epochs}, batch_size={batch_size}, lr={lr}")

                metrics = classifier_class.train(
                    model_instance,
                    xTrain=xTrain,
                    yTrain=yTrain,
                    xTest=xTest,
                    yTest=yTest,
                    metadata_train=None,
                    metadata_test=None,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    model_label=None  # NO guardamos automáticamente
                )
            else:
                # Modelos clásicos: usar fit() SIN auto-guardar
                print(f"[LocalTraining] Entrenando modelo clásico {model_name} con fit()")

                # Verificar si tiene método fit
                if hasattr(classifier_class, 'fit'):
                    result = classifier_class.fit(
                        instance=model_instance,
                        xTrain=xTrain,
                        yTrain=yTrain,
                        xTest=xTest,
                        yTest=yTest,
                        metadata_train=None,
                        metadata_test=None,
                        model_label=None,  # NO guardamos automáticamente
                        verbose=True
                    )
                    metrics = result.metrics
                else:
                    # Fallback a train si no tiene fit
                    metrics = classifier_class.train(
                        model_instance,
                        xTrain=xTrain,
                        yTrain=yTrain,
                        xTest=xTest,
                        yTest=yTest,
                        metadata_train=None,
                        metadata_test=None,
                        model_label=None  # NO guardamos automáticamente
                    )

            print(f"[LocalTraining] Entrenamiento completado")

            # ========== GUARDAR MODELO CON METADATA USANDO ModelStorage ==========
            from backend.helpers.model_storage import save_model_with_metadata

            # Preparar snapshot del experimento
            experiment_snapshot = {
                "id": experiment.id,
                "dataset": selected_subset["metadata"].get("dataset_name", "unknown"),
                "filters": experiment.filters or [],
                "transform": experiment.transform or [],
                "classifier_config": {
                    "model_name": model_name,
                    "config": model_instance.dict() if hasattr(model_instance, 'dict') else {}
                },
                "subset_info": {
                    "subset_path": selected_path,
                    "n_train_events": selected_subset["metadata"].get("n_train_events", 0),
                    "n_test_events": selected_subset["metadata"].get("n_test_events", 0),
                    "classes": selected_subset["metadata"].get("classes", [])
                }
            }

            # Preparar métricas para guardar
            if hasattr(metrics, 'model_dump'):
                metrics_dict = metrics.model_dump()
            elif hasattr(metrics, 'dict'):
                metrics_dict = metrics.dict()
            else:
                metrics_dict = {
                    "accuracy": getattr(metrics, 'accuracy', 0.0),
                    "precision": getattr(metrics, 'precision', 0.0),
                    "recall": getattr(metrics, 'recall', 0.0),
                    "f1_score": getattr(metrics, 'f1_score', 0.0),
                    "auc_roc": getattr(metrics, 'auc_roc', 0.0),
                    "confusion_matrix": getattr(metrics, 'confusion_matrix', []),
                    "loss": getattr(metrics, 'loss', []),
                    "evaluation_time": getattr(metrics, 'evaluation_time', "N/A")
                }

            # Hiperparámetros
            hyperparams = {}
            if is_neural_network:
                hyperparams = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr
                }

            # Guardar modelo con metadata completa
            saved_model_path = save_model_with_metadata(
                model_instance=model_instance,
                model_name=model_name,
                model_type=model_type_for_pipeline,
                metrics=metrics_dict,
                experiment_snapshot=experiment_snapshot,
                hyperparams=hyperparams
            )

            print(f"✅ [LocalTraining] Modelo y metadata guardados exitosamente")
            print(f"   📁 Ruta: {saved_model_path}")
            print(f"   🏷️  Tipo: {model_type_for_pipeline}")
            print(f"   🔬 Experimento: {experiment.id}")
            
            # ========== PASO 4: MOSTRAR MÉTRICAS ==========
            from app.components.CloudTrainingComponent import render_metrics_visualization

            # Mensaje de éxito con información del guardado
            status_success = dbc.Alert(
                [
                    html.I(className="fas fa-check-circle me-2"),
                    html.Div([
                        html.Strong(f"Entrenamiento de {model_name} completado exitosamente", style={"fontSize": "15px"}),
                        html.Br(),
                        html.Div([
                            html.I(className="fas fa-save me-1", style={"fontSize": "11px"}),
                            f"Modelo guardado en: {saved_model_path}"
                        ], style={"fontSize": "12px", "opacity": "0.9", "marginTop": "6px"}),
                        html.Div([
                            html.I(className="fas fa-flask me-1", style={"fontSize": "11px"}),
                            f"Experimento: {experiment.id}"
                        ], style={"fontSize": "12px", "opacity": "0.9", "marginTop": "4px"}),
                        html.Div([
                            html.I(className="fas fa-tag me-1", style={"fontSize": "11px"}),
                            f"Tipo: {model_type_for_pipeline}"
                        ], style={"fontSize": "12px", "opacity": "0.9", "marginTop": "4px"})
                    ])
                ],
                color="success"
            )

            metrics_display = render_metrics_visualization(metrics_dict)

            return status_success, metrics_display
        
        except Exception as train_error:
            print(f"[LocalTraining] ERROR durante entrenamiento: {train_error}")
            import traceback
            traceback.print_exc()
            
            return (
                dbc.Alert(
                    [
                        html.I(className="fas fa-times-circle me-2"),
                        html.Div([
                            html.Strong("Error durante el entrenamiento:"),
                            html.Br(),
                            html.Small(str(train_error), style={"fontSize": "12px"})
                        ])
                    ],
                    color="danger"
                ),
                no_update
            )
    
    except Exception as e:
        print(f"[LocalTraining] ERROR general: {e}")
        import traceback
        traceback.print_exc()
        
        return (
            dbc.Alert(
                [
                    html.I(className="fas fa-times-circle me-2"),
                    f"Error inesperado: {str(e)}"
                ],
                color="danger"
            ),
            no_update
        )
