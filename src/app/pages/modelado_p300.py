# app/pages/modelado_p300.py
import time, random, math, json
from dash import html, dcc, register_page, callback, Output, Input, State, no_update, ALL
from shared.fileUtils import get_dataset_metadata

# ← NUEVO: traemos los esquemas desde el backend al arrancar
from backend.classes.ClasificationModel.ClassifierSchemaFactory import ClassifierSchemaFactory


# ← NUEVO: cards de configuración interactiva
from app.components.model_config_cards import (
    create_config_card,
    create_model_selector_card
)

# ← NUEVO: componente de transformaciones
from app.components.RigthComlumn import get_rightColumn

# ← NUEVO: componente reusable de visualización
from app.components.TransformationViewer import get_transformation_viewer, register_transformation_callbacks

# ← NUEVO: imports para visualización (de extractores.py)
import numpy as np
from shared.fileUtils import get_dataset_metadata
import dash_bootstrap_components as dbc
from backend.classes.dataset import Dataset
from app.components.PlayGround import get_playGround
from app.components.SideBar import get_sideBar

register_page(__name__, path="/p300", name="Modelado P300")

# ---------- IDs ----------
TRAIN_CONFIG_STORE_ID  = "train-config-p300"
TRAIN_STATUS_STORE_ID  = "train-status-p300"
TRAIN_METRICS_STORE_ID = "train-metrics-p300"
TRAIN_INTERVAL_ID      = "train-interval-p300"

BTN_TRAIN_ID     = "btn-train-p300"
DD_CLASSES_ID    = "dd-classes-p300"
DD_CHANNELS_ID   = "dd-channels-p300"
IN_SPLIT_ID      = "in-split-p300"
IN_KFOLDS_ID     = "in-kfolds-p300"

STATUS_VIEW_ID   = "train-status-view-p300"
METRICS_VIEW_ID  = "train-metrics-view-p300"
DATASET_LABEL_ID = "ds-name-p300"

# ← NUEVO: layout y navegación de modelos
SCHEMAS_STORE_ID       = "classifier-schemas-store"
SELECTED_MODEL_STORE_ID= "selected-model-store"
SIDEBAR_ID             = "models-sidebar"
MAIN_VIEW_ID           = "model-config-container"

# ← NUEVO: Sistema de pasos (transformación + modelo)
CURRENT_STEP_STORE_ID = "current-step-p300"  # "transform" o "model"
TRANSFORM_VIEW_ID     = "transform-view-p300"
MODEL_VIEW_ID         = "model-view-p300"
STEP_INDICATOR_ID     = "step-indicator-p300"

# ← NUEVO: Stores para visualización de transformaciones (del extractores.py)
DATA_STORE_P300 = "signal-store-p300"
TRANSFORMED_DATA_STORE_P300 = "transformed-signal-store-p300"
PIPELINE_UPDATE_TRIGGER_P300 = "pipeline-update-trigger-p300"
AUTO_APPLY_PIPELINE_P300 = "auto-apply-pipeline-p300"
PLOTS_CONTAINER_P300 = "plots-container-p300"
CHANNEL_RANGE_STORE_P300 = "channel-range-store-p300"
SELECTED_CLASS_STORE_P300 = "selected-class-store-p300"
SELECTED_CHANNELS_STORE_P300 = "selected-channels-store-p300"
GRAPH_ID_P300 = "pg-main-plot-p300"
PG_WRAPPER_P300 = "pg-wrapper-p300"

# ---------- Carga de esquemas al inicio ----------
# Se calcula una vez al importar el módulo
CLASSIFIER_SCHEMAS = ClassifierSchemaFactory.get_all_classifier_schemas() or {}
MODEL_NAMES = list(CLASSIFIER_SCHEMAS.keys())

# ---------- UI helpers ----------
def _badge(text, kind="info"):
    return html.Span(text, className=f"badge {kind}")

def _progress(percent: float):
    pct = max(0, min(100, float(percent)))
    return html.Div([html.Div(className="progress__fill", style={"width": f"{pct:.0f}%"} )], className="progress")

# ← Indicador de pasos
def _step_indicator(current_step: str = "transform", has_transform: bool = False) -> html.Div:
    """
    Crea el indicador visual de pasos: Transformación → Modelo
    Los pasos son clickeables para navegar, pero el paso 2 solo se activa si hay transformación registrada
    """
    transform_active = current_step == "transform"
    model_active = current_step == "model"
    model_enabled = has_transform  # Solo habilitar modelo si hay transformación

    return html.Div([
        # Paso 1: Transformación (siempre clickeable)
        html.Div([
            html.Div("1", className="step-number", style={
                "background": "#4A90E2" if transform_active else "#666",
                "color": "white",
                "width": "30px",
                "height": "30px",
                "borderRadius": "50%",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontWeight": "bold",
                "fontSize": "14px"
            }),
            html.Span("Transformación", style={
                "marginLeft": "10px",
                "fontWeight": "600" if transform_active else "normal",
                "color": "#fff" if transform_active else "#999",
                "fontSize": "14px"
            })
        ], id="step-btn-transform-p300", style={
            "display": "flex",
            "alignItems": "center",
            "marginRight": "30px",
            "cursor": "pointer",
            "padding": "8px 12px",
            "borderRadius": "8px",
            "transition": "background 0.2s",
        }),

        # Separador
        html.Div("→", style={
            "fontSize": "20px",
            "color": "#666",
            "marginRight": "30px"
        }),

        # Paso 2: Modelo (solo clickeable si hay transformación)
        html.Div([
            html.Div("2", className="step-number", style={
                "background": "#4A90E2" if model_active else "#666",
                "color": "white",
                "width": "30px",
                "height": "30px",
                "borderRadius": "50%",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontWeight": "bold",
                "fontSize": "14px",
                "opacity": "1" if model_enabled else "0.5"
            }),
            html.Span("Modelo", style={
                "marginLeft": "10px",
                "fontWeight": "600" if model_active else "normal",
                "color": "#fff" if model_active else "#999",
                "fontSize": "14px",
                "opacity": "1" if model_enabled else "0.5"
            })
        ], id="step-btn-model-p300", style={
            "display": "flex",
            "alignItems": "center",
            "cursor": "pointer" if model_enabled else "not-allowed",
            "padding": "8px 12px",
            "borderRadius": "8px",
            "transition": "background 0.2s",
            "opacity": "1" if model_enabled else "0.6"
        })
    ], style={
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "padding": "20px",
        "background": "rgba(0,0,0,0.3)",
        "borderRadius": "8px",
        "marginBottom": "0px"
    })

# ← Barra lateral derecha con cards de modelos
def _models_sidebar(models: list[str]) -> html.Div:
    """Crea la barra lateral derecha con las cards de modelos."""
    model_cards = [
        create_model_selector_card(model, idx)
        for idx, model in enumerate(models)
    ]

    return html.Div([
        html.H2("Modelos de clasificación", className="right-panel-title"),
        html.Div(model_cards, className="right-panel-container")
    ], id=SIDEBAR_ID, className="models-sidebar", style={
        "height": "100%",
        "overflowY": "auto",
        "overflowX": "hidden"
    })


# ← Funciones helper copiadas de extractores.py para metadata y controles
def create_metadata_section(meta: dict):
    """Crea secciones de metadata para el playground (copiado de extractores.py)"""
    if not isinstance(meta, dict):
        return {}, {}
    classes = meta.get("classes", []) or []

    # Usar sistema centralizado de colores para consistencia
    from shared.class_colors import get_class_color
    class_color_map = {}
    for idx, label in enumerate(classes):
        class_color_map[str(label)] = get_class_color(str(label), idx)

    sfreq = (
        meta.get("sampling_frequency_hz")
        or meta.get("sfreq")
        or ((meta.get("unique_sfreqs") or [None])[0] if isinstance(meta.get("unique_sfreqs"), (list, tuple)) else None)
    )
    if isinstance(sfreq, str):
        try:
            sfreq = float(sfreq)
        except Exception:
            sfreq = None
    n_channels = (
        meta.get("n_channels")
        or len(meta.get("channel_names") or [])
        or len(meta.get("channel_name_union") or [])
        or None
    )
    custom = {
        "dataset_name": meta.get("dataset_name"),
        "num_classes": meta.get("num_classes", len(classes)),
        "sfreq": float(sfreq) if isinstance(sfreq, (int, float)) else None,
        "n_channels": int(n_channels) if isinstance(n_channels, (int, float)) else None,
        "eeg_unit": meta.get("eeg_unit", "V"),
    }
    return class_color_map, custom


def create_navigation_controls(meta: dict):
    """Crea los controles de navegación de canales y filtrado por clase"""
    classes = meta.get("classes", []) if isinstance(meta, dict) else []

    return html.Div([
        # Navegación de canales
        html.Div([
            html.Div(
                id='channel-nav-info-p300',
                children="Canales 0 - 7 de 0",
                style={
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "color": "var(--text)",
                    "marginBottom": "6px",
                    "textAlign": "center"
                }
            ),
            html.Div([
                html.Button(
                    '← Anteriores',
                    id='btn-prev-channels-p300',
                    n_clicks=0,
                    disabled=True,
                    style={
                        "padding": "3px 8px",
                        "borderRadius": "var(--radius-sm)",
                        "border": "none",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "not-allowed",
                        "fontSize": "10px",
                        "fontWeight": "500",
                        "opacity": "0.5",
                        "flex": "1"
                    }
                ),
                html.Button(
                    'Siguientes →',
                    id='btn-next-channels-p300',
                    n_clicks=0,
                    disabled=True,
                    style={
                        "padding": "3px 8px",
                        "borderRadius": "var(--radius-sm)",
                        "border": "none",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "not-allowed",
                        "fontSize": "10px",
                        "fontWeight": "500",
                        "opacity": "0.5",
                        "flex": "1"
                    }
                ),
            ], style={
                "display": "flex",
                "gap": "4px",
                "marginBottom": "12px"
            })
        ]),

        # Divisor
        html.Hr(style={
            "border": "none",
            "borderTop": "1px solid var(--border-weak)",
            "margin": "8px 0",
            "opacity": "0.4"
        }),

        # Filtro por clase en fila (sin título)
        html.Div([
            html.Div([
                html.Button(
                    'Todas',
                    id='btn-all-classes-p300',
                    n_clicks=0,
                    style={
                        "padding": "3px 6px",
                        "flex": "1",
                        "borderRadius": "var(--radius-sm)",
                        "border": "1px solid var(--accent-1)",
                        "background": "var(--accent-1)",
                        "color": "var(--text)",
                        "cursor": "pointer",
                        "fontSize": "10px",
                        "fontWeight": "500",
                        "opacity": "1",
                        "whiteSpace": "nowrap"
                    }
                ),
            ] + [
                html.Button(
                    str(cls),
                    id={'type': 'btn-filter-class-p300', 'index': idx},
                    n_clicks=0,
                    style={
                        "padding": "3px 6px",
                        "flex": "1",
                        "borderRadius": "var(--radius-sm)",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer",
                        "fontSize": "10px",
                        "fontWeight": "500",
                        "opacity": "0.8",
                        "whiteSpace": "nowrap"
                    }
                ) for idx, cls in enumerate(classes)
            ], style={
                "display": "flex",
                "gap": "4px",
                "marginBottom": "12px"
            })
        ]),

        # Divisor
        html.Hr(style={
            "border": "none",
            "borderTop": "1px solid var(--border-weak)",
            "margin": "8px 0",
            "opacity": "0.4"
        }),

    # Historial del Pipeline
        html.Div(id='pipeline-history-viewer-p300', children=[
            html.Div("Cargando historial...", style={
                "fontSize": "9px",
                "color": "var(--text-muted)",
                "textAlign": "center",
                "padding": "8px"
            })
        ]),

        # Divisor
        html.Hr(style={
            "border": "none",
            "borderTop": "1px solid var(--border-weak)",
            "margin": "8px 0",
            "opacity": "0.4"
        }),

        # Toggle Auto-aplicar Pipeline
        html.Div([
            html.Div([
                html.Span("Auto-aplicar pipeline", style={
                    "fontSize": "10px",
                    "fontWeight": "600",
                    "color": "#ddd",
                    "flex": "1"
                }),
                html.Button(
                    "ON",
                    id="toggle-auto-apply-p300",
                    n_clicks=0,
                    style={
                        "padding": "2px 8px",
                        "fontSize": "9px",
                        "borderRadius": "3px",
                        "border": "1px solid #4CAF50",
                        "background": "#4CAF50",
                        "color": "white",
                        "cursor": "pointer",
                        "fontWeight": "600"
                    }
                )
            ], style={
                "display": "flex",
                "alignItems": "center",
                "gap": "8px",
                "padding": "6px",
                "background": "#1a1a1a",
                "borderRadius": "4px",
                "border": "1px solid #333"
            })
        ], style={"marginBottom": "12px"}),

        # Divisor
        html.Hr(style={
            "border": "none",
            "borderTop": "1px solid var(--border-weak)",
            "margin": "8px 0",
            "opacity": "0.4"
        }),

        # Selector de canales específicos
        html.Div([
            # Header
            html.Div([
                html.Div("Canales específicos", style={
                    "fontSize": "10px",
                    "fontWeight": "600",
                    "color": "var(--text)",
                    "flex": "1"
                }),
                html.Div([
                    html.Button("Todos", id='btn-select-all-channels-p300', n_clicks=0, style={
                        "padding": "2px 6px",
                        "fontSize": "8px",
                        "borderRadius": "3px",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer",
                        "marginRight": "4px"
                    }),
                    html.Button("Limpiar", id='btn-clear-channels-p300', n_clicks=0, style={
                        "padding": "2px 6px",
                        "fontSize": "8px",
                        "borderRadius": "3px",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer",
                        "marginRight": "4px"
                    }),
                    html.Button("Solo EEG", id='btn-only-eeg-channels-p300', n_clicks=0, style={
                        "padding": "2px 6px",
                        "fontSize": "8px",
                        "borderRadius": "3px",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer"
                    })
                ], style={"display": "flex"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),

            # Checklist scrollable
            html.Div([
                dcc.Checklist(
                    id='checklist-channel-selection-p300',
                    options=[],
                    value=[],
                    labelStyle={
                        "display": "block",
                        "padding": "2px 4px",
                        "fontSize": "9px",
                        "cursor": "pointer"
                    },
                    inputStyle={
                        "marginRight": "6px",
                        "cursor": "pointer"
                    },
                    style={
                        "color": "var(--text)",
                        "lineHeight": "1.4"
                    }
                )
            ], style={
                "maxHeight": "150px",
                "overflowY": "auto",
                "overflowX": "hidden",
                "padding": "4px",
                "border": "1px solid var(--border-weak)",
                "borderRadius": "var(--radius-sm)",
                "background": "var(--card-bg)"
            }),

            # Contador
            html.Div(id='channel-count-display-p300', children="0 canales seleccionados", style={
                "fontSize": "8px",
                "color": "var(--text-muted)",
                "marginTop": "4px",
                "textAlign": "right"
            })
        ])
    ])


# ← Layout maestro con 3 columnas: configuración (izq) | transformaciones (centro) | modelos (der)
layout = html.Div([
    # Stores para esquemas y estado - usando storage_type='session' para persistir
    dcc.Store(id=SCHEMAS_STORE_ID, data=CLASSIFIER_SCHEMAS, storage_type='session'),
    dcc.Store(id=SELECTED_MODEL_STORE_ID, storage_type='session'),
    dcc.Store(id=CURRENT_STEP_STORE_ID, data="transform", storage_type='session'),  # Paso actual: transform o model
    dcc.Store(id="page-classifier-type", data="P300", storage_type='session'),  # Tipo de clasificador para esta página
    dcc.Store(id=TRAIN_CONFIG_STORE_ID, storage_type='session'),
    dcc.Store(id=TRAIN_STATUS_STORE_ID, storage_type='session'),
    dcc.Store(id=TRAIN_METRICS_STORE_ID, storage_type='session'),
    dcc.Interval(id=TRAIN_INTERVAL_ID, interval=1000, disabled=True),

    # ← NUEVO: Stores para visualización de transformaciones
    dcc.Store(id=DATA_STORE_P300),
    dcc.Store(id=TRANSFORMED_DATA_STORE_P300),
    dcc.Store(id=PIPELINE_UPDATE_TRIGGER_P300, data=0),
    dcc.Store(id=AUTO_APPLY_PIPELINE_P300, data=True),  # Auto-aplicar ON por defecto
    dcc.Store(id=CHANNEL_RANGE_STORE_P300, data={"start": 0, "count": 8}),
    dcc.Store(id=SELECTED_CLASS_STORE_P300, data=None),
    dcc.Store(id=SELECTED_CHANNELS_STORE_P300, data=None),
    dcc.Store(id='model-type-p300', data="p300"),  # Identificador del modelo para el pipeline
    dcc.Store(id='has-transform-p300', data=False),  # Indica si hay transformación registrada

    # Modal para mostrar JSON del historial
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="modal-history-json-title-p300")),
        dbc.ModalBody(html.Pre(id="modal-history-json-content-p300", style={
            "fontSize": "10px",
            "maxHeight": "500px",
            "overflow": "auto",
            "background": "var(--card-bg)",
            "padding": "12px",
            "borderRadius": "var(--radius-sm)",
            "color": "var(--text)"
        })),
    ], id="modal-history-json-p300", size="lg", is_open=False),

    # Contenedor principal con layout vertical: Indicador arriba + Contenido abajo
    html.Div([
        # Indicador de pasos en la parte superior
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
        # Badge de transformación aplicada (se rellena vía callback)
        html.Div(id="current-transform-badge-p300", style={
            "width": "100%",
            "padding": "4px 20px 0 20px",
            "boxSizing": "border-box",
            "fontSize": "12px",
            "color": "#aaa",
            "minHeight": "20px"
        }),
        # Historial de transformaciones (lista de pasos de tipo 'transform')
        html.Div(id="transform-history-p300", style={
            "width": "100%",
            "padding": "0 20px 10px 20px",
            "boxSizing": "border-box",
            "fontSize": "11px",
            "color": "#888",
            "minHeight": "20px"
        }),

        # Contenedor del contenido principal (cambia según el paso)
        html.Div(id=MAIN_VIEW_ID, children=[
            # PASO DE TRANSFORMACIÓN: Layout de 3 columnas (igual que extractores.py)
            html.Div([
                # Sidebar izquierdo: Data tree
                html.Div(
                    id="sidebar-wrapper-p300",
                    children=[get_sideBar("Data")],
                    className="sideBar-container",
                    style={"width": "260px", "padding": "1rem"}
                ),
                # Centro: Playground con visualización
                html.Div(
                    id=PG_WRAPPER_P300,
                    children=get_playGround(
                        "Transformación P300",
                        "Selecciona un dataset en 'Cargar Datos'",
                        {},
                        {},
                        graph_id=GRAPH_ID_P300,
                        multi=True,
                        plots_container_id=PLOTS_CONTAINER_P300
                    ),
                    style={"flex": "1", "padding": "1rem"}
                ),
                # Derecha: Transformaciones
                html.Div(
                    id="dynamic-sidebar-p300",
                    children=get_rightColumn("transformP300"),
                    style={"width": "340px", "padding": "1rem"}
                ),
            ], style={"display": "flex", "flex": "1", "overflow": "hidden"})
        ], style={
            "flex": "1",
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden"
        })
    ], style={
        "display": "flex",
        "flexDirection": "column",
        "height": "100%",
        "width": "100%"
    })
])
# Mostrar nombre de la transform aplicada actualmente
@callback(
    Output("current-transform-badge-p300", "children"),
    Input(TRANSFORMED_DATA_STORE_P300, "data")
)
def show_current_transform(transformed):
    if isinstance(transformed, dict):
        meta = transformed.get("metadata") or {}
        model_type = str((meta or {}).get("model_type", "")).lower()
        if model_type != "p300":
            return ""  # No mostrar datos de otro modelo para evitar confusión
        name = transformed.get("applied_transform_name")
        if not name:
            viz = transformed.get("viz") or {}
            if isinstance(viz, dict):
                name = viz.get("transform_name")
        if name:
            return html.Span([
                html.Span("Transformación actual del modelo (P300): ", style={"color": "#888"}),
                html.Strong(str(name), style={"color": "#fff"})
            ])
    return ""

# Historial de transformaciones aplicadas (derivado de metadata.execution_log)
@callback(
    Output("transform-history-p300", "children"),
    Input(TRANSFORMED_DATA_STORE_P300, "data")
)
def show_transform_history_p300(transformed):
    if not isinstance(transformed, dict):
        return ""
    meta = transformed.get("metadata") or {}
    if not isinstance(meta, dict):
        return ""
    model_type = str(meta.get("model_type", "")).lower()
    if model_type != "p300":
        return ""  # Filtrar solo historial del modelo P300
    log = meta.get("execution_log") or []
    if not isinstance(log, list) or not log:
        return ""
    # Filtrar pasos de tipo transform exitosos
    transform_steps = [step for step in log if isinstance(step, dict) and step.get("type") == "transform" and step.get("status") == "success"]
    if not transform_steps:
        return ""
    current_name = transformed.get("applied_transform_name")
    items = []
    for idx, step in enumerate(transform_steps, 1):
        name = step.get("name") or "(sin nombre)"
        is_current = (current_name == name)
        items.append(html.Div([
            html.Span(f"{idx}. {name}", style={
                "color": "#fff" if is_current else "#ccc",
                "fontWeight": "600" if is_current else "normal"
            }),
            html.Span(" (actual)" if is_current else "", style={"color": "#4A90E2", "fontSize": "10px", "marginLeft": "6px"})
        ], style={"marginBottom": "2px"}))
    return html.Div([
        html.Span("Historial de transformaciones (P300):", style={"display": "block", "marginBottom": "4px", "color": "#999"}),
        html.Div(items)
    ])

# ---------- Callbacks ----------

# ===== ACTUALIZACIÓN DEL PLAYGROUND CON METADATA =====

@callback(
    Output(PG_WRAPPER_P300, "children"),
    Input("selected-dataset", "data")
)
def update_playground_desc(selected_dataset):
    """Actualiza el playground con metadata del dataset seleccionado (igual que extractores.py)"""
    desc = selected_dataset or "Selecciona un dataset en 'Cargar Datos'"
    if not selected_dataset:
        return get_playGround(
            "Transformación P300",
            desc,
            {},
            {},
            graph_id=GRAPH_ID_P300,
            multi=True,
            plots_container_id=PLOTS_CONTAINER_P300
        )
    try:
        meta = get_dataset_metadata(selected_dataset)
    except Exception as e:
        return get_playGround(
            "Transformación P300",
            f"{desc} (sin metadata: {e})",
            {},
            {},
            graph_id=GRAPH_ID_P300,
            multi=True,
            plots_container_id=PLOTS_CONTAINER_P300
        )

    meta_dict, custom_dict = create_metadata_section(meta)
    nav_controls = create_navigation_controls(meta)
    return get_playGround(
        "Transformación P300",
        desc,
        meta_dict,
        custom_dict,
        graph_id=GRAPH_ID_P300,
        multi=True,
        navigation_controls=nav_controls,
        plots_container_id=PLOTS_CONTAINER_P300
    )


# ===== DETECCIÓN DE TRANSFORMACIÓN REGISTRADA =====

@callback(
    Output('has-transform-p300', 'data'),
    Input(PIPELINE_UPDATE_TRIGGER_P300, 'data')
)
def check_transform_registered(trigger):
    """Verifica si hay transformación registrada en el experimento"""
    from backend.classes.Experiment import Experiment
    try:
        exp = Experiment._load_latest_experiment()
        # Verificar si existe transformación en P300Classifier
        if exp.P300Classifier and isinstance(exp.P300Classifier, dict):
            for model_name, model_config in exp.P300Classifier.items():
                if isinstance(model_config, dict) and 'transform' in model_config:
                    return True
        return False
    except Exception:
        return False


# ===== ACTUALIZACIÓN DEL INDICADOR DE PASOS =====

@callback(
    Output(STEP_INDICATOR_ID, 'children'),
    [Input(CURRENT_STEP_STORE_ID, 'data'),
     Input('has-transform-p300', 'data')]
)
def update_step_indicator(current_step, has_transform):
    """Actualiza el indicador de pasos con el estado actual"""
    return _step_indicator(current_step or "transform", has_transform or False)


# ===== NAVEGACIÓN ENTRE PASOS POR CLIC =====

@callback(
    [
        Output(CURRENT_STEP_STORE_ID, "data"),
        Output(MAIN_VIEW_ID, "children"),
        Output("dynamic-sidebar-p300", "children")
    ],
    [Input("step-btn-transform-p300", "n_clicks"),
     Input("step-btn-model-p300", "n_clicks")],
    [State(CURRENT_STEP_STORE_ID, "data"),
     State('has-transform-p300', 'data')],
    prevent_initial_call=True
)
def navigate_steps(transform_clicks, model_clicks, current_step, has_transform):
    """Navega entre pasos al hacer clic en el indicador"""
    from dash import callback_context as ctx

    if not ctx.triggered:
        return no_update, no_update, no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Click en paso de transformación
    if button_id == "step-btn-transform-p300":
        new_step = "transform"
        new_view = html.Div([
            # Layout de 3 columnas para transformación
            html.Div([
                html.Div(
                    id="sidebar-wrapper-p300",
                    children=[get_sideBar("Data")],
                    className="sideBar-container",
                    style={"width": "260px", "padding": "1rem"}
                ),
                html.Div(
                    id=PG_WRAPPER_P300,
                    children=get_playGround(
                        "Transformación P300",
                        "Selecciona un dataset en 'Cargar Datos'",
                        {}, {},
                        graph_id=GRAPH_ID_P300,
                        multi=True,
                        plots_container_id=PLOTS_CONTAINER_P300
                    ),
                    style={"flex": "1", "padding": "1rem"}
                ),
                html.Div(
                    id="dynamic-sidebar-p300",
                    children=get_rightColumn("transformP300"),
                    style={"width": "340px", "padding": "1rem"}
                ),
            ], style={"display": "flex"})
        ])
        new_sidebar = get_rightColumn("transformP300")
        return new_step, new_view, new_sidebar

    # Click en paso de modelo (solo si hay transformación)
    elif button_id == "step-btn-model-p300" and has_transform:
        new_step = "model"
        # Vista de modelo con layout de 2 columnas: centro (mensaje) + derecha (sidebar modelos)
        new_view = html.Div([
            html.Div([
                # Columna central: Mensaje de bienvenida
                html.Div([
                    html.H3("Paso 2: Configurar Modelo", className="text-center mb-3", style={"color": "white"}),
                    html.P(
                        "Selecciona un modelo de la barra derecha para configurarlo de manera interactiva.",
                        className="text-center",
                        style={"color": "rgba(255,255,255,0.7)"}
                    ),
                    html.Div([
                        html.I(className="fas fa-brain fa-5x", style={"color": "#4A90E2"})
                    ], className="text-center mt-5")
                ], className="welcome-message", style={
                    "padding": "60px 20px",
                    "borderRadius": "8px",
                    "minHeight": "400px",
                    "flex": "1"
                }),
                # Columna derecha: Sidebar de modelos
                html.Div(
                    id="dynamic-sidebar-p300",
                    children=_models_sidebar(MODEL_NAMES),
                    style={"width": "340px", "padding": "1rem"}
                ),
            ], style={"display": "flex", "flex": "1"})
        ])
        new_sidebar = _models_sidebar(MODEL_NAMES)
        return new_step, new_view, new_sidebar

    return no_update, no_update, no_update


# Los callbacks de navegación legacy se eliminaron - ahora se usa navegación por clic en pasos


# ===== CALLBACKS ORIGINALES DE MODELOS =====

@callback(
    [Output(DD_CLASSES_ID, "options"),
     Output(DD_CHANNELS_ID, "options"),
     Output(DATASET_LABEL_ID, "children")],
    Input("selected-dataset", "data")
)
def fill_meta_options(selected_dataset):
    if not selected_dataset:
        return [], [], "Dataset: —"
    try:
        meta = get_dataset_metadata(selected_dataset)
        classes = [{"label": str(c), "value": str(c)} for c in (meta.get("classes") or [])]
        chans   = [{"label": ch, "value": ch} for ch in (meta.get("channel_names") or meta.get("channel_name_union") or [])]
        return classes, chans, f"Dataset: {meta.get('dataset_name') or selected_dataset}"
    except Exception:
        return [], [], f"Dataset: {selected_dataset} (sin metadata)"

@callback(
    [Output(TRAIN_CONFIG_STORE_ID, "data"),
     Output(TRAIN_STATUS_STORE_ID, "data"),
     Output(TRAIN_INTERVAL_ID, "disabled")],
    Input(BTN_TRAIN_ID, "n_clicks"),
    [State(DD_CLASSES_ID, "value"),
     State(DD_CHANNELS_ID, "value"),
     State(IN_SPLIT_ID, "value"),
     State(IN_KFOLDS_ID, "value")],
    prevent_initial_call=True
)
def start_training(n, classes, channels, split, kfolds):
    cfg = {
        "classes": classes or [],
        "channels": channels or [],
        "test_split": float(split or 20.0),
        "kfolds": int(kfolds or 5),
        "ts": time.time(),
        "n_steps": 8
    }
    status = {"status": "running", "step": 0, "t0": time.time()}
    return cfg, status, False  # habilita Interval

@callback(
    [Output(TRAIN_STATUS_STORE_ID, "data", allow_duplicate=True),
     Output(TRAIN_METRICS_STORE_ID, "data"),
     Output(TRAIN_INTERVAL_ID, "disabled", allow_duplicate=True)],
    Input(TRAIN_INTERVAL_ID, "n_intervals"),
    [State(TRAIN_STATUS_STORE_ID, "data"),
     State(TRAIN_CONFIG_STORE_ID, "data")],
    prevent_initial_call=True
)
def tick_training(n, status, cfg):
    if not status or status.get("status") != "running":
        return no_update, no_update, True

    step = int(status.get("step", 0)) + 1
    n_steps = int((cfg or {}).get("n_steps", 8))

    if step < n_steps:
        return {"status": "running", "step": step, "t0": status.get("t0")}, no_update, False

    # terminar (mock)
    if random.random() < 0.8:
        acc = round(random.uniform(0.75, 0.94), 3)
        f1  = round(random.uniform(0.70, 0.92), 3)
        classes = (cfg or {}).get("classes") or ["target","non-target"]
        per_class = [{"class": str(c),
                      "precision": round(random.uniform(0.7,0.95),3),
                      "recall":    round(random.uniform(0.7,0.95),3),
                      "f1":        round(random.uniform(0.7,0.95),3)} for c in classes]
        metrics = {"summary": {"accuracy": acc, "f1_macro": f1}, "per_class": per_class}
        return {"status": "finished", "step": step, "t0": status.get("t0")}, metrics, True
    else:
        return {"status": "error", "step": step, "t0": status.get("t0"),
                "message":"Fallo de entrenamiento simulado"}, no_update, True

@callback(Output(STATUS_VIEW_ID, "children"),
          Input(TRAIN_STATUS_STORE_ID, "data"),
          State(TRAIN_CONFIG_STORE_ID, "data"))
def render_status(status, cfg):
    if not status:
        return html.Div("Esperando a iniciar entrenamiento…", style={"opacity":0.7})

    s = status.get("status")
    step = int(status.get("step", 0))
    n_steps = int((cfg or {}).get("n_steps", 8))
    pct = (step / max(1, n_steps)) * 100.0

    if s == "running":
        return html.Div([
            _badge("Entrenando… va bien", "info"),
            _progress(pct)
        ])
    if s == "finished":
        return html.Div([_badge("Entrenamiento terminado", "ok")])
    if s == "error":
        return html.Div([
            _badge("Error", "error"),
            html.Div(status.get("message") or "Ocurrió un error.", style={"marginTop":".25rem"})
        ])
    return html.Div("—")

@callback(Output(METRICS_VIEW_ID, "children"),
          Input(TRAIN_METRICS_STORE_ID, "data"))
def render_metrics(m):
    if not m:
        return html.Div("Esperando resultados…", style={"opacity":0.7, "minHeight":"1rem"})
    summary = m.get("summary", {})
    per_class = m.get("per_class", [])
    table = html.Table(
        [
            html.Thead(html.Tr([html.Th("Clase"), html.Th("Precisión"), html.Th("Recall"), html.Th("F1")])),
            html.Tbody([
                html.Tr([html.Td(r["class"]), html.Td(r["precision"]), html.Td(r["recall"]), html.Td(r["f1"])])
                for r in per_class
            ])
        ],
        style={"width":"100%","borderCollapse":"collapse"}
    )
    return html.Div([
        html.Div(f"Accuracy: {summary.get('accuracy','—')}  |  F1-macro: {summary.get('f1_macro','—')}",
                 style={"fontWeight":"700","margin":"0 0 .5rem 0"}),
        table
    ])

# =========================
# Navegación de modelos
# =========================

# 1) Al hacer click en cualquier botón de "Configurar" de un modelo
@callback(
    Output(SELECTED_MODEL_STORE_ID, "data"),
    Input({"type": "model-selector-btn", "index": ALL, "model": ALL}, "n_clicks"),
    State(SELECTED_MODEL_STORE_ID, "data"),
    prevent_initial_call=True
)
def select_model(n_clicks_list, current_sel):
    """Guarda el modelo seleccionado cuando se hace click en 'Configurar'."""
    if not n_clicks_list or not any(n_clicks_list):
        return no_update

    from dash import callback_context as ctx
    if not ctx.triggered:
        return no_update

    # Obtener el botón que disparó el callback
    trig = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        btn_id = json.loads(trig)
        model_name = btn_id.get("model")
        return {"name": model_name}
    except Exception as e:
        print(f"Error al seleccionar modelo: {e}")
        return no_update


# 2) Renderiza la card de configuración según el modelo seleccionado
@callback(
    Output(MAIN_VIEW_ID, "children", allow_duplicate=True),
    [Input(SELECTED_MODEL_STORE_ID, "data"),
     Input(SCHEMAS_STORE_ID, "data"),
     Input("page-classifier-type", "data")],
    State(CURRENT_STEP_STORE_ID, "data"),
    prevent_initial_call=True
)
def render_config_card(selected, schemas, classifier_type, current_step):
    """Renderiza la card de configuración del modelo seleccionado (solo en paso 'model')."""

    # Solo renderizar si estamos en el paso de modelo
    if current_step != "model":
        return no_update

    # Si no hay selección, muestra mensaje de bienvenida CON LAYOUT DE 2 COLUMNAS
    if not selected or not selected.get("name"):
        return html.Div([
            html.Div([
                # Columna central: Mensaje de bienvenida
                html.Div([
                    html.H3("Paso 2: Configurar Modelo", className="text-center mb-3", style={"color": "white"}),
                    html.P(
                        "Selecciona un modelo de la barra derecha para configurarlo de manera interactiva.",
                        className="text-center",
                        style={"color": "rgba(255, 255, 255, 0.7)"}
                    ),
                    html.Div([
                        html.I(className="fas fa-brain fa-5x", style={"color": "#4A90E2"})
                    ], className="text-center mt-5")
                ], className="welcome-message", style={
                    "padding": "60px 20px",
                    "borderRadius": "8px",
                    "minHeight": "400px",
                    "flex": "1"
                }),
                # Columna derecha: Sidebar de modelos
                html.Div(
                    id="dynamic-sidebar-p300",
                    children=_models_sidebar(MODEL_NAMES),
                    style={"width": "340px", "padding": "1rem"}
                ),
            ], style={"display": "flex", "flex": "1"})
        ])

    model_name = selected["name"]
    schema = (schemas or {}).get(model_name, {})

    if not schema:
        content = html.Div([
            html.H4(f"Error: No se encontró esquema para {model_name}", style={"color": "#ff6b6b"})
        ])
    else:
        # Crear card de configuración interactiva con el tipo de clasificador
        content = create_config_card(model_name, schema, classifier_type or "P300")

    # SIEMPRE mantener el layout de 2 columnas con la sidebar
    return html.Div([
        html.Div([
            # Columna central: Card del modelo
            html.Div(content, style={"flex": "1", "padding": "1rem"}),
            # Columna derecha: Sidebar de modelos
            html.Div(
                id="dynamic-sidebar-p300",
                children=_models_sidebar(MODEL_NAMES),
                style={"width": "340px", "padding": "1rem"}
            ),
        ], style={"display": "flex", "flex": "1"})
    ])


# 3) Botón "Volver" dentro de la card de configuración
@callback(
    Output(SELECTED_MODEL_STORE_ID, "data", allow_duplicate=True),
    Input("config-back-btn", "n_clicks"),
    prevent_initial_call=True
)
def back_to_welcome(_):
    """Limpia la selección para volver a la vista de bienvenida."""
    return None


# =========================
# Registrar callbacks del visor de transformaciones
# =========================
# NOTA: Los callbacks de filtrado por clase YA están incluidos en register_transformation_callbacks
# No necesitamos duplicarlos aquí
# Esto registra todos los callbacks (Python y clientside) del componente TransformationViewer
# con el prefijo "p300" y el tipo de modelo "p300"
# NOTA: Los callbacks del checklist de canales ya están incluidos en register_transformation_callbacks
register_transformation_callbacks(prefix="p300", model_type="p300")
