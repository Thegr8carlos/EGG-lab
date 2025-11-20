"""
Componente de Selección de Modelos para Simulación
===================================================

Permite al usuario seleccionar modelos P300 e Inner Speech entrenados
y visualizar su información (métricas, pipeline, window size).
"""

from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import json
from typing import Optional


def create_model_selector_section() -> html.Div:
    """
    Crea la sección completa de selección de modelos.

    Returns:
        Componente Dash con selectores y visualización de info
    """
    return html.Div([
        html.H5([
            html.I(className="fas fa-brain me-2"),
            "Selección de Modelos"
        ], className="mb-3", style={"color": "white", "fontWeight": "600"}),

        # Row con ambos selectores
        dbc.Row([
            # Modelo P300
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-wave-square me-2"),
                        "🔍 Modelo P300"
                    ], style={
                        "backgroundColor": "rgba(0, 255, 240, 0.1)",
                        "borderBottom": "2px solid var(--accent-2)"
                    }),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="p300-model-selector",
                            placeholder="Cargando modelos P300...",
                            style={
                                "backgroundColor": "#FFFFFF",
                                "color": "#000000",
                                "fontSize": "14px"
                            },
                            className="mb-3"
                        ),
                        html.Div(id="p300-model-info")
                    ])
                ], style={
                    "backgroundColor": "var(--card-bg)",
                    "border": "1px solid var(--accent-2)",
                    "borderRadius": "8px"
                })
            ], width=6),

            # Modelo Inner Speech
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-brain me-2"),
                        "🎯 Modelo Inner Speech"
                    ], style={
                        "backgroundColor": "rgba(255, 0, 229, 0.1)",
                        "borderBottom": "2px solid var(--accent-1)"
                    }),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="inner-model-selector",
                            placeholder="Cargando modelos Inner...",
                            style={
                                "backgroundColor": "#FFFFFF",
                                "color": "#000000",
                                "fontSize": "14px"
                            },
                            className="mb-3"
                        ),
                        html.Div(id="inner-model-info")
                    ])
                ], style={
                    "backgroundColor": "var(--card-bg)",
                    "border": "1px solid var(--accent-1)",
                    "borderRadius": "8px"
                })
            ], width=6)
        ], className="mb-3"),

        # Stores para guardar configuraciones de modelos
        dcc.Store(id="p300-model-config"),
        dcc.Store(id="inner-model-config"),
        dcc.Store(id="models-ready-flag", data=False)  # Flag cuando ambos están cargados
    ], className="mb-4", style={
        "backgroundColor": "rgba(0,0,0,0.15)",
        "padding": "20px",
        "borderRadius": "12px",
        "border": "1px solid rgba(255,255,255,0.1)"
    })


# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    Output("p300-model-selector", "options"),
    Output("p300-model-selector", "placeholder"),
    Input("selected-dataset", "data"),
    prevent_initial_call=False
)
def load_p300_models(dataset_name):
    """Carga lista de modelos P300 disponibles"""
    from backend.helpers.simulation_utils import list_available_models

    try:
        models = list_available_models("p300")

        if not models:
            return [], "No hay modelos P300 disponibles"

        # Crear opciones para dropdown
        options = [
            {
                "label": f"{m['model_name']} - Exp#{m['experiment_id']} - Acc: {m['metrics'].get('accuracy', 0):.1%} ({m['timestamp']})",
                "value": json.dumps({
                    "snapshot_path": m['snapshot_path'],
                    "pkl_path": m['pkl_path'],
                    "model_name": m['model_name'],
                    "experiment_id": m['experiment_id']
                })
            }
            for m in models
        ]

        return options, "Selecciona un modelo P300..."

    except Exception as e:
        print(f"[load_p300_models] Error: {e}")
        return [], f"Error cargando modelos: {str(e)}"


@callback(
    Output("inner-model-selector", "options"),
    Output("inner-model-selector", "placeholder"),
    Input("selected-dataset", "data"),
    prevent_initial_call=False
)
def load_inner_models(dataset_name):
    """Carga lista de modelos Inner Speech disponibles"""
    from backend.helpers.simulation_utils import list_available_models

    try:
        models = list_available_models("inner")

        if not models:
            return [], "No hay modelos Inner disponibles"

        # Crear opciones para dropdown
        options = [
            {
                "label": f"{m['model_name']} - Exp#{m['experiment_id']} - Acc: {m['metrics'].get('accuracy', 0):.1%} ({m['timestamp']})",
                "value": json.dumps({
                    "snapshot_path": m['snapshot_path'],
                    "pkl_path": m['pkl_path'],
                    "model_name": m['model_name'],
                    "experiment_id": m['experiment_id']
                })
            }
            for m in models
        ]

        return options, "Selecciona un modelo Inner Speech..."

    except Exception as e:
        print(f"[load_inner_models] Error: {e}")
        return [], f"Error cargando modelos: {str(e)}"


@callback(
    Output("p300-model-info", "children"),
    Output("p300-model-config", "data"),
    Input("p300-model-selector", "value"),
    prevent_initial_call=True
)
def display_p300_model_info(selected_model_json):
    """
    Muestra información del modelo P300 seleccionado y carga su configuración.
    """
    if not selected_model_json:
        return None, None

    try:
        from backend.helpers.simulation_utils import load_model_for_inference

        selected = json.loads(selected_model_json)

        # Cargar configuración completa del modelo
        config = load_model_for_inference(
            selected['snapshot_path'],
            selected['pkl_path']
        )

        # Crear card con información
        info_card = dbc.Alert([
            html.Div([
                html.I(className="fas fa-check-circle me-2", style={"color": "#28a745"}),
                html.Strong("Modelo cargado exitosamente", style={"fontSize": "14px"})
            ], className="mb-2"),

            html.Hr(style={"margin": "10px 0", "borderColor": "rgba(255,255,255,0.2)"}),

            # Window size
            html.Div([
                html.I(className="fas fa-ruler me-2", style={"fontSize": "11px"}),
                html.Small([
                    html.Strong("Window: "),
                    f"{config['window_size_samples']} samples ({config['window_size_samples']/1024:.2f}s @ 1024Hz)"
                ], style={"fontSize": "12px"})
            ], className="mb-1"),

            # Pipeline
            html.Div([
                html.I(className="fas fa-cogs me-2", style={"fontSize": "11px"}),
                html.Small([
                    html.Strong("Pipeline: "),
                    f"{len(config['pipeline_config']['filters'])} filtros + Transformación"
                ], style={"fontSize": "12px"})
            ], className="mb-1"),

            # Experiment ID
            html.Div([
                html.I(className="fas fa-flask me-2", style={"fontSize": "11px"}),
                html.Small([
                    html.Strong("Experimento: "),
                    f"#{config['model_metadata']['experiment_id']}"
                ], style={"fontSize": "12px"})
            ]),

        ], color="success", className="mb-0", style={"fontSize": "13px", "padding": "12px"})

        # Serializar config para guardar en Store (sin model_instance)
        config_serializable = {
            k: v for k, v in config.items() if k != 'model_instance'
        }
        config_serializable['snapshot_path'] = selected['snapshot_path']
        config_serializable['pkl_path'] = selected['pkl_path']

        return info_card, config_serializable

    except Exception as e:
        import traceback
        traceback.print_exc()

        error_card = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error cargando modelo: {str(e)}"
        ], color="danger", className="mb-0")

        return error_card, None


@callback(
    Output("inner-model-info", "children"),
    Output("inner-model-config", "data"),
    Input("inner-model-selector", "value"),
    prevent_initial_call=True
)
def display_inner_model_info(selected_model_json):
    """
    Muestra información del modelo Inner Speech seleccionado y carga su configuración.
    """
    if not selected_model_json:
        return None, None

    try:
        from backend.helpers.simulation_utils import load_model_for_inference

        selected = json.loads(selected_model_json)

        # Cargar configuración completa del modelo
        config = load_model_for_inference(
            selected['snapshot_path'],
            selected['pkl_path']
        )

        # Crear card con información
        info_card = dbc.Alert([
            html.Div([
                html.I(className="fas fa-check-circle me-2", style={"color": "#28a745"}),
                html.Strong("Modelo cargado exitosamente", style={"fontSize": "14px"})
            ], className="mb-2"),

            html.Hr(style={"margin": "10px 0", "borderColor": "rgba(255,255,255,0.2)"}),

            # Window size
            html.Div([
                html.I(className="fas fa-ruler me-2", style={"fontSize": "11px"}),
                html.Small([
                    html.Strong("Window: "),
                    f"{config['window_size_samples']} samples ({config['window_size_samples']/1024:.2f}s @ 1024Hz)"
                ], style={"fontSize": "12px"})
            ], className="mb-1"),

            # Pipeline
            html.Div([
                html.I(className="fas fa-cogs me-2", style={"fontSize": "11px"}),
                html.Small([
                    html.Strong("Pipeline: "),
                    f"{len(config['pipeline_config']['filters'])} filtros + Transformación"
                ], style={"fontSize": "12px"})
            ], className="mb-1"),

            # Clases
            html.Div([
                html.I(className="fas fa-tags me-2", style={"fontSize": "11px"}),
                html.Small([
                    html.Strong("Clases: "),
                    f"{len(config['model_metadata']['classes'])} ({', '.join(config['model_metadata']['classes'][:3])}...)"
                ], style={"fontSize": "12px"})
            ], className="mb-1"),

            # Experiment ID
            html.Div([
                html.I(className="fas fa-flask me-2", style={"fontSize": "11px"}),
                html.Small([
                    html.Strong("Experimento: "),
                    f"#{config['model_metadata']['experiment_id']}"
                ], style={"fontSize": "12px"})
            ]),

        ], color="success", className="mb-0", style={"fontSize": "13px", "padding": "12px"})

        # Serializar config para guardar en Store (sin model_instance)
        config_serializable = {
            k: v for k, v in config.items() if k != 'model_instance'
        }
        config_serializable['snapshot_path'] = selected['snapshot_path']
        config_serializable['pkl_path'] = selected['pkl_path']

        return info_card, config_serializable

    except Exception as e:
        import traceback
        traceback.print_exc()

        error_card = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error cargando modelo: {str(e)}"
        ], color="danger", className="mb-0")

        return error_card, None


@callback(
    Output("models-ready-flag", "data"),
    Input("p300-model-config", "data"),
    Input("inner-model-config", "data"),
    prevent_initial_call=False
)
def check_models_ready(p300_config, inner_config):
    """
    Actualiza flag cuando ambos modelos están cargados.
    """
    return bool(p300_config and inner_config)
