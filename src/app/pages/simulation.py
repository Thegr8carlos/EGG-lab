"""
Página de Simulación en Tiempo Real
====================================
Mock de la simulación del pipeline completo:
1. Modelo P300 detecta si hay P300 (0 o 1)
2. Si hay P300 (1), el modelo de clasificación determina la clase
"""

from dash import html, dcc, register_page, callback, Output, Input, State
import dash_bootstrap_components as dbc
from app.components.SideBar import get_sideBar
from app.components.DataSimulation import (
    get_simulation_view,
    register_simulation_clientside,
    register_simulation_legend,
)

register_page(__name__, path="/simulation", name="Simulation")

# ===== IDs Únicos para Simulación =====
CONTAINER_ID = "simulation-view-container"
STORE_ID = "full-signal-data-sim"
LABEL_STORE = "label-color-store-sim"
LEGEND_ID = "dynamic-color-legend-sim"
GRAPH_ID = "signal-graph-sim"
INTERVAL_ID = "interval-component-sim"

# IDs para el mock de modelos
P300_STATUS_ID = "p300-status-display"
CLASSIFICATION_STATUS_ID = "classification-status-display"
MODEL_LOG_ID = "model-log-display"

layout = html.Div(
    [
        # Sidebar
        html.Div(
            id="sidebar-wrapper",
            children=[get_sideBar("Data")],
            className="sideBar-container",
            style={"width": "260px", "padding": "1rem"},
        ),

        # Main content
        html.Div(
            [
                # Header con información
                html.Div([
                    html.H2("🎮 Simulación en Tiempo Real",
                           style={"color": "var(--accent-3)", "marginBottom": "0.5rem"}),
                    html.P([
                        "Visualiza el comportamiento de los modelos en tiempo real. ",
                        html.Strong("Modelo P300", style={"color": "var(--accent-2)"}),
                        " detecta la presencia de P300, y si se detecta, el ",
                        html.Strong("Modelo de Clasificación", style={"color": "var(--accent-1)"}),
                        " determina la clase específica."
                    ], style={"color": "var(--text)", "fontSize": "1rem", "marginBottom": "2rem"})
                ], style={"marginBottom": "2rem"}),

                # Panel de Estado de Modelos
                dbc.Row([
                    # Modelo P300
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                html.H5("🔍 Modelo P300", style={"margin": 0, "color": "var(--text)"}),
                                style={
                                    "backgroundColor": "rgba(0, 255, 240, 0.1)",
                                    "borderBottom": "2px solid var(--accent-2)"
                                }
                            ),
                            dbc.CardBody([
                                html.Div(id=P300_STATUS_ID, children=[
                                    html.Div([
                                        html.Span("Estado: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                                        html.Span("Esperando señal...", style={"color": "var(--text-muted)"})
                                    ], style={"marginBottom": "1rem"}),
                                    html.Div([
                                        html.Span("Última predicción: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                                        html.Span("--", style={"color": "var(--text-muted)"})
                                    ], style={"marginBottom": "1rem"}),
                                    html.Div([
                                        html.Span("Confianza: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                                        html.Span("--", style={"color": "var(--text-muted)"})
                                    ]),
                                ])
                            ])
                        ], style={
                            "backgroundColor": "var(--card-bg)",
                            "border": "1px solid var(--accent-2)",
                            "borderRadius": "var(--radius-md)",
                            "height": "100%"
                        })
                    ], width=6),

                    # Modelo de Clasificación
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                html.H5("🎯 Modelo Clasificación", style={"margin": 0, "color": "var(--text)"}),
                                style={
                                    "backgroundColor": "rgba(255, 0, 229, 0.1)",
                                    "borderBottom": "2px solid var(--accent-1)"
                                }
                            ),
                            dbc.CardBody([
                                html.Div(id=CLASSIFICATION_STATUS_ID, children=[
                                    html.Div([
                                        html.Span("Estado: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                                        html.Span("Inactivo (esperando P300=1)", style={"color": "var(--text-muted)"})
                                    ], style={"marginBottom": "1rem"}),
                                    html.Div([
                                        html.Span("Clase detectada: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                                        html.Span("--", style={"color": "var(--text-muted)"})
                                    ], style={"marginBottom": "1rem"}),
                                    html.Div([
                                        html.Span("Confianza: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                                        html.Span("--", style={"color": "var(--text-muted)"})
                                    ]),
                                ])
                            ])
                        ], style={
                            "backgroundColor": "var(--card-bg)",
                            "border": "1px solid var(--accent-1)",
                            "borderRadius": "var(--radius-md)",
                            "height": "100%"
                        })
                    ], width=6),
                ], style={"marginBottom": "2rem"}),

                # Log de Actividad
                html.Div([
                    html.H5("📊 Log de Actividad del Pipeline",
                           style={"color": "var(--accent-3)", "marginBottom": "1rem"}),
                    html.Div(
                        id=MODEL_LOG_ID,
                        children=[
                            html.Div("🟢 Sistema iniciado - Esperando selección de archivo...",
                                   style={"color": "var(--text)", "fontFamily": "monospace", "fontSize": "0.9rem"})
                        ],
                        style={
                            "backgroundColor": "rgba(10, 1, 36, 0.5)",
                            "border": "1px solid var(--border-weak)",
                            "borderRadius": "var(--radius-md)",
                            "padding": "1rem",
                            "maxHeight": "200px",
                            "overflowY": "auto",
                            "fontFamily": "monospace"
                        }
                    )
                ], style={"marginBottom": "2rem"}),

                # Instrucciones
                dbc.Alert([
                    html.H6("💡 Cómo usar la simulación:", style={"marginBottom": "1rem"}),
                    html.Ul([
                        html.Li("Selecciona un archivo EEG desde el sidebar izquierdo"),
                        html.Li("El sistema cargará la señal y comenzará la simulación automática"),
                        html.Li("Observa cómo el Modelo P300 detecta potenciales P300 en tiempo real"),
                        html.Li("Cuando se detecta P300 (salida=1), el Modelo de Clasificación determina la clase"),
                        html.Li("El log muestra el flujo completo del pipeline")
                    ], style={"marginBottom": 0})
                ], color="info", style={"marginBottom": "2rem"}),

                # Visualización de señal en tiempo real
                html.Div(
                    id=CONTAINER_ID,
                    children=[get_simulation_view(
                        container_id="sim-view-inner",
                        full_signal_store_id=STORE_ID,
                        label_color_store_id=LABEL_STORE,
                        legend_container_id=LEGEND_ID,
                        graph_id=GRAPH_ID,
                        interval_id=INTERVAL_ID
                    )],
                ),

                # Nota de desarrollo
                html.Div([
                    html.Hr(style={"borderColor": "var(--border-weak)", "margin": "2rem 0"}),
                    html.P([
                        "⚠️ ",
                        html.Strong("Nota:", style={"color": "var(--accent-2)"}),
                        " Esta es una ",
                        html.Strong("vista previa (mock)", style={"color": "var(--accent-1)"}),
                        " de la funcionalidad de simulación. Los modelos aún están en desarrollo. ",
                        "La lógica completa de inferencia en tiempo real se implementará próximamente."
                    ], style={"color": "var(--text-muted)", "fontSize": "0.9rem", "fontStyle": "italic", "textAlign": "center"})
                ])
            ],
            style={
                "flex": "1",
                "marginLeft": "280px",
                "padding": "2rem",
                "overflowY": "auto",
                "height": "100vh",
            },
        ),
    ],
    style={"display": "flex", "height": "100vh"},
)


# Registrar callbacks del componente de visualización
register_simulation_clientside(GRAPH_ID, INTERVAL_ID, STORE_ID)
register_simulation_legend(LEGEND_ID, LABEL_STORE)


# ===== Callback: Cargar datos RAW del archivo seleccionado =====
@callback(
    [Output(STORE_ID, "data"),
     Output(LABEL_STORE, "data"),
     Output(INTERVAL_ID, "disabled")],
    [Input("selected-file-path", "data")],
    prevent_initial_call=True
)
def load_signal_data(selected_file_path):
    """Carga la señal EEG RAW completa cuando se selecciona un archivo desde el sidebar"""
    from backend.classes.dataset import Dataset

    if not selected_file_path:
        return None, None, True

    try:
        # Usar Dataset.load_signal_data() que carga el archivo RAW completo
        # Esta función ya maneja la transposición y retorna el formato correcto
        signal_data, interval_disabled = Dataset.load_signal_data(selected_file_path)

        if signal_data is None:
            print(f"[Simulation] ❌ No se pudo cargar el archivo: {selected_file_path}")
            return None, None, True

        # Extraer label_color_map para el store de leyenda
        label_color_map = signal_data.get("label_color_map", {})
        label_store_data = {"label_color_map": label_color_map}

        print(f"[Simulation] ✅ Datos RAW cargados:")
        print(f"  - Canales: {signal_data['num_channels']}")
        print(f"  - Timepoints: {signal_data['num_timepoints']}")
        print(f"  - Labels únicos: {len(label_color_map)}")

        return signal_data, label_store_data, interval_disabled

    except Exception as e:
        print(f"[Simulation] ❌ Error cargando datos: {e}")
        import traceback
        traceback.print_exc()
        return None, None, True


# ===== Mock Callback: Simular comportamiento de modelos en tiempo real =====
@callback(
    [Output(P300_STATUS_ID, "children"),
     Output(CLASSIFICATION_STATUS_ID, "children"),
     Output(MODEL_LOG_ID, "children")],
    [Input(INTERVAL_ID, "n_intervals")],  # Sincronizado con la animación
    [State(STORE_ID, "data"),
     State(MODEL_LOG_ID, "children")]
)
def update_model_status(n_intervals, signal_data, current_log):
    """Mock: Simula predicciones de modelos sincronizadas con la ventana visible"""
    # Prevenir ejecución si no hay datos válidos
    if not signal_data or n_intervals is None or n_intervals == 0:
        p300_status = [
            html.Div([
                html.Span("Estado: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                html.Span("Esperando señal...", style={"color": "var(--text-muted)"})
            ], style={"marginBottom": "1rem"}),
        ]
        classification_status = [
            html.Div([
                html.Span("Estado: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                html.Span("Inactivo", style={"color": "var(--text-muted)"})
            ], style={"marginBottom": "1rem"}),
        ]
        # Asegurar que current_log no sea None
        if current_log is None:
            current_log = [html.Div("🟢 Sistema iniciado - Esperando selección de archivo...",
                                   style={"color": "var(--text)", "fontFamily": "monospace", "fontSize": "0.9rem"})]
        return p300_status, classification_status, current_log

    # Calcular la posición actual en la señal (sincronizado con el gráfico)
    import random
    STEP = 17  # Mismo STEP que en el clientside
    num_timepoints = signal_data.get("num_timepoints", 0)
    current_position = (n_intervals * STEP) % num_timepoints if num_timepoints > 0 else 0

    # Calcular tiempo en segundos
    sfreq = signal_data.get("metadata", {}).get("sampling_frequency_hz", 1024.0)
    current_time_sec = current_position / sfreq if sfreq > 0 else 0

    # Mock: Simular detección de P300 con probabilidad variable
    # Aumentamos probabilidad cada 3 segundos para simular eventos
    time_mod = int(current_time_sec) % 5
    p300_probability = 0.6 if time_mod < 1 else 0.15
    p300_detected = random.random() < p300_probability
    p300_confidence = random.uniform(0.75, 0.95) if p300_detected else random.uniform(0.1, 0.35)

    # Modelo P300
    p300_status = [
        html.Div([
            html.Span("Estado: ", style={"fontWeight": "bold", "color": "var(--text)"}),
            html.Span("✅ Activo" if p300_detected else "⏸️ Procesando",
                     style={"color": "var(--accent-3)" if p300_detected else "var(--text-muted)"})
        ], style={"marginBottom": "1rem"}),
        html.Div([
            html.Span("Tiempo actual: ", style={"fontWeight": "bold", "color": "var(--text)"}),
            html.Span(f"{current_time_sec:.2f}s (muestra {current_position})",
                     style={"color": "var(--accent-2)", "fontSize": "0.9rem"})
        ], style={"marginBottom": "1rem"}),
        html.Div([
            html.Span("Predicción P300: ", style={"fontWeight": "bold", "color": "var(--text)"}),
            html.Span(f"{1 if p300_detected else 0}",
                     style={"color": "var(--accent-3)" if p300_detected else "#FF235A", "fontSize": "1.2rem", "fontWeight": "bold"})
        ], style={"marginBottom": "1rem"}),
        html.Div([
            html.Span("Confianza: ", style={"fontWeight": "bold", "color": "var(--text)"}),
            html.Span(f"{p300_confidence:.2%}",
                     style={"color": "var(--accent-2)"})
        ]),
    ]

    # Modelo de Clasificación (solo activo si P300=1)
    if p300_detected:
        predicted_class = random.choice(["arriba", "abajo", "izquierda", "derecha"])
        class_confidence = random.uniform(0.75, 0.95)

        classification_status = [
            html.Div([
                html.Span("Estado: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                html.Span("✅ Clasificando", style={"color": "var(--accent-3)"})
            ], style={"marginBottom": "1rem"}),
            html.Div([
                html.Span("Clase detectada: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                html.Span(predicted_class.capitalize(), style={"color": "var(--accent-1)", "fontSize": "1.1rem"})
            ], style={"marginBottom": "1rem"}),
            html.Div([
                html.Span("Confianza: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                html.Span(f"{class_confidence:.2%}", style={"color": "var(--accent-2)"})
            ]),
        ]

        # Agregar al log con timestamp
        new_log_entry = html.Div(
            f"[t={current_time_sec:.2f}s] 🎯 P300 detectado! → Clase: '{predicted_class}' (conf: {class_confidence:.1%})",
            style={"color": "var(--accent-3)", "marginBottom": "0.5rem", "fontFamily": "monospace"}
        )
    else:
        classification_status = [
            html.Div([
                html.Span("Estado: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                html.Span("⏸️ En espera (P300=0)", style={"color": "var(--text-muted)"})
            ], style={"marginBottom": "1rem"}),
            html.Div([
                html.Span("Clase detectada: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                html.Span("--", style={"color": "var(--text-muted)"})
            ], style={"marginBottom": "1rem"}),
            html.Div([
                html.Span("Confianza: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                html.Span("--", style={"color": "var(--text-muted)"})
            ]),
        ]

        # Solo agregar entrada al log cada 30 frames (~0.5s) para no saturar
        if n_intervals % 30 == 0:
            new_log_entry = html.Div(
                f"[t={current_time_sec:.2f}s] ⏸️ No P300 detectado",
                style={"color": "var(--text-muted)", "marginBottom": "0.5rem", "fontFamily": "monospace"}
            )
        else:
            new_log_entry = None

    # Actualizar log (mantener últimos 15 mensajes)
    if not isinstance(current_log, list):
        current_log = [current_log]

    # Solo agregar si hay nuevo log
    if new_log_entry is not None:
        updated_log = [new_log_entry] + current_log[:14]
    else:
        updated_log = current_log

    return p300_status, classification_status, updated_log
