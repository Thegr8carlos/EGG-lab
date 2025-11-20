"""
Página de Simulación P300 + Inner Speech
=========================================

Sistema completo de simulación en tiempo real para evaluar modelos
P300 e Inner Speech con datos de sesiones EEG completas.
"""

from dash import html, dcc, register_page, callback, clientside_callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
from app.components.SideBar import get_sideBar
from app.components.ModelSelector import create_model_selector_section
# (SimulationRealtime ya no se usa - implementación simple integrada)
import numpy as np
from typing import List, Dict

register_page(__name__, path="/simulation", name="Simulation")

# Global cache for real-time simulation (evita serializar resultados grandes en Store)
_realtime_cache = {
    "engine": None,
    "results": [],
    "raw_signal": None,
    "labels": None
}


# ============================================================================
# LAYOUT PRINCIPAL
# ============================================================================

layout = html.Div([
    # Sidebar
    html.Div(
        id="sidebar-wrapper",
        children=[get_sideBar("Data")],
        className="sideBar-container",
        style={"width": "260px", "padding": "1rem"}
    ),

    # Main content
    html.Div([
        # Header
        html.Div([
            html.H2([
                html.I(className="fas fa-brain me-3"),
                "Simulación P300 + Inner Speech"
            ], style={"color": "var(--accent-3)", "marginBottom": "0.5rem"}),

            html.P([
                "Sistema de simulación para evaluar modelos en condiciones realistas. ",
                html.Strong("P300", style={"color": "var(--accent-2)"}),
                " detecta eventos y ",
                html.Strong("Inner Speech", style={"color": "var(--accent-1)"}),
                " clasifica la acción mental."
            ], style={"color": "var(--text)", "fontSize": "1rem", "opacity": "0.9", "marginBottom": "2rem"})
        ]),

        # Paso 1: Selección de Modelos
        create_model_selector_section(),

        html.Hr(style={"borderColor": "rgba(255,255,255,0.2)", "margin": "2rem 0"}),

        # Paso 2: Configuración de Simulación
        html.Div([
            html.H5([
                html.I(className="fas fa-sliders-h me-2"),
                "Configuración de Simulación"
            ], className="mb-3", style={"color": "white", "fontWeight": "600"}),

            dbc.Row([
                # Hop Size
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label([
                                html.I(className="fas fa-arrows-alt-h me-2", style={"fontSize": "12px"}),
                                "Hop Size (Solapamiento de Ventanas)"
                            ], style={"color": "white", "fontSize": "14px", "fontWeight": "500"}),

                            dcc.Slider(
                                id="hop-size-slider",
                                min=25,
                                max=75,
                                step=5,
                                value=50,
                                marks={
                                    25: {'label': '25%', 'style': {'color': 'white', 'fontSize': '11px'}},
                                    50: {'label': '50%', 'style': {'color': 'var(--accent-3)', 'fontSize': '12px', 'fontWeight': 'bold'}},
                                    75: {'label': '75%', 'style': {'color': 'white', 'fontSize': '11px'}}
                                },
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mt-2"
                            ),

                            html.Small([
                                html.I(className="fas fa-info-circle me-1", style={"fontSize": "10px"}),
                                "50% = overlap estándar. Menor = más rápido, Mayor = más detección"
                            ], className="text-muted", style={"fontSize": "11px", "marginTop": "8px", "display": "block"})
                        ])
                    ], style={
                        "backgroundColor": "rgba(0,0,0,0.2)",
                        "border": "1px solid rgba(255,255,255,0.1)",
                        "borderRadius": "8px"
                    })
                ], width=12)  # Expandido a ancho completo ya que quitamos el selector de modo
            ], className="mb-3"),

            # Segunda fila: Time Range y Thresholds
            dbc.Row([
                # Time Range Selection
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label([
                                html.I(className="fas fa-clock me-2", style={"fontSize": "12px"}),
                                "Rango de Tiempo (segundos)"
                            ], style={"color": "white", "fontSize": "14px", "fontWeight": "500"}),

                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.Small("Inicio", style={"color": "#aaa", "fontSize": "10px"}),
                                        dbc.Input(
                                            id="time-start-input",
                                            type="number",
                                            min=0,
                                            value=0,
                                            step=1,
                                            style={"fontSize": "12px", "padding": "5px"}
                                        )
                                    ], width=5),
                                    dbc.Col([
                                        html.Small("Fin", style={"color": "#aaa", "fontSize": "10px"}),
                                        dbc.Input(
                                            id="time-end-input",
                                            type="number",
                                            min=0,
                                            value=0,
                                            placeholder="Todo",
                                            step=1,
                                            style={"fontSize": "12px", "padding": "5px"}
                                        )
                                    ], width=5),
                                    dbc.Col([
                                        html.Br(),
                                        dbc.Checkbox(
                                            id="use-full-session",
                                            label="Todo",
                                            value=True,
                                            style={"fontSize": "11px", "marginTop": "18px"}
                                        )
                                    ], width=2)
                                ], className="mt-2")
                            ]),

                            html.Div(id="session-duration-info", className="mt-2", style={"fontSize": "11px", "color": "#6c757d"})
                        ])
                    ], style={
                        "backgroundColor": "rgba(0,0,0,0.2)",
                        "border": "1px solid rgba(255,255,255,0.1)",
                        "borderRadius": "8px"
                    })
                ], width=6),

                # Confidence Thresholds
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label([
                                html.I(className="fas fa-percentage me-2", style={"fontSize": "12px"}),
                                "Thresholds de Confianza"
                            ], style={"color": "white", "fontSize": "14px", "fontWeight": "500"}),

                            html.Div([
                                # P300 Threshold
                                html.Div([
                                    html.Small([
                                        html.I(className="fas fa-wave-square me-1", style={"fontSize": "9px"}),
                                        "P300: "
                                    ], style={"color": "var(--accent-2)", "fontSize": "11px"}),
                                    html.Span(id="p300-threshold-value", children="50%", style={"color": "white", "fontSize": "11px", "fontWeight": "bold"}),
                                    dcc.Slider(
                                        id="p300-threshold-slider",
                                        min=0,
                                        max=100,
                                        step=5,
                                        value=50,
                                        marks={0: '0%', 50: '50%', 100: '100%'},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="mb-2"
                                    )
                                ], className="mb-2"),

                                # Inner Speech Threshold
                                html.Div([
                                    html.Small([
                                        html.I(className="fas fa-brain me-1", style={"fontSize": "9px"}),
                                        "Inner: "
                                    ], style={"color": "var(--accent-1)", "fontSize": "11px"}),
                                    html.Span(id="inner-threshold-value", children="50%", style={"color": "white", "fontSize": "11px", "fontWeight": "bold"}),
                                    dcc.Slider(
                                        id="inner-threshold-slider",
                                        min=0,
                                        max=100,
                                        step=5,
                                        value=50,
                                        marks={0: '0%', 50: '50%', 100: '100%'},
                                        tooltip={"placement": "bottom", "always_visible": False}
                                    )
                                ])
                            ], className="mt-2")
                        ])
                    ], style={
                        "backgroundColor": "rgba(0,0,0,0.2)",
                        "border": "1px solid rgba(255,255,255,0.1)",
                        "borderRadius": "8px"
                    })
                ], width=6)
            ], className="mb-3"),

            # Botón Iniciar
            html.Div([
                dbc.Button(
                    [
                        html.I(className="fas fa-rocket me-2"),
                        "Iniciar Simulación"
                    ],
                    id="btn-start-simulation",
                    color="success",
                    size="lg",
                    disabled=True,  # Se habilita cuando ambos modelos estén cargados
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
                    id="btn-start-simulation-hint",
                    children="Selecciona ambos modelos para habilitar",
                    className="text-muted mt-2 d-block text-center",
                    style={"fontSize": "11px"}
                )
            ])

        ], className="mb-4", style={
            "backgroundColor": "rgba(0,0,0,0.15)",
            "padding": "20px",
            "borderRadius": "12px",
            "border": "1px solid rgba(255,255,255,0.1)"
        }),

        html.Hr(style={"borderColor": "rgba(255,255,255,0.2)", "margin": "2rem 0"}),

        # Contenedor de Visualización (cambia según modo)
        html.Div(id="simulation-view-container", children=[
            # Mensaje inicial
            dbc.Alert([
                html.I(className="fas fa-arrow-up me-2"),
                "Configura los modelos y parámetros arriba, luego presiona 'Iniciar Simulación'"
            ], color="info", className="text-center")
        ]),

        # Stores
        dcc.Store(id="simulation-engine-store"),
        dcc.Store(id="raw-signal-store"),
        dcc.Store(id="labels-store"),
        dcc.Store(id="simulation-results-store"),

        # Store para modo tiempo real NUEVO (clientside callback)
        dcc.Store(id="rt-simulation-data", data={
            "results": [],
            "current_window": 0,
            "total_windows": 0,
            "inner_classes": [],
            "is_complete": False
        }),

        # Store para señal raw completa (visualización como dataset.py)
        dcc.Store(id="rt-full-signal-data"),

        # Store oculto para modo de visualización (siempre full_session)
        dcc.Store(id="visualization-mode", data="full_session"),

        # Stores para modo tiempo real (VIEJO - mantener para compatibilidad)
        dcc.Store(id="realtime-state", data={
            "is_running": False,
            "current_window": 0,
            "total_windows": 0,
            "results": []
        }),

        # Interval para procesamiento incremental (VIEJO)
        dcc.Interval(
            id="realtime-interval",
            interval=2000,
            n_intervals=0,
            disabled=True
        ),

        # Stores para NUEVO componente de tiempo real
        dcc.Store(id="sim-rt-results", data={
            "results": [],
            "current_window": 0,
            "total_windows": 0,
            "is_complete": False
        }),
        dcc.Store(id="sim-rt-metadata"),
        dcc.Interval(
            id="sim-rt-interval",
            interval=2000,
            n_intervals=0,
            disabled=True
        ),
        html.Div(id="sim-rt-progress"),
        html.Div(id="sim-rt-stats"),
        dcc.Graph(id="sim-rt-graph", style={"display": "none"})  # Hidden initially

    ], style={
        "flex": "1",
        "marginLeft": "280px",
        "padding": "2rem",
        "overflowY": "auto",
        "height": "100vh"
    })
], style={"display": "flex", "height": "100vh"})


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def create_full_session_plots(
    raw_signal: np.ndarray,
    labels: np.ndarray,
    results: List[Dict],
    metrics: Dict,
    sfreq: float
) -> html.Div:
    """
    Crea visualizaciones completas de la sesión procesada.

    Args:
        raw_signal: Señal raw (n_channels, n_samples)
        labels: Labels reales (n_samples,)
        results: Lista de resultados de SimulationEngine
        metrics: Dict con métricas globales
        sfreq: Frecuencia de muestreo

    Returns:
        Div con plots y métricas
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Convertir labels a array plano
    labels_flat = labels.flatten() if labels.ndim > 1 else labels

    # ========== PLOT 1: Señal Raw + Eventos Reales ==========
    # Usar primer canal para visualización
    time_axis = np.arange(raw_signal.shape[1]) / sfreq
    channel_0 = raw_signal[0, :]

    # Identificar eventos reales (cambios de label)
    event_changes = []
    event_labels_list = []
    prev_label = labels_flat[0]

    for i, label in enumerate(labels_flat):
        if label != prev_label:
            event_changes.append(i / sfreq)
            event_labels_list.append(str(label))
            prev_label = label

    fig_raw = go.Figure()

    # Señal
    fig_raw.add_trace(go.Scatter(
        x=time_axis,
        y=channel_0,
        mode='lines',
        name='Canal 0',
        line=dict(color='rgba(100, 150, 255, 0.7)', width=1),
        hovertemplate='<b>Tiempo:</b> %{x:.2f}s<br><b>Amplitud:</b> %{y:.2f}<extra></extra>'
    ))

    # Marcadores de eventos reales
    for event_time, event_label in zip(event_changes, event_labels_list):
        color = 'rgba(255, 100, 100, 0.3)' if event_label == 'rest' else 'rgba(100, 255, 100, 0.3)'
        fig_raw.add_vline(
            x=event_time,
            line_dash="dash",
            line_color=color,
            annotation_text=event_label,
            annotation_position="top"
        )

    fig_raw.update_layout(
        title="📊 Señal Raw EEG + Eventos Reales",
        xaxis_title="Tiempo (s)",
        yaxis_title="Amplitud (μV)",
        template="plotly_dark",
        height=500,  # Aumentado de 400 a 500
        hovermode='x unified',
        showlegend=True,
        # Habilitar zoom y pan
        dragmode='pan',
        # Agregar range slider para navegación fácil
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type='linear'
        )
    )

    # ========== PLOT 2: Timeline de Predicciones ==========
    # Extraer datos de predicciones
    pred_times = [r['time_sec'] for r in results]
    p300_preds = [r['p300_prediction'] for r in results]
    inner_preds = [r['inner_prediction'] if r['inner_prediction'] is not None else -1 for r in results]
    is_correct = [r['is_correct'] for r in results]

    fig_pred = go.Figure()

    # Traza 1: Detecciones P300 (scatter) - usar Scattergl para mejor performance
    p300_detected_times = [t for t, p in zip(pred_times, p300_preds) if p == 1]
    p300_detected_y = [1] * len(p300_detected_times)

    fig_pred.add_trace(go.Scattergl(  # Cambiado de Scatter a Scattergl
        x=p300_detected_times,
        y=p300_detected_y,
        mode='markers',
        name='P300 Detectado',
        marker=dict(color='cyan', size=6, symbol='diamond'),  # Size reducido de 10 a 6
        hovertemplate='<b>P300 detectado</b><br>Tiempo: %{x:.2f}s<extra></extra>'
    ))

    # Traza 1.5: Predicciones de Inner Speech (mostrar clase predicha cuando P300=1)
    # Colores por clase
    class_colors = {
        'rest': 'rgba(128, 128, 128, 0.6)',      # Gris
        'arriba': 'rgba(255, 100, 100, 0.8)',    # Rojo
        'abajo': 'rgba(100, 255, 100, 0.8)',     # Verde
        'izquierda': 'rgba(100, 100, 255, 0.8)', # Azul
        'derecha': 'rgba(255, 255, 100, 0.8)',   # Amarillo
        'adelante': 'rgba(255, 100, 255, 0.8)',  # Magenta
        'atras': 'rgba(100, 255, 255, 0.8)'      # Cyan
    }

    # Obtener clases únicas del modelo Inner (si hay resultados con inner_prediction)
    inner_class_names = []
    if results and 'inner_prediction' in results[0]:
        # Intentar obtener nombres de clases del metadata
        # Por ahora usaremos los índices y las extraeremos de los resultados
        for r in results:
            if r['inner_prediction'] is not None and r['p300_prediction'] == 1:
                # Necesitamos mapear inner_prediction (índice) a nombre de clase
                # Por ahora lo dejamos como índice
                pass

    # Agrupar por clase de Inner Speech
    from collections import defaultdict
    inner_by_class = defaultdict(lambda: {'times': [], 'y': []})

    for r in results:
        if r['p300_prediction'] == 1 and r['inner_prediction'] is not None:
            inner_idx = r['inner_prediction']
            # Mapear índice a nombre si es posible
            class_name = f"Clase {inner_idx}"  # Placeholder
            inner_by_class[class_name]['times'].append(r['time_sec'])
            inner_by_class[class_name]['y'].append(0.75)  # Posición entre P300 (1) y clasificación (0.5)

    # Agregar trazas por cada clase Inner Speech
    for class_name, data in inner_by_class.items():
        if data['times']:
            # Asignar color (usar gris si no está en el diccionario)
            color = class_colors.get(class_name.lower().replace('clase ', ''), 'rgba(200, 200, 200, 0.6)')

            fig_pred.add_trace(go.Scattergl(
                x=data['times'],
                y=data['y'],
                mode='markers',
                name=f'Inner: {class_name}',
                marker=dict(color=color, size=8, symbol='square'),
                hovertemplate=f'<b>Inner Speech: {class_name}</b><br>Tiempo: %{{x:.2f}}s<extra></extra>'
            ))

    # Traza 2: Predicciones correctas/incorrectas
    correct_times = [t for t, c in zip(pred_times, is_correct) if c]
    correct_y = [0.5] * len(correct_times)

    incorrect_times = [t for t, c in zip(pred_times, is_correct) if not c]
    incorrect_y = [0.5] * len(incorrect_times)

    fig_pred.add_trace(go.Scattergl(  # Cambiado de Scatter a Scattergl
        x=correct_times,
        y=correct_y,
        mode='markers',
        name='✅ Correcto',
        marker=dict(color='green', size=5, symbol='circle'),  # Size reducido de 8 a 5
        hovertemplate='<b>Predicción correcta</b><br>Tiempo: %{x:.2f}s<extra></extra>'
    ))

    fig_pred.add_trace(go.Scattergl(  # Cambiado de Scatter a Scattergl
        x=incorrect_times,
        y=incorrect_y,
        mode='markers',
        name='❌ Incorrecto',
        marker=dict(color='red', size=5, symbol='x'),  # Size reducido de 8 a 5
        hovertemplate='<b>Predicción incorrecta</b><br>Tiempo: %{x:.2f}s<extra></extra>'
    ))

    fig_pred.update_layout(
        title="🎯 Timeline de Predicciones (P300 + Inner Speech)",
        xaxis_title="Tiempo (s)",
        yaxis=dict(
            tickvals=[0.5, 0.75, 1],
            ticktext=['Correctitud', 'Clase Inner', 'Detección P300'],
            range=[0, 1.5]
        ),
        template="plotly_dark",
        height=450,  # Aumentado de 400 a 450 para dar más espacio
        hovermode='x unified',
        showlegend=True,
        # Habilitar zoom y pan
        dragmode='pan',
        # Agregar range slider
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type='linear',
            range=[0, time_axis[-1]]  # Mover el range aquí
        )
    )

    # Sincronizar rango inicial con plot de señal raw
    fig_raw.update_xaxes(range=[0, time_axis[-1]])

    # ========== MÉTRICAS CARD ==========
    metrics_card = create_metrics_card(metrics, results)

    # ========== COMPONENTE FINAL ==========
    return html.Div([
        # Métricas arriba
        metrics_card,

        html.Hr(style={"borderColor": "rgba(255,255,255,0.2)", "margin": "1.5rem 0"}),

        # Plot de señal raw
        dcc.Graph(
            figure=fig_raw,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                'scrollZoom': True,  # Habilitar zoom con scroll
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'simulation_raw_signal',
                    'height': 1000,
                    'width': 1800,
                    'scale': 2
                }
            }
        ),

        # Plot de predicciones
        dcc.Graph(
            figure=fig_pred,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                'scrollZoom': True,  # Habilitar zoom con scroll
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'simulation_predictions',
                    'height': 800,
                    'width': 1800,
                    'scale': 2
                }
            }
        )
    ])


def create_realtime_view(
    current_window: int,
    total_windows: int,
    current_result: Dict = None,
    recent_results: List[Dict] = None
) -> html.Div:
    """
    Crea vista en tiempo real con diagrama animado del pipeline.

    Args:
        current_window: Índice de ventana actual
        total_windows: Total de ventanas
        current_result: Resultado de la ventana actual
        recent_results: Últimos N resultados para mostrar en plots

    Returns:
        Div con visualización en tiempo real
    """
    import plotly.graph_objects as go

    # Progreso
    progress = (current_window / total_windows * 100) if total_windows > 0 else 0

    # ========== DIAGRAMA DEL PIPELINE ==========
    pipeline_diagram = create_pipeline_diagram(current_result)

    # ========== PLOT DE SEÑAL EN TIEMPO REAL ==========
    # Mostrar últimas N ventanas procesadas
    realtime_plot = create_realtime_signal_plot(recent_results) if recent_results else html.Div()

    # ========== MÉTRICAS EN TIEMPO REAL ==========
    realtime_metrics = create_realtime_metrics(recent_results) if recent_results else html.Div()

    return html.Div([
        # Header con progreso
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-broadcast-tower me-2", style={"color": "var(--accent-2)"}),
                    f"Simulación en Tiempo Real - Ventana {current_window}/{total_windows}"
                ], className="mb-3"),

                dbc.Progress(
                    value=progress,
                    label=f"{progress:.1f}%",
                    className="mb-2",
                    style={"height": "25px", "fontSize": "14px"},
                    color="success" if progress < 100 else "info"
                ),

                html.Small(
                    f"Procesando ventana {current_window} de {total_windows}..." if progress < 100 else "✅ Procesamiento completado",
                    className="text-muted"
                )
            ])
        ], className="mb-3", style={"backgroundColor": "rgba(0,0,0,0.3)", "border": "1px solid rgba(255,255,255,0.2)"}),

        # Diagrama del pipeline
        pipeline_diagram,

        html.Hr(style={"borderColor": "rgba(255,255,255,0.2)", "margin": "1.5rem 0"}),

        # Plots y métricas en tiempo real
        dbc.Row([
            dbc.Col([realtime_plot], width=8),
            dbc.Col([realtime_metrics], width=4)
        ])
    ])


def create_pipeline_diagram(current_result: Dict = None) -> dbc.Card:
    """
    Crea diagrama visual del pipeline de procesamiento.

    Args:
        current_result: Resultado actual para mostrar en el diagrama

    Returns:
        Card con diagrama del pipeline
    """
    # Estados del pipeline
    p300_pred = current_result.get('p300_prediction', None) if current_result else None
    inner_pred = current_result.get('inner_prediction', None) if current_result else None
    is_correct = current_result.get('is_correct', None) if current_result else None

    # Colores según estado
    p300_color = "success" if p300_pred == 1 else "danger" if p300_pred == 0 else "secondary"
    inner_color = "info" if inner_pred is not None else "secondary"
    result_color = "success" if is_correct else "danger" if is_correct is not None else "secondary"

    # Mapeo de colores para bordes
    color_map = {
        'success': 'green',
        'danger': 'red',
        'secondary': '#6c757d',
        'info': 'cyan'
    }

    return dbc.Card([
        dbc.CardHeader(html.H5([
            html.I(className="fas fa-project-diagram me-2"),
            "🔄 Pipeline de Procesamiento"
        ], className="mb-0")),

        dbc.CardBody([
            # Diagrama de flujo horizontal
            html.Div([
                # Paso 1: Raw Signal
                html.Div([
                    html.Div([
                        html.I(className="fas fa-wave-square", style={"fontSize": "30px", "color": "var(--accent-3)"}),
                        html.Div("Raw Signal", className="mt-2", style={"fontSize": "12px", "fontWeight": "500"}),
                        html.Small("(137 ch)", className="text-muted", style={"fontSize": "10px"})
                    ], className="pipeline-step")
                ], className="pipeline-item"),

                # Flecha
                html.Div([
                    html.I(className="fas fa-arrow-right", style={"fontSize": "20px", "color": "var(--accent-2)"})
                ], className="pipeline-arrow"),

                # Paso 2: Transform
                html.Div([
                    html.Div([
                        html.I(className="fas fa-cogs", style={"fontSize": "30px", "color": "var(--accent-1)"}),
                        html.Div("Transform", className="mt-2", style={"fontSize": "12px", "fontWeight": "500"}),
                        html.Small("Wavelet/FFT", className="text-muted", style={"fontSize": "10px"})
                    ], className="pipeline-step")
                ], className="pipeline-item"),

                # Flecha
                html.Div([
                    html.I(className="fas fa-arrow-right", style={"fontSize": "20px", "color": "var(--accent-2)"})
                ], className="pipeline-arrow"),

                # Paso 3: P300 Model
                html.Div([
                    html.Div([
                        html.I(className="fas fa-brain", style={"fontSize": "30px", "color": "cyan"}),
                        html.Div("P300 Model", className="mt-2", style={"fontSize": "12px", "fontWeight": "500"}),
                        dbc.Badge(
                            f"{'✅ Event' if p300_pred == 1 else '❌ Rest' if p300_pred == 0 else '⏳ Waiting'}",
                            color=p300_color,
                            className="mt-1",
                            style={"fontSize": "10px"}
                        )
                    ], className="pipeline-step", style={
                        "border": f"2px solid {color_map.get(p300_color, '#6c757d')}"
                    })
                ], className="pipeline-item"),

                # Flecha condicional
                html.Div([
                    html.I(
                        className="fas fa-arrow-right" if p300_pred == 1 else "fas fa-times",
                        style={"fontSize": "20px", "color": "green" if p300_pred == 1 else "red"}
                    )
                ], className="pipeline-arrow"),

                # Paso 4: Inner Speech Model
                html.Div([
                    html.Div([
                        html.I(className="fas fa-comments", style={
                            "fontSize": "30px",
                            "color": "var(--accent-1)" if p300_pred == 1 else "#6c757d"
                        }),
                        html.Div("Inner Speech", className="mt-2", style={"fontSize": "12px", "fontWeight": "500"}),
                        dbc.Badge(
                            f"Clase {inner_pred}" if inner_pred is not None else ("⏸️ Skipped" if p300_pred == 0 else "⏳ Waiting"),
                            color=inner_color,
                            className="mt-1",
                            style={"fontSize": "10px"}
                        )
                    ], className="pipeline-step", style={
                        "border": f"2px solid {color_map.get(inner_color, '#6c757d')}",
                        "opacity": "1" if p300_pred == 1 else "0.4"
                    })
                ], className="pipeline-item"),

                # Flecha
                html.Div([
                    html.I(className="fas fa-arrow-right", style={"fontSize": "20px", "color": "var(--accent-2)"})
                ], className="pipeline-arrow"),

                # Paso 5: Result
                html.Div([
                    html.Div([
                        html.I(
                            className=f"fas fa-{'check-circle' if is_correct else 'times-circle' if is_correct is not None else 'clock'}",
                            style={"fontSize": "30px", "color": "green" if is_correct else "red" if is_correct is not None else "#6c757d"}
                        ),
                        html.Div("Result", className="mt-2", style={"fontSize": "12px", "fontWeight": "500"}),
                        dbc.Badge(
                            "✅ Correct" if is_correct else "❌ Incorrect" if is_correct is not None else "⏳ Waiting",
                            color=result_color,
                            className="mt-1",
                            style={"fontSize": "10px"}
                        )
                    ], className="pipeline-step")
                ], className="pipeline-item")

            ], style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "padding": "20px",
                "backgroundColor": "rgba(0,0,0,0.2)",
                "borderRadius": "8px"
            })
        ])
    ], className="mb-3", style={"backgroundColor": "rgba(0,0,0,0.3)", "border": "1px solid rgba(255,255,255,0.2)"})


def create_realtime_signal_plot(recent_results: List[Dict]) -> dcc.Graph:
    """
    Crea plot de señal con predicciones en tiempo real (últimas N ventanas).
    """
    import plotly.graph_objects as go

    if not recent_results:
        return dcc.Graph(figure=go.Figure())

    # Extraer datos
    times = [r['time_sec'] for r in recent_results]
    p300_preds = [r['p300_prediction'] for r in recent_results]
    is_correct = [r['is_correct'] for r in recent_results]

    fig = go.Figure()

    # P300 detections
    p300_times = [t for t, p in zip(times, p300_preds) if p == 1]
    fig.add_trace(go.Scattergl(
        x=p300_times,
        y=[1] * len(p300_times),
        mode='markers',
        name='P300',
        marker=dict(color='cyan', size=8, symbol='diamond')
    ))

    # Correctitud
    correct_times = [t for t, c in zip(times, is_correct) if c]
    incorrect_times = [t for t, c in zip(times, is_correct) if not c]

    fig.add_trace(go.Scattergl(
        x=correct_times,
        y=[0.5] * len(correct_times),
        mode='markers',
        name='Correcto',
        marker=dict(color='green', size=6)
    ))

    fig.add_trace(go.Scattergl(
        x=incorrect_times,
        y=[0.5] * len(incorrect_times),
        mode='markers',
        name='Incorrecto',
        marker=dict(color='red', size=6)
    ))

    fig.update_layout(
        title="Predicciones Recientes",
        xaxis_title="Tiempo (s)",
        yaxis=dict(
            tickvals=[0.5, 1],
            ticktext=['Clasificación', 'P300']
        ),
        template="plotly_dark",
        height=350,
        showlegend=True,
        hovermode='x unified'
    )

    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_realtime_metrics(recent_results: List[Dict]) -> dbc.Card:
    """
    Crea card con métricas en tiempo real.
    """
    if not recent_results:
        return dbc.Card()

    total = len(recent_results)
    correct = sum(1 for r in recent_results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    p300_detected = sum(1 for r in recent_results if r['p300_prediction'] == 1)
    p300_rate = p300_detected / total if total > 0 else 0

    return dbc.Card([
        dbc.CardHeader(html.H6("📊 Métricas", className="mb-0")),
        dbc.CardBody([
            html.Div([
                html.H3(f"{accuracy:.1%}", style={"color": "var(--accent-3)"}),
                html.Small("Accuracy", className="text-muted")
            ], className="text-center mb-3"),

            html.Div([
                html.H5(f"{correct}/{total}", style={"color": "white"}),
                html.Small("Correctas", className="text-muted")
            ], className="text-center mb-3"),

            html.Div([
                html.H5(f"{p300_rate:.1%}", style={"color": "cyan"}),
                html.Small("Detección P300", className="text-muted")
            ], className="text-center")
        ])
    ], style={"backgroundColor": "rgba(0,0,0,0.3)", "border": "1px solid rgba(255,255,255,0.2)"})


def create_metrics_card(metrics: Dict, results: List[Dict]) -> dbc.Card:
    """
    Crea card con resumen de métricas.

    Args:
        metrics: Dict de get_metrics_summary()
        results: Lista de resultados

    Returns:
        Card con métricas
    """
    total = metrics['total_windows']
    correct = metrics['correct']
    accuracy = metrics['accuracy']
    p300_detected = metrics['p300_detected']
    p300_rate = metrics['p300_detection_rate']

    # Métricas por clase
    by_class = metrics.get('by_class', {})

    # Crear tabla de métricas por clase
    class_rows = []
    for class_name, class_metrics in sorted(by_class.items()):
        class_rows.append(
            html.Tr([
                html.Td(class_name, style={"fontWeight": "500"}),
                html.Td(f"{class_metrics['total']}", className="text-center"),
                html.Td(f"{class_metrics['correct']}", className="text-center"),
                html.Td(f"{class_metrics['accuracy']:.1%}", className="text-center", style={
                    "color": "green" if class_metrics['accuracy'] > 0.7 else "orange"
                })
            ])
        )

    return dbc.Card([
        dbc.CardHeader(html.H5([
            html.I(className="fas fa-chart-bar me-2"),
            "📈 Métricas de Simulación"
        ], className="mb-0")),

        dbc.CardBody([
            dbc.Row([
                # Métricas globales
                dbc.Col([
                    html.Div([
                        html.H3(f"{accuracy:.1%}", style={"color": "var(--accent-3)", "marginBottom": "0"}),
                        html.Small("Accuracy Global", className="text-muted")
                    ], className="text-center mb-3"),

                    html.Div([
                        html.H5(f"{correct} / {total}", style={"color": "white"}),
                        html.Small("Ventanas Correctas", className="text-muted")
                    ], className="text-center")
                ], width=3),

                dbc.Col([
                    html.Div([
                        html.H3(f"{p300_rate:.1%}", style={"color": "var(--accent-2)", "marginBottom": "0"}),
                        html.Small("Tasa Detección P300", className="text-muted")
                    ], className="text-center mb-3"),

                    html.Div([
                        html.H5(f"{p300_detected} / {total}", style={"color": "white"}),
                        html.Small("P300 Detectados", className="text-muted")
                    ], className="text-center")
                ], width=3),

                # Tabla de métricas por clase
                dbc.Col([
                    html.H6("Métricas por Clase", className="mb-2", style={"color": "white"}),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Clase"),
                            html.Th("Total", className="text-center"),
                            html.Th("Correctas", className="text-center"),
                            html.Th("Accuracy", className="text-center")
                        ])),
                        html.Tbody(class_rows)
                    ], bordered=True, hover=True, size="sm", style={"fontSize": "12px", "backgroundColor": "rgba(0,0,0,0.4)", "color": "white"})
                ], width=6)
            ])
        ])
    ], style={
        "backgroundColor": "rgba(0,0,0,0.3)",
        "border": "1px solid rgba(255,255,255,0.2)"
    }, className="mb-3")


# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    Output("btn-start-simulation", "disabled"),
    Output("btn-start-simulation-hint", "children"),
    Input("models-ready-flag", "data"),
    Input("selected-file-path", "data"),
    prevent_initial_call=False
)
def update_start_button_state(models_ready, selected_file):
    """
    Habilita/deshabilita botón de inicio según estado de modelos y archivo.
    """
    if not models_ready:
        return True, "⚠️ Selecciona ambos modelos (P300 + Inner Speech)"

    if not selected_file:
        return True, "⚠️ Selecciona un archivo EEG desde el sidebar"

    return False, "✅ Todo listo! Presiona para iniciar"


# Este callback fue removido y su lógica se fusionó con start_simulation
# para evitar conflictos de callbacks duplicados con el mismo trigger


# ============================================================================
# CALLBACKS PARA CONFIGURACIÓN
# ============================================================================

@callback(
    Output("p300-threshold-value", "children"),
    Input("p300-threshold-slider", "value")
)
def update_p300_threshold_label(value):
    """Actualiza label del threshold P300"""
    return f"{value}%"


@callback(
    Output("inner-threshold-value", "children"),
    Input("inner-threshold-slider", "value")
)
def update_inner_threshold_label(value):
    """Actualiza label del threshold Inner Speech"""
    return f"{value}%"


@callback(
    Output("time-start-input", "disabled"),
    Output("time-end-input", "disabled"),
    Input("use-full-session", "value")
)
def toggle_time_inputs(use_full):
    """Habilita/deshabilita inputs de tiempo según checkbox 'Todo'"""
    return use_full, use_full


@callback(
    Output("session-duration-info", "children"),
    Input("raw-signal-store", "data")
)
def display_session_duration(raw_data):
    """Muestra duración de la sesión cuando se carga"""
    if not raw_data:
        return ""

    try:
        shape = raw_data.get('shape', [0, 0])
        n_samples = shape[1] if len(shape) > 1 else 0
        duration_sec = n_samples / 1024.0  # Asumiendo 1024 Hz

        minutes = int(duration_sec // 60)
        seconds = int(duration_sec % 60)

        return [
            html.I(className="fas fa-info-circle me-1"),
            f"Duración total: {minutes}m {seconds}s ({duration_sec:.1f}s)"
        ]
    except:
        return ""


# ============================================================================
# COMPONENTES DE VISUALIZACIÓN
# ============================================================================

def create_full_session_view(hop_percent: float) -> html.Div:
    """Vista de sesión completa (SE IMPLEMENTARÁ EN TAREA 6)"""
    return html.Div([
        html.H5([
            html.I(className="fas fa-chart-line me-2"),
            "Procesamiento de Sesión Completa"
        ], className="mb-3", style={"color": "white"}),

        # Botón procesar
        dbc.Button(
            [html.I(className="fas fa-play me-2"), f"▶️ Procesar Sesión (Hop: {hop_percent}%)"],
            id="btn-process-full-session",
            color="primary",
            size="lg",
            className="mb-3 w-100"
        ),

        # Progress
        dbc.Progress(
            id="processing-progress",
            value=0,
            striped=True,
            animated=True,
            className="mb-3",
            style={"height": "25px"}
        ),
        html.Div(id="processing-status", className="mb-3"),

        # Placeholder para plots (TAREA 6)
        html.Div(id="full-session-plots", children=[
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "Presiona 'Procesar Sesión' para comenzar el análisis"
            ], color="info")
        ])
    ])


@callback(
    Output("processing-progress", "value"),
    Output("processing-status", "children"),
    Output("full-session-plots", "children"),
    Input("btn-process-full-session", "n_clicks"),
    State("raw-signal-store", "data"),
    State("labels-store", "data"),
    State("p300-model-config", "data"),
    State("inner-model-config", "data"),
    State("hop-size-slider", "value"),
    prevent_initial_call=True
)
def process_full_session(n_clicks, raw_data, labels_data, p300_config, inner_config, hop_percent):
    """
    Procesa toda la sesión con ambos modelos y genera visualizaciones.
    """
    if not all([raw_data, labels_data, p300_config, inner_config]):
        return 0, dbc.Alert("Datos incompletos", color="danger"), no_update

    try:
        import numpy as np
        from backend.helpers.simulation_engine import SimulationEngine
        from backend.helpers.simulation_utils import load_model_for_inference

        # Cargar señal raw y labels desde archivos
        raw_signal = np.load(raw_data['file_path'])
        labels = np.load(labels_data['file_path'])

        print(f"\n[process_full_session] Iniciando procesamiento:")
        print(f"  Raw signal: {raw_signal.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Hop: {hop_percent}%")

        # Re-cargar instancias de modelos (no estaban en stores)
        p300_full = load_model_for_inference(
            p300_config['snapshot_path'],
            p300_config['pkl_path']
        )

        inner_full = load_model_for_inference(
            inner_config['snapshot_path'],
            inner_config['pkl_path']
        )

        # Crear motor de simulación
        engine = SimulationEngine(
            raw_signal=raw_signal,
            labels=labels,
            sfreq=1024.0,  # InnerSpeech dataset
            p300_model_config=p300_full,
            inner_model_config=inner_full,
            hop_percent=hop_percent
        )

        # Procesar toda la sesión
        results = engine.process_all_windows(verbose=True)

        # Calcular métricas
        metrics = engine.get_metrics_summary()

        print(f"\n[process_full_session] ✅ Procesamiento completado")
        print(f"  Total ventanas: {metrics['total_windows']}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  P300 detectado: {metrics['p300_detected']} ({metrics['p300_detection_rate']:.2%})")

        # Crear visualizaciones
        plots = create_full_session_plots(
            raw_signal=raw_signal,
            labels=labels,
            results=results,
            metrics=metrics,
            sfreq=1024.0
        )

        # Status de éxito
        status = dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            html.Strong(f"✅ Procesamiento completado: {metrics['total_windows']} ventanas analizadas"),
            html.Br(),
            html.Small(f"Accuracy global: {metrics['accuracy']:.2%} | P300 detectado: {metrics['p300_detection_rate']:.2%}")
        ], color="success", className="mt-2")

        return 100, status, plots

    except Exception as e:
        import traceback
        traceback.print_exc()

        error_alert = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong("Error procesando sesión"),
            html.Br(),
            html.Small(str(e))
        ], color="danger")

        return 0, error_alert, no_update

# ============================================================================
# CALLBACKS PARA MODO TIEMPO REAL
# ============================================================================

@callback(
    Output("simulation-view-container", "children", allow_duplicate=True),
    Output("realtime-interval", "disabled"),
    Output("realtime-state", "data"),
    Output("simulation-engine-store", "data"),
    Output("raw-signal-store", "data"),
    Output("labels-store", "data"),
    Output("sim-rt-results", "data", allow_duplicate=True),
    Output("sim-rt-metadata", "data", allow_duplicate=True),
    Output("sim-rt-interval", "disabled", allow_duplicate=True),
    Output("rt-simulation-data", "data", allow_duplicate=True),  # Resetear Store de resultados
    Output("rt-full-signal-data", "data", allow_duplicate=True),  # NUEVO: Señal completa para visualización
    Input("btn-start-simulation", "n_clicks"),
    State("visualization-mode", "value"),
    State("p300-model-config", "data"),
    State("inner-model-config", "data"),
    State("hop-size-slider", "value"),
    State("selected-file-path", "data"),
    State("use-full-session", "value"),
    State("time-start-input", "value"),
    State("time-end-input", "value"),
    State("p300-threshold-slider", "value"),
    State("inner-threshold-slider", "value"),
    prevent_initial_call=True
)
def start_simulation(n_clicks, viz_mode, p300_config, inner_config, hop_percent, file_path,
                    use_full_session, time_start, time_end, p300_threshold, inner_threshold):
    """
    Inicia la simulación según el modo seleccionado (realtime o full_session).
    Fusiona la carga de archivos y la inicialización del motor en un solo callback.
    """
    if not all([p300_config, inner_config, file_path]):
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    try:
        import numpy as np
        from pathlib import Path
        from backend.helpers.simulation_engine import SimulationEngine
        from backend.helpers.simulation_utils import load_model_for_inference

        # ========== PASO 1: CARGAR ARCHIVOS RAW Y LABELS ==========
        file_p = Path(file_path)

        # Construir path a Aux directory
        if "Data" in str(file_p):
            aux_path = str(file_p).replace("Data/", "Aux/").replace("Data\\", "Aux\\")
            aux_dir = Path(aux_path).parent
        else:
            aux_dir = Path("Aux") / file_p.parent

        # Buscar archivo .npy raw
        raw_file_candidates = list(aux_dir.glob("*.npy"))
        raw_file = None

        for candidate in raw_file_candidates:
            if "task-innerspeech_eeg.npy" in candidate.name and "Labels" not in str(candidate):
                raw_file = candidate
                break

        if not raw_file or not raw_file.exists():
            error = dbc.Alert(f"No se encontró archivo raw .npy en {aux_dir}", color="danger")
            return error, True, no_update, no_update, None, None

        # Buscar archivo de labels
        labels_dir = aux_dir / "Labels"
        labels_file = None

        if labels_dir.exists():
            labels_candidates = list(labels_dir.glob("*.npy"))
            if labels_candidates:
                labels_file = labels_candidates[0]

        if not labels_file or not labels_file.exists():
            error = dbc.Alert(f"No se encontró archivo de labels en {labels_dir}", color="danger")
            return error, True, no_update, no_update, None, None

        # Cargar señal raw y labels
        raw_signal = np.load(raw_file)
        labels = np.load(labels_file)

        print(f"[start_simulation] Raw signal cargado: {raw_signal.shape}")
        print(f"[start_simulation] Labels cargadas: {labels.shape}")

        # ========== APLICAR RANGO DE TIEMPO ==========
        sfreq = 1024.0
        if not use_full_session and time_start is not None and time_end is not None:
            # Convertir tiempo a samples
            start_sample = int(time_start * sfreq)
            end_sample = int(time_end * sfreq) if time_end > 0 else raw_signal.shape[1]

            # Validar rango
            start_sample = max(0, start_sample)
            end_sample = min(raw_signal.shape[1], end_sample)

            if end_sample > start_sample:
                # Recortar señal y labels
                raw_signal = raw_signal[:, start_sample:end_sample]
                labels = labels[:, start_sample:end_sample] if labels.ndim > 1 else labels[start_sample:end_sample]

                print(f"[start_simulation] ✂️ Señal recortada: {time_start}s - {time_end}s ({end_sample-start_sample} samples)")
                print(f"[start_simulation] Nueva shape: {raw_signal.shape}")

        # Guardar datos en stores (con shape original para referencia)
        raw_signal_data = {
            "file_path": str(raw_file),
            "shape": list(raw_signal.shape)
        }

        labels_data = {
            "file_path": str(labels_file),
            "shape": list(labels.shape)
        }

        # ========== PASO 2: CARGAR MODELOS Y CREAR ENGINE ==========

        # Re-cargar instancias de modelos
        p300_full = load_model_for_inference(
            p300_config['snapshot_path'],
            p300_config['pkl_path']
        )

        inner_full = load_model_for_inference(
            inner_config['snapshot_path'],
            inner_config['pkl_path']
        )

        # Crear motor de simulación
        engine = SimulationEngine(
            raw_signal=raw_signal,
            labels=labels,
            sfreq=1024.0,
            p300_model_config=p300_full,
            inner_model_config=inner_full,
            hop_percent=hop_percent
        )

        # Guardar engine data (necesitamos guardarlo de forma serializable)
        engine_data = {
            'total_windows': len(engine.window_indices),
            'window_indices': engine.window_indices,
            'raw_path': raw_signal_data['file_path'],
            'labels_path': labels_data['file_path'],
            'p300_snapshot': p300_config['snapshot_path'],
            'p300_pkl': p300_config['pkl_path'],
            'inner_snapshot': inner_config['snapshot_path'],
            'inner_pkl': inner_config['pkl_path'],
            'hop_percent': hop_percent,
            'p300_threshold': p300_threshold / 100.0,  # Convertir a decimal (0-1)
            'inner_threshold': inner_threshold / 100.0  # Convertir a decimal (0-1)
        }

        print(f"[start_simulation] Thresholds: P300={p300_threshold}%, Inner={inner_threshold}%")

        # ========== PASO 3: CREAR VISTA SEGÚN MODO ==========
        if viz_mode == "realtime":
            # Modo tiempo real: Sistema CLIENTSIDE (como dataset.py)
            print(f"\n[start_simulation] Modo tiempo real: Preparando {len(engine.window_indices)} ventanas...")
            print(f"[start_simulation] Sistema clientside - procesamiento en batches de 50 ventanas\n")

            # Guardar engine en cache global
            global _realtime_cache
            _realtime_cache["engine"] = engine
            _realtime_cache["inner_classes"] = inner_config['model_metadata'].get('classes', [])
            _realtime_cache["current_window"] = 0
            _realtime_cache["total_windows"] = len(engine.window_indices)
            _realtime_cache["results"] = []  # Lista de resultados acumulados
            _realtime_cache["is_processing"] = False  # Flag para evitar callbacks concurrentes

            # Vista para modo tiempo real (Store ya existe en layout principal)
            realtime_view = html.Div([
                # DEBUG: Mostrar estado del Store (para verificar que se actualiza)
                html.Div(id="rt-debug-info", style={"padding": "10px", "backgroundColor": "rgba(255,255,0,0.1)", "marginBottom": "10px", "borderRadius": "5px"}),

                # Barra de progreso (server-side)
                html.Div(id="rt-progress-bar", children=[
                    dbc.Alert([
                        html.H5([html.I(className="fas fa-play-circle me-2"), "⚡ Iniciando..."], className="mb-2"),
                        dbc.Progress(value=0, striped=True, animated=True, color="info", style={"height": "25px"})
                    ], color="info")
                ]),

                # Gráfico de señal EEG con overlays (como dataset.py)
                dcc.Graph(
                    id="rt-signal-graph",
                    config={'displayModeBar': True, 'displaylogo': False},
                    style={"height": "800px"}  # Más alto para ver mejor los canales
                ),

                # Interval que dispara backend cada 100ms (procesa 1 ventana)
                dcc.Interval(id="rt-process-interval", interval=100, n_intervals=0, disabled=False)

            ], style={"padding": "20px", "backgroundColor": "rgba(0,0,0,0.15)", "borderRadius": "12px"})

            print(f"[start_simulation] ✅ Vista clientside creada. Habilitando procesamiento...")

            # Preparar señal completa para visualización (formato como dataset.py)
            # Crear mapa de colores para labels
            unique_labels = np.unique(labels)
            label_colors = {}
            color_palette = [
                "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A",
                "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E2"
            ]
            for i, label in enumerate(unique_labels):
                # Labels pueden ser strings o números, convertir todo a string
                label_colors[str(label)] = color_palette[i % len(color_palette)]

            # Preparar datos de señal completa
            rt_signal_data = {
                "data": raw_signal.tolist(),  # Convertir a lista para JSON
                "labels": [str(l) for l in labels.flatten()],  # Convertir labels a strings
                "num_channels": raw_signal.shape[0],
                "num_timepoints": raw_signal.shape[1],
                "label_color_map": label_colors,
                "sfreq": 1024.0,
                "unique_labels": [str(l) for l in unique_labels]
            }

            # Inicializar Store con datos del engine
            rt_store_data = {
                "results": [],
                "current_window": 0,
                "total_windows": len(engine.window_indices),
                "inner_classes": inner_config['model_metadata'].get('classes', []),
                "is_complete": False
            }

            return (
                realtime_view,          # simulation-view-container.children
                True,                   # realtime-interval.disabled (deshabilitar el viejo)
                no_update,              # realtime-state.data (no tocar)
                engine_data,            # simulation-engine-store.data
                raw_signal_data,        # raw-signal-store.data
                labels_data,            # labels-store.data
                no_update,              # sim-rt-results (no tocar)
                no_update,              # sim-rt-metadata (no tocar)
                no_update,              # sim-rt-interval (no tocar)
                rt_store_data,          # rt-simulation-data.data
                rt_signal_data          # rt-full-signal-data.data (NUEVO)
            )

        else:
            # Modo sesión completa: procesar todo de una vez
            results = engine.process_all_windows(verbose=True)
            metrics = engine.get_metrics_summary()

            plots = create_full_session_plots(
                raw_signal=raw_signal,
                labels=labels,
                results=results,
                metrics=metrics,
                sfreq=1024.0
            )

            return plots, True, no_update, engine_data, raw_signal_data, labels_data, no_update, no_update, no_update, no_update, no_update

    except Exception as e:
        import traceback
        traceback.print_exc()

        error_view = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong("Error iniciando simulación"),
            html.Br(),
            html.Small(str(e))
        ], color="danger")

        return error_view, True, no_update, no_update, None, None, no_update, no_update, no_update, no_update, no_update


# ============================================================================
# CALLBACK BACKEND: Solo actualiza Store (clientside renderiza)
# ============================================================================

@callback(
    Output("rt-simulation-data", "data"),
    Output("rt-progress-bar", "children"),
    Output("rt-process-interval", "disabled"),
    Input("rt-process-interval", "n_intervals"),
    State("rt-simulation-data", "data"),
    prevent_initial_call=True
)
def rt_process_batch(n_intervals, store_data):
    """
    Backend callback SIMPLE: Procesa 1 ventana a la vez (RÁPIDO).
    El clientside callback se encarga del rendering automáticamente.

    IMPORTANTE: Usa _realtime_cache para tracking de progreso, NO store_data["current_window"],
    porque State se captura cuando el callback se encola, no cuando se ejecuta.
    """
    global _realtime_cache

    # BLOQUEO: Si ya hay un callback procesando, saltar
    if _realtime_cache.get("is_processing", False):
        return no_update, no_update, no_update

    _realtime_cache["is_processing"] = True

    try:
        engine = _realtime_cache.get("engine")
        if not engine:
            _realtime_cache["is_processing"] = False
            return store_data, dbc.Alert("Error: Motor no disponible", color="danger"), True

        # LEER PROGRESO DEL CACHE (NO del store_data State, que puede estar desactualizado)
        current_window = _realtime_cache.get("current_window", 0)
        total_windows = _realtime_cache.get("total_windows", 0)

        # LEER RESULTS DEL CACHE también (mismo problema que current_window)
        if "results" not in _realtime_cache:
            _realtime_cache["results"] = []
        results = _realtime_cache["results"]
        batch_size = 1  # PROCESAR 1 VENTANA A LA VEZ (rápido)

        # Si ya terminamos
        if current_window >= total_windows:
            _realtime_cache["is_processing"] = False
            final_store_data = {
                "results": results,
                "current_window": current_window,
                "total_windows": total_windows,
                "inner_classes": results_data.get("inner_classes", []),
                "is_complete": True
            }
            print(f"\n[rt_process_batch] ✅ Completado: {len(results)} ventanas")

            progress_bar = dbc.Alert([
                html.H5([html.I(className="fas fa-check-circle me-2"), "✅ Completado"], className="mb-2"),
                html.P(f"Total: {len(results)} ventanas"),
                dbc.Progress(value=100, color="success", style={"height": "25px"})
            ], color="success")

            return final_store_data, progress_bar, True  # Deshabilitar interval

        # Procesar batch
        end_window = min(current_window + batch_size, total_windows)

        import time
        start_time = time.time()

        for i in range(current_window, end_window):
            try:
                result = engine.process_window(i)

                # Agregar nombre de clase para clientside
                if result.get('inner_prediction') is not None:
                    inner_classes = _realtime_cache.get("inner_classes", [])
                    idx = result['inner_prediction']
                    result['inner_class_name'] = inner_classes[idx] if idx < len(inner_classes) else f"C{idx}"

                results.append(result)

            except Exception as e:
                print(f"[rt_process_batch ERROR] ❌ Error procesando ventana {i}: {e}")
                import traceback
                traceback.print_exc()
                # Continuar con la siguiente ventana
                break

        elapsed = time.time() - start_time

        # ACTUALIZAR CACHE con nuevo progreso
        _realtime_cache["current_window"] = end_window

        # LIBERAR BLOQUEO
        _realtime_cache["is_processing"] = False

        # CREAR NUEVO DICCIONARIO (Dash necesita nuevo objeto para detectar cambio)
        new_store_data = {
            "results": results,
            "current_window": end_window,  # Solo para display en el clientside
            "total_windows": total_windows,
            "inner_classes": results_data.get("inner_classes", []) if store_data else _realtime_cache.get("inner_classes", []),
            "is_complete": False
        }

        progress_pct = (end_window / total_windows) * 100

        # Barra de progreso simple
        progress_bar = dbc.Alert([
            html.H5([html.I(className="fas fa-sync fa-spin me-2"), f"⚡ {progress_pct:.1f}%"], className="mb-2"),
            html.P(f"Ventanas: {end_window}/{total_windows}", className="mb-2"),
            dbc.Progress(value=progress_pct, striped=True, animated=True, color="info", style={"height": "25px"})
        ], color="info")

        # OPTIMIZACIÓN: Solo actualizar gráfico cada 50 ventanas (o al final)
        # Esto hace que la visualización sea mucho más rápida
        if end_window % 50 == 0 or end_window == total_windows:
            print(f"[rt_process_batch] ⏱️ {elapsed:.2f}s | ✅ {end_window}/{total_windows} ({progress_pct:.1f}%)")
            print(f"[rt_process_batch] 📤 Actualizando visualización con {len(results)} resultados")
            return new_store_data, progress_bar, False  # Actualizar Store y gráfico
        else:
            # Solo actualizar barra de progreso, NO el Store (para no re-renderizar el gráfico)
            return no_update, progress_bar, False

    except Exception as e:
        _realtime_cache["is_processing"] = False
        import traceback
        traceback.print_exc()
        return store_data, dbc.Alert(f"Error: {e}", color="danger"), True


# ============================================================================
# DEBUG CALLBACK: Verificar que el Store se actualiza
# ============================================================================

@callback(
    Output("rt-debug-info", "children"),
    Input("rt-simulation-data", "data"),
    prevent_initial_call=False
)
def debug_store_updates(store_data):
    """Callback de debug para verificar que el Store se actualiza."""
    if not store_data:
        return html.Div("DEBUG: Store vacío", style={"color": "yellow"})

    results_len = len(results_data.get("results", []))
    current_window = results_data.get("current_window", 0)
    total_windows = results_data.get("total_windows", 0)

    print(f"[DEBUG_CALLBACK] ✅ Store actualizado: results={results_len}, current_window={current_window}/{total_windows}")

    return html.Div([
        html.Strong("DEBUG Store: ", style={"color": "yellow"}),
        html.Span(f"Results: {results_len}, Window: {current_window}/{total_windows}", style={"color": "white"})
    ])


# ============================================================================
# CLIENTSIDE CALLBACK: Renderiza timeline en el navegador (SUPER RÁPIDO)
# ============================================================================

clientside_callback(
    """
    function(results_data, signal_data) {
        console.log('[CLIENTSIDE] Ejecutando callback');
        console.log('[CLIENTSIDE] Results:', results_data ? results_data.results.length : 0, 'resultados');
        console.log('[CLIENTSIDE] Signal data:', signal_data ? 'LOADED' : 'WAITING');

        // Validar que tengamos la señal
        if (!signal_data || !signal_data.data) {
            return {
                data: [],
                layout: {
                    title: "⏳ Cargando señal EEG...",
                    template: "plotly_dark",
                    height: 800
                }
            };
        }

        // Extraer datos de señal
        const signal = signal_data.data;
        const labels = signal_data.labels;
        const num_channels = signal_data.num_channels;
        const num_timepoints = signal_data.num_timepoints;
        const sfreq = signal_data.sfreq;
        const label_color_map = signal_data.label_color_map || {};

        // Vector de tiempo (en segundos)
        const time_vector = Array.from({length: num_timepoints}, (_, i) => i / sfreq);

        // Seleccionar canales a mostrar (primeros 8 canales para no saturar)
        const channels_to_show = Math.min(8, num_channels);
        const traces = [];

        // ========== RENDERIZAR SEÑAL EEG (subset de canales) ==========
        const channel_spacing = 50;  // Separación vertical entre canales (en microvolts)

        for (let ch = 0; ch < channels_to_show; ch++) {
            const channel_data = signal[ch];
            const offset = ch * channel_spacing;

            // Aplicar offset vertical para separar canales
            const y_data = channel_data.map(val => val * 1e6 + offset);  // Convertir a microvolts

            traces.push({
                x: time_vector,
                y: y_data,
                mode: 'lines',
                name: `Ch ${ch}`,
                line: {
                    color: 'rgba(100, 200, 255, 0.8)',
                    width: 0.5
                },
                hovertemplate: `Canal ${ch}<br>Tiempo: %{x:.2f}s<br>Amplitud: %{y:.1f} µV<extra></extra>`,
                showlegend: false
            });
        }

        // ========== AÑADIR REGIONES SOMBREADAS (Shapes) ==========
        const shapes = [];

        // Procesar resultados si existen
        if (results_data && results_data.results) {
            const results = results_data.results;
            const inner_classes = results_data.inner_classes || [];

            // Mapa de colores para Inner Speech
            const class_colors = {
                'rest': 'rgba(128, 128, 128, 0.3)',
                'arriba': 'rgba(255, 100, 100, 0.3)',
                'abajo': 'rgba(100, 255, 100, 0.3)',
                'izquierda': 'rgba(100, 100, 255, 0.3)',
                'derecha': 'rgba(255, 255, 100, 0.3)',
                'adelante': 'rgba(255, 100, 255, 0.3)',
                'atras': 'rgba(100, 255, 255, 0.3)'
            };

            results.forEach(r => {
                const start_time = r.start_sample / sfreq;
                const end_time = r.end_sample / sfreq;

                // Región sombreada para P300 detectado (cyan claro)
                if (r.p300_prediction === 1) {
                    shapes.push({
                        type: 'rect',
                        x0: start_time,
                        x1: end_time,
                        y0: -channel_spacing,
                        y1: channels_to_show * channel_spacing,
                        fillcolor: 'rgba(0, 255, 255, 0.15)',
                        line: {width: 0},
                        layer: 'below'
                    });
                }

                // Región sombreada para Inner Speech (coloreada por clase)
                if (r.inner_class_name) {
                    const color = class_colors[r.inner_class_name] || 'rgba(255, 0, 229, 0.3)';
                    shapes.push({
                        type: 'rect',
                        x0: start_time,
                        x1: end_time,
                        y0: -channel_spacing * 0.5,
                        y1: -channel_spacing * 0.2,
                        fillcolor: color,
                        line: {color: 'white', width: 1},
                        layer: 'above'
                    });

                    // Añadir anotación con el nombre de la clase
                    shapes.push({
                        type: 'line',
                        x0: (start_time + end_time) / 2,
                        x1: (start_time + end_time) / 2,
                        y0: -channel_spacing * 0.5,
                        y1: -channel_spacing * 0.2,
                        line: {color: 'white', width: 0}
                    });
                }
            });
        }

        // ========== LAYOUT ==========
        const num_results = results_data && results_data.results ? results_data.results.length : 0;
        const progress_pct = results_data ? (results_data.current_window / results_data.total_windows * 100).toFixed(1) : 0;

        const layout = {
            title: `🧠 Simulación EEG en Tiempo Real - ${num_results} ventanas procesadas (${progress_pct}%)`,
            xaxis: {
                title: 'Tiempo (segundos)',
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                showgrid: true,
                zeroline: false,
                rangeslider: {visible: false}  // Deshabilitar rangeslider para mejor rendimiento
            },
            yaxis: {
                title: 'Canales (µV)',
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                showgrid: false,
                zeroline: false,
                fixedrange: true  // Deshabilitar zoom vertical
            },
            plot_bgcolor: 'rgba(10, 1, 36, 0.95)',
            paper_bgcolor: 'rgba(10, 1, 36, 0.95)',
            font: {color: 'white', size: 11},
            height: 800,
            shapes: shapes,  // Añadir regiones sombreadas
            showlegend: false,
            hovermode: 'closest',
            margin: {t: 60, b: 60, l: 80, r: 40},
            annotations: []  // Podríamos añadir labels de clases aquí
        };

        console.log('[CLIENTSIDE] Renderizando', traces.length, 'canales +', shapes.length, 'regiones sombreadas');
        return {data: traces, layout: layout};
    }
    """,
    Output("rt-signal-graph", "figure"),
    Input("rt-simulation-data", "data"),
    Input("rt-full-signal-data", "data"),
    prevent_initial_call=False
)


# (Código viejo eliminado - ahora usamos sistema clientside)
