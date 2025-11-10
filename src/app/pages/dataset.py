"""
Dashboard de Análisis de Dataset - Página de Visualización Científica

Paleta de Colores (definida en assets/palette.css):
--------------------------------------------------
Colores Principales:
  --color-6 / --accent-3: #38FF97  (Verde brillante - principal)
  --color-5 / --accent-2: #00FFF0  (Cyan - secundario)
  --color-4 / --accent-1: #FF00E5  (Magenta - acentos)
  --color-3 / --text:     #F9F6FF  (Blanco nieve - texto)

Colores de Frecuencias:
  Delta:  #FF235A  (Rojo)
  Theta:  #FF00E5  (Magenta)
  Alpha:  #00FFF0  (Cyan)
  Beta:   #38FF97  (Verde)
  Gamma:  #FFD400  (Amarillo)

Superficies:
  --card-bg:        Transparencia del surface-2
  --card-bg-strong: Transparencia más fuerte
  --border-weak:    Borde sutil
  --border-strong:  Borde con accent-1
"""

from dash import html, dcc, register_page, callback, Output, Input, State
import dash_bootstrap_components as dbc
from backend.classes.dataset import Dataset
from app.components.SideBar import get_sideBar
from app.components.DataView import (
    get_dataset_view,
    register_dataset_clientside,
    register_dataset_legend,
)
import json
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

register_page(__name__, path="/dataset", name="Dataset")

# ===== IDs =====
SELECTED_DATASET_STORE = "selected-dataset-metadata"
METADATA_DISPLAY = "dataset-metadata-display"
OVERVIEW_PANEL = "dataset-overview-panel"
STATS_SECTION = "dataset-stats-section"
SPATIAL_SECTION = "dataset-spatial-section"
SIGNAL_SECTION = "dataset-signal-section"
RAW_SIGNAL_SECTION = "dataset-raw-signal-section"
HEATMAP_SECTION = "dataset-heatmap-section"
FREQUENCY_SECTION = "dataset-frequency-section"
QUALITY_SECTION = "dataset-quality-section"

# Topomap IDs
TOPOMAP_GRAPH = "topomap-graph"
HEATMAP_TOPOMAP = "heatmap-topomap"
HEATMAP_CLASS_DROPDOWN = "heatmap-class-dropdown"
HEATMAP_TIME_SLIDER = "heatmap-time-slider"

# IDs para el plot original de señal raw
CONTAINER_ID = "dataset-view-dataset"
STORE_ID = "full-signal-data-dataset"
LABEL_STORE = "label-color-store-dataset"
LEGEND_ID = "dynamic-color-legend-dataset"
GRAPH_ID = "signal-graph-dataset"
INTERVAL_ID = "interval-component-dataset"

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
                # Store para metadata del dataset seleccionado
                dcc.Store(id=SELECTED_DATASET_STORE),

                # Header
                html.H2("📊 Análisis de Dataset", style={"marginBottom": "1rem", "color": "var(--text)"}),

                # Overview panel (siempre visible)
                html.Div(id=OVERVIEW_PANEL, style={"marginBottom": "1.5rem"}),

                # Tabs horizontales (solo una abierta a la vez)
                dbc.Tabs(
                    [
                        dbc.Tab(
                            html.Div(id=STATS_SECTION, style={"padding": "1.5rem"}),
                            label="📈 Estadísticas",
                            tab_id="tab-stats",
                        ),
                        dbc.Tab(
                            html.Div(id=SPATIAL_SECTION, style={"padding": "1.5rem"}),
                            label="🧠 Topomap",
                            tab_id="tab-spatial",
                        ),
                        dbc.Tab(
                            html.Div(id=SIGNAL_SECTION, style={"padding": "1.5rem"}),
                            label="📊 Por Canal",
                            tab_id="tab-signal",
                        ),
                        dbc.Tab(
                            html.Div(
                                get_dataset_view(
                                    container_id=CONTAINER_ID,
                                    full_signal_store_id=STORE_ID,
                                    label_color_store_id=LABEL_STORE,
                                    legend_container_id=LEGEND_ID,
                                    graph_id=GRAPH_ID,
                                    interval_id=INTERVAL_ID,
                                ),
                                style={"padding": "1.5rem"}
                            ),
                            label="〰️ Señal Raw",
                            tab_id="tab-raw",
                        ),
                        dbc.Tab(
                            html.Div(id=HEATMAP_SECTION, style={"padding": "1.5rem"}),
                            label="🗺️ Heatmap",
                            tab_id="tab-heatmap",
                        ),
                        dbc.Tab(
                            html.Div(id=FREQUENCY_SECTION, style={"padding": "1.5rem"}),
                            label="🌊 Frecuencias",
                            tab_id="tab-frequency",
                        ),
                        dbc.Tab(
                            html.Div(id=QUALITY_SECTION, style={"padding": "1.5rem"}),
                            label="✨ Calidad",
                            tab_id="tab-quality",
                        ),
                    ],
                    id="dataset-tabs",
                    active_tab="tab-stats",
                    style={"marginBottom": "2rem"}
                ),
            ],
            style={
                "flex": "1",
                "padding": "1rem 2rem",
                "maxWidth": "1600px",
                "margin": "0 auto",
            },
        ),
    ],
    style={"display": "flex", "minHeight": "100vh"},
)


# ===== Callback: Cargar metadata del dataset seleccionado =====
@callback(
    Output(SELECTED_DATASET_STORE, "data"),
    Input("selected-dataset", "data"),
)
def load_dataset_metadata(selected_dataset):
    """Carga el archivo dataset_metadata.json del dataset seleccionado."""
    if not selected_dataset:
        return None

    try:
        metadata_path = Path(f"Aux/{selected_dataset}/dataset_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"[Dataset Page] Metadata cargado para {selected_dataset}")
            return metadata
        else:
            print(f"[Dataset Page] No se encontró metadata para {selected_dataset}")
            return None
    except Exception as e:
        print(f"[Dataset Page] Error cargando metadata: {e}")
        return None


# ===== Callback: Overview Panel =====
@callback(
    Output(OVERVIEW_PANEL, "children"),
    Input(SELECTED_DATASET_STORE, "data"),
)
def update_overview(metadata):
    """Muestra resumen general del dataset."""
    if not metadata:
        return html.Div(
            "Selecciona un dataset del árbol de archivos para ver su análisis.",
            style={"color": "var(--text)", "padding": "2rem", "textAlign": "center"}
        )

    # Calcular score de calidad (estrellas)
    quality = metadata.get("quality_metrics", {})
    balance = quality.get("events_balance", "unknown")
    bad_channels = quality.get("total_bad_channels", 0)

    quality_score = 5
    if balance == "imbalanced":
        quality_score -= 1
    if bad_channels > 0:
        quality_score -= min(2, bad_channels)

    stars = "⭐" * quality_score + "☆" * (5 - quality_score)

    return dbc.Card(
        dbc.CardBody([
            html.H4(f"📂 {metadata['dataset_name']}", style={"marginBottom": "1rem", "color": "var(--text)"}),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Strong("Clases: ", style={"color": "var(--accent-3)"}),
                        html.Span(f"{metadata['num_classes']} ({', '.join(metadata['classes'][:3])}{'...' if len(metadata['classes']) > 3 else ''})")
                    ], width=3),
                    dbc.Col([
                        html.Strong("Archivos: ", style={"color": "var(--accent-3)"}),
                        html.Span(f"{metadata.get('total_files', 'N/A')}")
                    ], width=2),
                    dbc.Col([
                        html.Strong("Frecuencia: ", style={"color": "var(--accent-3)"}),
                        html.Span(f"{metadata['sampling_frequency_hz']} Hz")
                    ], width=2),
                    dbc.Col([
                        html.Strong("Duración: ", style={"color": "var(--accent-3)"}),
                        html.Span(f"{metadata['total_duration_sec'] / 3600:.1f}h")
                    ], width=2),
                    dbc.Col([
                        html.Strong("Calidad: ", style={"color": "var(--accent-3)"}),
                        html.Span(f"{stars} ({balance})")
                    ], width=3),
                ]),
            ], style={"fontSize": "0.95rem", "color": "var(--text)"})
        ]),
        style={
            "backgroundColor": "var(--card-bg-strong)",
            "border": "1px solid var(--border-strong)",
            "borderRadius": "var(--radius-md)"
        }
    )


# ===== Callback: Sección de Estadísticas Generales =====
@callback(
    Output(STATS_SECTION, "children"),
    Input(SELECTED_DATASET_STORE, "data"),
)
def update_stats_section(metadata):
    """Muestra estadísticas generales: eventos por clase, sesiones, etc."""
    if not metadata:
        return html.Div("Sin datos", style={"color": "var(--text)"})

    # Gráfico de barras: eventos por clase
    class_counts = metadata.get("class_counts_total", {})
    fig_classes = go.Figure(data=[
        go.Bar(
            x=list(class_counts.keys()),
            y=list(class_counts.values()),
            marker_color='#38FF97',  # var(--accent-3) / var(--color-6)
            marker_line=dict(color='#7DFFC9', width=1.5)  # var(--color-14) para borde
        )
    ])
    fig_classes.update_layout(
        title=dict(text="Distribución de Eventos por Clase", font=dict(color='#F9F6FF')),  # var(--text)
        xaxis_title="Clase",
        yaxis_title="Número de Eventos",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,1,36,0.3)',  # var(--color-2) con transparencia
        height=400,
        font=dict(color='#F9F6FF')  # var(--text)
    )

    # Tabla de sesiones
    sessions = metadata.get("sessions", [])
    if sessions:
        session_rows = []
        for i, sess in enumerate(sessions[:10]):  # Mostrar primeras 10
            session_rows.append(html.Tr([
                html.Td(sess.get("subject", "N/A")),
                html.Td(sess.get("session", "N/A")),
                html.Td(f"{sess.get('duration_sec', 0):.1f}s"),
                html.Td(sess.get("n_events", 0)),
                html.Td(sess.get("sampling_rate", 0)),
            ]))

        if len(sessions) > 10:
            session_rows.append(html.Tr([
                html.Td("...", colSpan=5, style={"textAlign": "center", "fontStyle": "italic"})
            ]))

        session_table = dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Subject"),
                    html.Th("Session"),
                    html.Th("Duración"),
                    html.Th("Eventos"),
                    html.Th("Freq. Muestreo"),
                ])),
                html.Tbody(session_rows)
            ],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            style={"fontSize": "0.85rem"}
        )
    else:
        session_table = html.Div("No hay información de sesiones", style={"color": "var(--text)"})

    return html.Div([
        dcc.Graph(figure=fig_classes),
        html.H5(f"📋 Sesiones ({len(sessions)} total)",
                style={
                    "marginTop": "2rem",
                    "marginBottom": "1rem",
                    "color": "var(--text)",
                    "borderBottom": "2px solid var(--accent-3)",
                    "paddingBottom": "0.5rem"
                }),
        session_table
    ])


# ===== Callback: Sección Espacial (Topomap) =====
@callback(
    Output(SPATIAL_SECTION, "children"),
    Input(SELECTED_DATASET_STORE, "data"),
)
def update_spatial_section(metadata):
    """Muestra topomap con ubicación de electrodos."""
    if not metadata:
        return html.Div("Sin datos", style={"color": "var(--text)"})

    montage = metadata.get("montage", {})

    if not montage.get("has_positions", False):
        return html.Div([
            html.P("⚠️ No se pudo inferir la ubicación de los electrodos."),
            html.P(f"Montage detectado: {montage.get('type', 'unknown')}"),
            html.P("Los canales de este dataset no coinciden con sistemas estándar (10-20, BioSemi, etc.).")
        ], style={"color": "var(--text)", "padding": "1rem"})

    # Crear topomap 2D
    positions = montage.get("positions", {})
    if not positions:
        return html.Div("No hay posiciones disponibles", style={"color": "var(--text)"})

    # Extraer coordenadas 2D (proyección desde arriba: x, y)
    ch_names = []
    x_coords = []
    y_coords = []

    for ch, pos in positions.items():
        ch_names.append(ch)
        x_coords.append(pos["x"])
        y_coords.append(pos["y"])

    # Crear scatter plot con círculo de cabeza
    fig = go.Figure()

    # Círculo de cabeza
    theta = np.linspace(0, 2*np.pi, 100)
    head_radius = 0.1
    fig.add_trace(go.Scatter(
        x=head_radius * np.cos(theta),
        y=head_radius * np.sin(theta),
        mode='lines',
        line=dict(color='#00FFF0', width=2),  # var(--accent-2) / var(--color-5)
        showlegend=False,
        hoverinfo='skip'
    ))

    # Nariz (triángulo arriba)
    fig.add_trace(go.Scatter(
        x=[0, -0.01, 0.01, 0],
        y=[head_radius, head_radius + 0.015, head_radius + 0.015, head_radius],
        mode='lines',
        fill='toself',
        fillcolor='#00FFF0',  # var(--accent-2)
        line=dict(color='#00FFF0'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Electrodos
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        marker=dict(
            size=12,
            color='#38FF97',  # var(--accent-3) / var(--color-6)
            line=dict(color='#7DFFC9', width=1.5)  # var(--color-14) para borde
        ),
        text=ch_names,
        textposition='top center',
        textfont=dict(size=9, color='#F9F6FF'),  # var(--text)
        hovertemplate='<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
        name='Electrodos'
    ))

    fig.update_layout(
        title=dict(text=f"🧠 Mapa de Electrodos ({montage.get('type', 'unknown')})", font=dict(color='#F9F6FF')),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.15, 0.15]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.15, 0.15],
            scaleanchor="x",
            scaleratio=1
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,1,36,0.2)',  # var(--color-2) con transparencia
        height=600,
        showlegend=False,
        font=dict(color='#F9F6FF')
    )

    return html.Div([
        html.P(f"🎯 Montage detectado: {montage.get('type', 'unknown')} ({montage.get('matched_channels', 0)}/{montage.get('total_channels', 0)} canales)",
               style={"color": "var(--text)", "fontSize": "0.95rem", "marginBottom": "1rem"}),
        dcc.Graph(figure=fig, id=TOPOMAP_GRAPH)
    ])


# ===== Callback: Sección de Análisis de Señal =====
@callback(
    Output(SIGNAL_SECTION, "children"),
    Input(SELECTED_DATASET_STORE, "data"),
)
def update_signal_section(metadata):
    """Muestra estadísticas de señal por canal."""
    if not metadata:
        return html.Div("Sin datos", style={"color": "var(--text)"})

    channel_stats = metadata.get("channel_stats", {})

    if not channel_stats:
        return html.Div("No hay estadísticas de canales disponibles", style={"color": "var(--text)"})

    # Crear tabla con estadísticas
    rows = []
    for ch, stats in list(channel_stats.items())[:20]:  # Primeros 20 canales
        rows.append(html.Tr([
            html.Td(ch),
            html.Td(f"{stats['mean']:.2e}"),
            html.Td(f"{stats['std']:.2e}"),
            html.Td(f"{stats['min']:.2e}"),
            html.Td(f"{stats['max']:.2e}"),
            html.Td(f"{stats['rms']:.2e}"),
        ]))

    if len(channel_stats) > 20:
        rows.append(html.Tr([
            html.Td("...", colSpan=6, style={"textAlign": "center", "fontStyle": "italic"})
        ]))

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("Canal"),
                html.Th("Media (V)"),
                html.Th("Desv. Std (V)"),
                html.Th("Mín (V)"),
                html.Th("Máx (V)"),
                html.Th("RMS (V)"),
            ])),
            html.Tbody(rows)
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        style={"fontSize": "0.85rem"}
    )

    return html.Div([
        html.P(f"📊 Estadísticas calculadas sobre {len(channel_stats)} canales (primeros archivos del dataset)",
               style={"color": "var(--text)", "fontSize": "0.95rem", "marginBottom": "1rem"}),
        table
    ])


# ===== Callback: Sección de Heatmap Temporal ⭐ =====
@callback(
    Output(HEATMAP_SECTION, "children"),
    Input(SELECTED_DATASET_STORE, "data"),
)
def update_heatmap_section(metadata):
    """Sección de heatmap temporal (placeholder por ahora)."""
    if not metadata:
        return html.Div("Sin datos", style={"color": "var(--text)"})

    classes = metadata.get("classes", [])
    montage = metadata.get("montage", {})

    if not montage.get("has_positions", False):
        return html.Div([
            html.P("⚠️ El heatmap temporal requiere posiciones de electrodos."),
            html.P("No se pudo inferir el montage para este dataset.")
        ], style={"color": "var(--text)", "padding": "1rem"})

    return html.Div([
        html.P("🗺️ Esta sección mostrará la activación cerebral en tiempo real para cada clase.",
               style={"marginBottom": "1rem", "color": "var(--text)", "fontSize": "1rem"}),

        dbc.Row([
            dbc.Col([
                html.Label("Selecciona una clase:", style={"marginBottom": "0.5rem", "color": "var(--accent-3)"}),
                dcc.Dropdown(
                    id=HEATMAP_CLASS_DROPDOWN,
                    options=[{"label": c, "value": c} for c in classes],
                    value=classes[0] if classes else None,
                    style={"backgroundColor": "var(--surface-1)", "color": "var(--text)"}
                ),
            ], width=4),
        ], style={"marginBottom": "1.5rem"}),

        html.Div([
            html.Label("⏱️ Tiempo (desliza para ver animación):", style={"marginBottom": "0.5rem", "color": "var(--accent-3)"}),
            dcc.Slider(
                id=HEATMAP_TIME_SLIDER,
                min=0,
                max=3.2,
                step=0.1,
                value=0,
                marks={i: f"{i}s" for i in np.arange(0, 3.3, 0.5)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={"marginBottom": "2rem"}),

        dcc.Graph(id=HEATMAP_TOPOMAP),

        html.Div([
            html.P("💡 Próximamente:", style={"fontWeight": "bold", "marginTop": "2rem", "color": "var(--accent-2)"}),
            html.Ul([
                html.Li("Cálculo del promedio de eventos por clase", style={"color": "var(--text-muted)"}),
                html.Li("Interpolación espacial de valores", style={"color": "var(--text-muted)"}),
                html.Li("Animación automática con play button", style={"color": "var(--text-muted)"}),
            ])
        ], style={"fontSize": "0.85rem"})
    ])


# ===== Callback: Sección de Frecuencias =====
@callback(
    Output(FREQUENCY_SECTION, "children"),
    Input(SELECTED_DATASET_STORE, "data"),
)
def update_frequency_section(metadata):
    """Muestra análisis de bandas de frecuencia."""
    if not metadata:
        return html.Div("Sin datos", style={"color": "var(--text)"})

    freq_bands = metadata.get("frequency_bands", {})

    if not freq_bands:
        return html.Div("No hay datos de bandas de frecuencia disponibles", style={"color": "var(--text)"})

    # Gráfico de barras: potencia por banda
    bands = list(freq_bands.keys())
    powers = [freq_bands[b]["mean_power"] for b in bands]

    # Colores de la paleta para cada banda
    band_colors = {
        'delta': '#FF235A',   # var(--color-7) - rojo
        'theta': '#FF00E5',   # var(--color-4) - magenta
        'alpha': '#00FFF0',   # var(--color-5) - cyan
        'beta': '#38FF97',    # var(--color-6) - verde
        'gamma': '#FFD400'    # var(--color-8) - amarillo
    }
    colors = [band_colors.get(band, '#38FF97') for band in bands]

    fig = go.Figure(data=[
        go.Bar(
            x=bands,
            y=powers,
            marker_color=colors,
            marker_line=dict(color='#F9F6FF', width=1),  # var(--text) para borde
            text=[f"{p:.2e}" for p in powers],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title=dict(text="🌊 Potencia Promedio por Banda de Frecuencia", font=dict(color='#F9F6FF')),
        xaxis_title="Banda",
        yaxis_title="Potencia (V²/Hz)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,1,36,0.3)',
        height=400,
        font=dict(color='#F9F6FF')
    )

    # Tabla con rangos
    rows = []
    for band, data in freq_bands.items():
        rows.append(html.Tr([
            html.Td(band.capitalize()),
            html.Td(f"{data['range'][0]} - {data['range'][1]} Hz"),
            html.Td(f"{data['mean_power']:.2e} V²/Hz"),
        ]))

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("Banda"),
                html.Th("Rango de Frecuencia"),
                html.Th("Potencia Promedio"),
            ])),
            html.Tbody(rows)
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        style={"fontSize": "0.85rem", "marginTop": "2rem"}
    )

    return html.Div([
        dcc.Graph(figure=fig),
        table
    ])


# ===== Callback: Sección de Calidad =====
@callback(
    Output(QUALITY_SECTION, "children"),
    Input(SELECTED_DATASET_STORE, "data"),
)
def update_quality_section(metadata):
    """Muestra métricas de calidad del dataset."""
    if not metadata:
        return html.Div("Sin datos", style={"color": "var(--text)"})

    quality = metadata.get("quality_metrics", {})

    if not quality:
        return html.Div("No hay métricas de calidad disponibles", style={"color": "var(--text)"})

    bad_channels = quality.get("bad_channels_list", [])
    balance = quality.get("events_balance", "unknown")
    imbalance_ratio = quality.get("class_imbalance_ratio", 1.0)

    # Gráfico de pastel: balance de clases
    class_counts = metadata.get("class_counts_total", {})
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(class_counts.keys()),
        values=list(class_counts.values()),
        hole=0.3,
        marker=dict(
            colors=['#38FF97', '#00FFF0', '#FF00E5', '#FFD400', '#FF235A', '#FF66F2', '#6BFFF1', '#7DFFC9'],
            line=dict(color='#F9F6FF', width=2)
        )
    )])
    fig_pie.update_layout(
        title=dict(text="⚖️ Balance de Clases", font=dict(color='#F9F6FF')),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,1,36,0.3)',
        height=400,
        font=dict(color='#F9F6FF')
    )

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("⚠️ Bad Channels", style={"color": "var(--text)", "fontSize": "1rem"}),
                        html.H3(
                            quality.get("total_bad_channels", 0),
                            style={"color": "#FF235A" if bad_channels else "#38FF97", "marginTop": "0.5rem"}
                        ),
                        html.P(
                            ", ".join(bad_channels) if bad_channels else "Ninguno",
                            style={"fontSize": "0.85rem", "color": "var(--text-muted)", "marginTop": "0.5rem"}
                        )
                    ])
                ], style={
                    "backgroundColor": "var(--card-bg)",
                    "border": "1px solid var(--border-weak)",
                    "borderRadius": "var(--radius-md)"
                })
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("⚖️ Balance de Clases", style={"color": "var(--text)", "fontSize": "1rem"}),
                        html.H3(
                            balance.capitalize(),
                            style={"color": "#38FF97" if balance == "balanced" else "#FFD400", "marginTop": "0.5rem"}
                        ),
                        html.P(
                            f"Ratio: {imbalance_ratio:.2f}",
                            style={"fontSize": "0.85rem", "color": "var(--text-muted)", "marginTop": "0.5rem"}
                        )
                    ])
                ], style={
                    "backgroundColor": "var(--card-bg)",
                    "border": "1px solid var(--border-weak)",
                    "borderRadius": "var(--radius-md)"
                })
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("📊 Eventos Totales", style={"color": "var(--text)", "fontSize": "1rem"}),
                        html.H3(
                            quality.get("n_events_total", 0),
                            style={"color": "var(--accent-3)", "marginTop": "0.5rem"}
                        ),
                        html.P(
                            f"En {quality.get('n_sessions', 0)} sesiones",
                            style={"fontSize": "0.85rem", "color": "var(--text-muted)", "marginTop": "0.5rem"}
                        )
                    ])
                ], style={
                    "backgroundColor": "var(--card-bg)",
                    "border": "1px solid var(--border-weak)",
                    "borderRadius": "var(--radius-md)"
                })
            ], width=4),
        ], style={"marginBottom": "2rem"}),

        dcc.Graph(figure=fig_pie)
    ])


# ===== Callbacks para el plot de señal raw original =====
@callback(
    Output(STORE_ID, "data"),
    Output(INTERVAL_ID, "disabled"),
    Input("selected-file-path", "data"),
)
def load_signal_data_dataset(selected_file_path):
    """Carga los datos de señal raw desde archivo seleccionado."""
    return Dataset.load_signal_data(selected_file_path)


# Registrar callbacks clientside para el plot
register_dataset_clientside(graph_id=GRAPH_ID, interval_id=INTERVAL_ID, store_id=STORE_ID)
register_dataset_legend(legend_container_id=LEGEND_ID, store_id=STORE_ID)
