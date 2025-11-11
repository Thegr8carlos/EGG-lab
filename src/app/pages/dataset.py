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

from dash import html, dcc, register_page, callback, Output, Input, State, dash_table
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
import pandas as pd

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
TOPOMAP_CLASS_DROPDOWN = "topomap-class-dropdown"
TOPOMAP_TIME_SLIDER = "topomap-time-slider"
TOPOMAP_ANIMATION_GRAPH = "topomap-animation-graph"
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

    # Tabla de sesiones con paginación
    sessions = metadata.get("sessions", [])
    if sessions:
        # Convertir a DataFrame para DataTable
        session_data = []
        for sess in sessions:
            session_data.append({
                "Subject": sess.get("subject", "N/A"),
                "Session": sess.get("session", "N/A"),
                "Duración": f"{sess.get('duration_sec', 0):.1f}s",
                "Eventos": sess.get("n_events", 0),
                "Freq. Muestreo": f"{sess.get('sampling_rate', 0)} Hz"
            })

        df_sessions = pd.DataFrame(session_data)

        session_table = dash_table.DataTable(
            data=df_sessions.to_dict('records'),
            columns=[{"name": col, "id": col} for col in df_sessions.columns],
            page_size=10,
            page_action='native',
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'backgroundColor': 'rgba(10,1,36,0.3)',
                'color': '#F9F6FF',
                'border': '1px solid rgba(56, 255, 151, 0.2)',
                'fontSize': '0.85rem'
            },
            style_header={
                'backgroundColor': 'rgba(56, 255, 151, 0.2)',
                'fontWeight': 'bold',
                'color': '#38FF97',
                'border': '1px solid #38FF97'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgba(10,1,36,0.5)'
                }
            ]
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

    # Obtener clases disponibles
    classes = metadata.get("classes", [])

    return html.Div([
        html.P(f"🎯 Montage detectado: {montage.get('type', 'unknown')} ({montage.get('matched_channels', 0)}/{montage.get('total_channels', 0)} canales)",
               style={"color": "var(--text)", "fontSize": "0.95rem", "marginBottom": "1rem"}),

        # Topomap estático (ubicación de electrodos)
        dcc.Graph(figure=fig, id=TOPOMAP_GRAPH),

        # Sección de animación temporal
        html.Hr(style={"borderColor": "var(--accent-3)", "marginTop": "2rem", "marginBottom": "2rem"}),

        html.H5("⚡ Visualización Temporal de Eventos",
                style={"color": "var(--accent-3)", "marginBottom": "1.5rem"}),

        html.P([
            "Usa el ",
            html.Strong("Sidebar izquierdo", style={"color": "var(--accent-2)"}),
            " para seleccionar un archivo EEG. Luego selecciona la clase y el tiempo para ver la activación cerebral:"
        ], style={"color": "var(--text)", "marginBottom": "1.5rem"}),

        dbc.Row([
            dbc.Col([
                html.Label("🏷️ Clase:", style={"marginBottom": "0.5rem", "color": "var(--text)"}),
                dcc.Dropdown(
                    id=TOPOMAP_CLASS_DROPDOWN,
                    options=[{"label": c, "value": c} for c in classes],
                    value=classes[0] if classes else None,
                    placeholder="Selecciona una clase",
                    style={
                        "backgroundColor": "rgba(10,1,36,0.9)",
                        "color": "#F9F6FF"
                    },
                    className="custom-dropdown"
                ),
            ], width=12),
        ], style={"marginBottom": "1.5rem"}),

        html.Div([
            html.Label("⏱️ Tiempo del evento (segundos):", style={"marginBottom": "0.5rem", "color": "var(--text)"}),
            dcc.Slider(
                id=TOPOMAP_TIME_SLIDER,
                min=0,
                max=3.2,
                step=0.05,
                value=0.3,
                marks={i/10: f"{i/10:.1f}s" for i in range(0, 33, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={"marginBottom": "2rem"}),

        # Gráfico de animación
        dcc.Loading(
            id="loading-topomap-animation",
            type="circle",
            children=html.Div(id=TOPOMAP_ANIMATION_GRAPH)
        )
    ])


# ===== Callback: Animación Temporal del Topomap =====
@callback(
    Output(TOPOMAP_ANIMATION_GRAPH, "children"),
    [Input("selected-file-path", "data"),  # 🔥 Usar el sidebar!
     Input(TOPOMAP_CLASS_DROPDOWN, "value"),
     Input(TOPOMAP_TIME_SLIDER, "value")],
    State(SELECTED_DATASET_STORE, "data")
)
def update_topomap_animation(selected_file_path, class_name, time_point, metadata):
    """Muestra activación cerebral en un punto temporal específico de un evento."""
    if not metadata or not selected_file_path or not class_name or time_point is None:
        return html.Div("Usa el sidebar para seleccionar un archivo EEG, luego elige clase y tiempo",
                       style={"color": "var(--text)", "padding": "2rem", "textAlign": "center"})

    montage = metadata.get("montage", {})
    positions = montage.get("positions", {})

    if not positions:
        return html.Div("No hay posiciones de electrodos disponibles", style={"color": "var(--text)"})

    dataset_name = metadata.get("dataset_name", "")

    try:
        # Extraer el path del sidebar (igual que en filtros.py)
        if isinstance(selected_file_path, dict):
            candidate = selected_file_path.get("path") or selected_file_path.get("file") or ""
        else:
            candidate = str(selected_file_path)

        candidate = candidate.strip()
        if not candidate:
            return html.Div("Path inválido desde el sidebar", style={"color": "var(--text)"})

        # Usar la función de Dataset para obtener eventos por clase (igual que filtros)
        res = Dataset.get_events_by_class(candidate, class_name=class_name)
        first_evt = res.get("first_event_file") if isinstance(res, dict) else None

        if not first_evt:
            return html.Div(f"No se encontraron eventos de clase '{class_name}' en el archivo seleccionado",
                          style={"color": "var(--text)"})

        # Cargar el evento
        event_data = np.load(first_evt, allow_pickle=False)  # Shape: (n_channels, n_timepoints)

        # Obtener la frecuencia de muestreo y nombres de canales desde metadata
        try:
            from shared.fileUtils import get_dataset_metadata
            meta = get_dataset_metadata(candidate.split('/')[0])  # "nieto_inner_speech"
            sfreq = meta.get("sampling_frequency_hz", 250.0)

            # La key correcta es 'channel_names' (no 'channels')!
            ch_names = meta.get("channel_names", metadata.get("channel_names", []))

            print(f"[Topomap] Frecuencia: {sfreq} Hz, Canales encontrados: {len(ch_names) if ch_names else 0}")
        except Exception as e:
            print(f"[Topomap] Error obteniendo metadata: {e}")
            sfreq = 250.0
            ch_names = metadata.get("channel_names", [])

        # Si aún no hay canales, simplemente usar índices basados en el shape del evento
        if not ch_names or len(ch_names) == 0:
            print(f"[Topomap] No hay channel_names, generando nombres genéricos para {event_data.shape[0]} canales")
            ch_names = [f"Ch{i+1}" for i in range(event_data.shape[0])]

        # Calcular el índice de muestra correspondiente al tiempo seleccionado
        time_index = int(time_point * sfreq)

        if time_index >= event_data.shape[1]:
            time_index = event_data.shape[1] - 1

        # Extraer valores de todos los canales en este punto temporal
        channel_values = event_data[:, time_index]

        # Verificar que el número de canales coincide
        if len(ch_names) != len(channel_values):
            # Ajustar: si hay más canales en metadata, usar solo los primeros N
            if len(ch_names) > len(channel_values):
                ch_names = ch_names[:len(channel_values)]
                print(f"[Topomap] Ajustando canales: usando {len(channel_values)} canales")
            else:
                return html.Div(
                    f"Error: El evento tiene {len(channel_values)} canales pero metadata solo tiene {len(ch_names)}",
                    style={"color": "#FF235A"}
                )

        # Crear diccionario canal -> valor
        channel_value_dict = {ch: val for ch, val in zip(ch_names, channel_values)}

        # Crear topomap con valores
        x_coords = []
        y_coords = []
        values = []
        labels = []

        for ch in ch_names:
            if ch in positions and ch in channel_value_dict:
                pos = positions[ch]
                x_coords.append(pos["x"])
                y_coords.append(pos["y"])
                values.append(channel_value_dict[ch])
                labels.append(ch)

        # Normalizar valores para colormap
        values = np.array(values)
        v_min, v_max = values.min(), values.max()

        # Crear figura
        fig = go.Figure()

        # Círculo de cabeza
        theta = np.linspace(0, 2*np.pi, 100)
        head_radius = 0.1
        fig.add_trace(go.Scatter(
            x=head_radius * np.cos(theta),
            y=head_radius * np.sin(theta),
            mode='lines',
            line=dict(color='#00FFF0', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Nariz
        fig.add_trace(go.Scatter(
            x=[0, -0.01, 0.01, 0],
            y=[head_radius, head_radius + 0.015, head_radius + 0.015, head_radius],
            mode='lines',
            fill='toself',
            fillcolor='#00FFF0',
            line=dict(color='#00FFF0'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Electrodos con valores (colormap)
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(
                size=16,
                color=values,
                colorscale='RdBu_r',
                cmin=v_min,
                cmax=v_max,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Amplitud (V)",
                        font=dict(color='#F9F6FF')
                    ),
                    tickfont=dict(color='#F9F6FF'),
                    bgcolor='rgba(10,1,36,0.8)',
                    bordercolor='#38FF97',
                    borderwidth=1
                ),
                line=dict(color='#F9F6FF', width=1)
            ),
            text=labels,
            textposition='top center',
            textfont=dict(size=8, color='#F9F6FF'),
            hovertemplate='<b>%{text}</b><br>Valor: %{marker.color:.2e} V<extra></extra>',
            name='Electrodos'
        ))

        fig.update_layout(
            title=dict(
                text=f"🧠 Activación Cerebral - Clase: {class_name} - Tiempo: {time_point:.2f}s",
                font=dict(color='#F9F6FF')
            ),
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
            plot_bgcolor='rgba(10,1,36,0.2)',
            height=600,
            showlegend=False,
            font=dict(color='#F9F6FF')
        )

        return dcc.Graph(figure=fig)

    except Exception as e:
        return html.Div(f"Error al cargar eventos: {str(e)}", style={"color": "#FF235A"})


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

    # Crear tabla con estadísticas usando DataTable con paginación
    channel_data = []
    for ch, stats in channel_stats.items():
        channel_data.append({
            "Canal": ch,
            "Media (V)": f"{stats['mean']:.2e}",
            "Desv. Std (V)": f"{stats['std']:.2e}",
            "Mín (V)": f"{stats['min']:.2e}",
            "Máx (V)": f"{stats['max']:.2e}",
            "RMS (V)": f"{stats['rms']:.2e}"
        })

    df_channels = pd.DataFrame(channel_data)

    table = dash_table.DataTable(
        data=df_channels.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df_channels.columns],
        page_size=15,
        page_action='native',
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'backgroundColor': 'rgba(10,1,36,0.3)',
            'color': '#F9F6FF',
            'border': '1px solid rgba(56, 255, 151, 0.2)',
            'fontSize': '0.85rem'
        },
        style_header={
            'backgroundColor': 'rgba(56, 255, 151, 0.2)',
            'fontWeight': 'bold',
            'color': '#38FF97',
            'border': '1px solid #38FF97'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgba(10,1,36,0.5)'
            }
        ]
    )

    return html.Div([
        html.P(f"📊 Estadísticas calculadas sobre {len(channel_stats)} canales (primeros archivos del dataset)",
               style={"color": "var(--text)", "fontSize": "0.95rem", "marginBottom": "1rem"}),
        table
    ])


# ===== Callback: Sección ERP Promedio (antes Heatmap) ⭐ =====
@callback(
    Output(HEATMAP_SECTION, "children"),
    Input(SELECTED_DATASET_STORE, "data"),
)
def update_heatmap_section(metadata):
    """
    Muestra visualización ERP promedio por clase.
    ERP = Event-Related Potential (potencial relacionado a eventos).
    """
    if not metadata:
        return html.Div("Sin datos", style={"color": "var(--text)"})

    return html.Div([
        html.H5("📊 Análisis ERP (Event-Related Potential)",
                style={"color": "var(--accent-3)", "marginBottom": "1.5rem"}),

        html.P([
            "Esta visualización fue movida a la sección ",
            html.Strong("🧠 Topomap", style={"color": "var(--accent-2)"}),
            " donde puedes ver la activación cerebral espacialmente en tiempo real."
        ], style={"color": "var(--text)", "marginBottom": "2rem", "fontSize": "1rem"}),

        html.Div([
            html.P("💡 Análisis ERP Adicionales Disponibles:", style={"fontWeight": "bold", "color": "var(--accent-2)"}),
            html.Ul([
                html.Li("Ver promedio temporal de eventos por clase en la sección 〰️ Señal Raw", style={"color": "var(--text)"}),
                html.Li("Ver activación espacial dinámica en la sección 🧠 Topomap (scroll hacia abajo)", style={"color": "var(--text)"}),
                html.Li("Ver distribución de potencia espectral en la sección 🌊 Frecuencias", style={"color": "var(--text)"}),
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

        # Gráfico de pastel para balance de clases
        dcc.Graph(figure=fig_pie),

        # Información adicional de calidad de señal
        html.Hr(style={"borderColor": "var(--accent-3)", "marginTop": "2rem", "marginBottom": "2rem"}),

        html.H5("📡 Calidad de Señal por Canal",
                style={"color": "var(--accent-3)", "marginBottom": "1.5rem"}),

        html.Div([
            html.P("💡 Métricas de Calidad:", style={"fontWeight": "bold", "color": "var(--text)", "marginBottom": "1rem"}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("✓ ", style={"color": "#38FF97", "fontSize": "1.2rem"}),
                        html.Span("RMS (Root Mean Square): ", style={"fontWeight": "bold", "color": "var(--text)"}),
                        html.Span("Potencia promedio de la señal", style={"color": "var(--text-muted)"})
                    ], style={"marginBottom": "0.5rem"}),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Span("✓ ", style={"color": "#38FF97", "fontSize": "1.2rem"}),
                        html.Span("Desv. Std: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                        html.Span("Variabilidad de la señal", style={"color": "var(--text-muted)"})
                    ], style={"marginBottom": "0.5rem"}),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("✓ ", style={"color": "#38FF97", "fontSize": "1.2rem"}),
                        html.Span("Rango dinámico: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                        html.Span("Diferencia entre máximo y mínimo", style={"color": "var(--text-muted)"})
                    ], style={"marginBottom": "0.5rem"}),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Span("⚠ ", style={"color": "#FFD400", "fontSize": "1.2rem"}),
                        html.Span("Bad channels: ", style={"fontWeight": "bold", "color": "var(--text)"}),
                        html.Span("Canales con ruido excesivo o señal plana", style={"color": "var(--text-muted)"})
                    ], style={"marginBottom": "0.5rem"}),
                ], width=6),
            ])
        ], style={"padding": "1rem", "backgroundColor": "rgba(10,1,36,0.3)", "borderRadius": "8px", "marginBottom": "2rem"}),

        # Gráfico de barras: RMS por canal (top 10)
        html.Div([
            html.P("📊 Puedes ver el RMS y otras estadísticas de cada canal en la pestaña '📊 Por Canal'",
                   style={"color": "var(--text)", "fontSize": "0.9rem", "fontStyle": "italic"})
        ])
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
