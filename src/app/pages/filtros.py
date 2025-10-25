# filtros.py - SOLUCIÓN HÍBRIDA: WebGL con controles de navegación
from pathlib import Path
import time
import numpy as np
import dash
from dash import html, dcc, register_page, callback, Output, Input, State, clientside_callback, no_update, ALL, ctx
from shared.fileUtils import get_dataset_metadata
import dash_bootstrap_components as dbc

from app.components.PageContainer import get_page_container
from app.components.PlayGround import get_playGround
from app.components.RigthComlumn import get_rightColumn
from app.components.SideBar import get_sideBar

from backend.classes.dataset import Dataset

register_page(__name__, path="/filtros", name="Filtros")

GRAPH_ID = "pg-main-plot-filtros"
EVENTS_STORE_ID = "events-store-filtros"
DATA_STORE_ID = "signal-store-filtros"
FILTERED_DATA_STORE_ID = "filtered-signal-store-filtros"
CHANNEL_RANGE_STORE = "channel-range-store"
SELECTED_CLASS_STORE = "selected-class-store"
SELECTED_CHANNELS_STORE = "selected-channels-store"  # Nuevo: canales específicos seleccionados

layout = html.Div(
    [
        html.Div(
            id="sidebar-wrapper",
            children=[get_sideBar("Data")],
            className="sideBar-container",
            style={"width": "260px", "padding": "1rem"},
        ),
        html.Div(
            id="pg-wrapper-filtros",
            children=get_playGround("Filtros", "Description", {}, {}, graph_id=GRAPH_ID, multi=True),
            style={"flex": "1", "padding": "1rem"},
        ),
        html.Div(
            get_rightColumn("filter"),
            style={"width": "340px", "padding": "1rem"},
        ),
        dcc.Store(id=EVENTS_STORE_ID),
        dcc.Store(id=DATA_STORE_ID),
        dcc.Store(id=FILTERED_DATA_STORE_ID),
        dcc.Store(id=CHANNEL_RANGE_STORE, data={"start": 0, "count": 8}),
        dcc.Store(id=SELECTED_CLASS_STORE, data=None),
        dcc.Store(id=SELECTED_CHANNELS_STORE, data=None),  # Nuevo: canales seleccionados
    ],
    style={"display": "flex"},
)

def create_metadata_section(meta: dict):
    if not isinstance(meta, dict):
        return {}, {}
    classes = meta.get("classes", []) or []
    class_color_map = {}
    for idx, label in enumerate(classes):
        hue = (idx * 47) % 360
        class_color_map[str(label)] = f"hsl({hue}, 70%, 50%)"
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
        # Navegación de canales (sin título)
        html.Div([
            html.Div(
                id='channel-nav-info',
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
                    id='btn-prev-channels',
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
                    id='btn-next-channels',
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
                    id='btn-all-classes',
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
                    id={'type': 'btn-filter-class', 'index': idx},
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

        # Selector de canales específicos ✅ MEJORADO CON CHECKLIST
        html.Div([
            # Header con título y botones de ayuda
            html.Div([
                html.Div("Canales específicos", style={
                    "fontSize": "10px",
                    "fontWeight": "600",
                    "color": "var(--text)",
                    "flex": "1"
                }),
                html.Div([
                    html.Button("Todos", id="btn-select-all-channels", n_clicks=0, style={
                        "padding": "2px 6px",
                        "fontSize": "8px",
                        "borderRadius": "3px",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer",
                        "marginRight": "4px"
                    }),
                    html.Button("Limpiar", id="btn-clear-channels", n_clicks=0, style={
                        "padding": "2px 6px",
                        "fontSize": "8px",
                        "borderRadius": "3px",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer",
                        "marginRight": "4px"
                    }),
                    html.Button("Solo EEG", id="btn-only-eeg-channels", n_clicks=0, style={
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
                    id='checklist-channel-selection',
                    options=[],  # Se llena dinámicamente
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

            # Contador de canales seleccionados
            html.Div(id="channel-count-display", children="0 canales seleccionados", style={
                "fontSize": "8px",
                "color": "var(--text-muted)",
                "marginTop": "4px",
                "textAlign": "right"
            })
        ])
    ])


@callback(
    Output("pg-wrapper-filtros", "children"),
    Input("selected-dataset", "data")
)
def update_playground_desc(selected_dataset):
    desc = selected_dataset or "Selecciona un dataset en 'Cargar Datos'"
    if not selected_dataset:
        return get_playGround("Filtros", desc, {}, {}, graph_id=GRAPH_ID, multi=True)
    try:
        meta = get_dataset_metadata(selected_dataset)
    except Exception as e:
        return get_playGround("Filtros", f"{desc} (sin metadata: {e})", {}, {}, graph_id=GRAPH_ID, multi=True)

    meta_dict, custom_dict = create_metadata_section(meta)
    nav_controls = create_navigation_controls(meta)
    return get_playGround("Filtros", desc, meta_dict, custom_dict, graph_id=GRAPH_ID, multi=True, navigation_controls=nav_controls)


@callback(
    Output('checklist-channel-selection', 'options'),
    Input('selected-dataset', 'data')
)
def populate_channel_checklist(selected_dataset):
    """Llena el checklist de canales cuando se selecciona un dataset"""
    if not selected_dataset:
        return []

    try:
        # Obtener nombres de canales usando la nueva función
        channel_names = Dataset.get_all_channel_names(selected_dataset)

        if not channel_names:
            print(f"[populate_channel_checklist] No se encontraron canales para {selected_dataset}")
            return []

        # Crear opciones para el checklist
        options = [{"label": ch, "value": ch} for ch in channel_names]

        print(f"[populate_channel_checklist] Cargados {len(options)} canales para {selected_dataset}")
        return options

    except Exception as e:
        print(f"[populate_channel_checklist] ERROR: {e}")
        return []


@callback(
    Output(SELECTED_CHANNELS_STORE, 'data'),
    Input('checklist-channel-selection', 'value')
)
def save_selected_channels(selected_channels):
    """Guarda los canales seleccionados en el store"""
    if not selected_channels or len(selected_channels) == 0:
        print("[save_selected_channels] Ningún canal seleccionado, mostrando todos")
        return None  # None = mostrar todos los canales

    print(f"[save_selected_channels] Canales seleccionados: {selected_channels}")
    return selected_channels


@callback(
    Output('channel-count-display', 'children'),
    Input('checklist-channel-selection', 'value')
)
def update_channel_count(selected_channels):
    """Actualiza el contador de canales seleccionados"""
    count = len(selected_channels) if selected_channels else 0
    if count == 0:
        return "Todos los canales"
    elif count == 1:
        return "1 canal seleccionado"
    else:
        return f"{count} canales seleccionados"


@callback(
    Output('checklist-channel-selection', 'value'),
    [
        Input('btn-select-all-channels', 'n_clicks'),
        Input('btn-clear-channels', 'n_clicks'),
        Input('btn-only-eeg-channels', 'n_clicks')
    ],
    [
        State('checklist-channel-selection', 'options'),
        State('checklist-channel-selection', 'value')
    ],
    prevent_initial_call=True
)
def handle_channel_buttons(n_all, n_clear, n_eeg, options, current_value):
    """Maneja los botones de ayuda para selección de canales"""
    if not ctx.triggered:
        return no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-select-all-channels':
        # Seleccionar todos los canales
        all_channels = [opt['value'] for opt in options]
        print(f"[handle_channel_buttons] Seleccionados todos ({len(all_channels)} canales)")
        return all_channels

    elif button_id == 'btn-clear-channels':
        # Limpiar selección
        print("[handle_channel_buttons] Limpiados todos los canales")
        return []

    elif button_id == 'btn-only-eeg-channels':
        # Solo canales EEG (excluir Status y otros)
        eeg_channels = [opt['value'] for opt in options if opt['value'] != 'Status' and not opt['value'].startswith('EXG')]
        print(f"[handle_channel_buttons] Seleccionados solo EEG ({len(eeg_channels)} canales)")
        return eeg_channels

    return no_update


@callback(
    [
        Output(EVENTS_STORE_ID, "data"),
        Output(DATA_STORE_ID, "data"),
    ],
    [
        Input("selected-file-path", "data"),
        Input(SELECTED_CLASS_STORE, "data"),
        Input(SELECTED_CHANNELS_STORE, "data")  # ✨ Nuevo: canales seleccionados
    ],
    State("selected-dataset", "data")
)
def pass_selected_path(selected_file_path, selected_class, selected_channels, dataset_name):
    if selected_file_path is None:
        return no_update, no_update

    if isinstance(selected_file_path, dict):
        candidate = selected_file_path.get("path") or selected_file_path.get("file") or ""
    else:
        candidate = str(selected_file_path)

    candidate = candidate.strip()
    if not candidate:
        return no_update, no_update

    payload = {"path": candidate, "ts": time.time()}

    data_payload = no_update
    try:
        # Usar la nueva función que filtra por clase
        res = Dataset.get_events_by_class(candidate, class_name=selected_class)
        first_evt = res.get("first_event_file") if isinstance(res, dict) else None
        if first_evt:
            # ✨ NUEVO: Cargar con filtro de canales si hay selección
            if selected_channels and len(selected_channels) > 0 and dataset_name:
                print(f"[pass_selected_path] Cargando evento con {len(selected_channels)} canales específicos")
                result = Dataset.load_event_with_channels(first_evt, selected_channels, dataset_name)
                arr = result["data"]
                print(f"[pass_selected_path] Evento filtrado shape: {arr.shape} (canales: {result['channel_names']})")
            else:
                print(f"[pass_selected_path] Cargando evento completo (todos los canales)")
                arr = np.load(first_evt, allow_pickle=False)

            # Extraer nombre del archivo (ej: "abajo[439.357]{441.908}.npy")
            import os
            import re
            file_name = os.path.basename(first_evt)

            # Extraer sesión del path (ej: "sub-02/ses-03")
            session_match = re.search(r'(sub-\d+)/(ses-\d+)', first_evt)
            session_info = f"{session_match.group(1)}/{session_match.group(2)}" if session_match else "Unknown"

            # Calcular duración (necesitamos la frecuencia de muestreo)
            # Intentar obtener metadata para la frecuencia
            try:
                from shared.fileUtils import get_dataset_metadata
                meta = get_dataset_metadata(candidate.split('/')[0])  # "nieto_inner_speech"
                sfreq = meta.get("sampling_frequency_hz", 1024.0)
            except:
                sfreq = 1024.0  # Default

            n_samples = arr.shape[1] if arr.ndim == 2 else arr.shape[0]
            duration_sec = n_samples / sfreq

            # ✨ Obtener nombres de canales para mostrar en plots
            if selected_channels and len(selected_channels) > 0:
                # Usar los canales seleccionados por el usuario
                channel_names_for_plots = selected_channels
            else:
                # Obtener todos los nombres de canales del dataset
                try:
                    all_channel_names = Dataset.get_all_channel_names(dataset_name)
                    channel_names_for_plots = all_channel_names if all_channel_names else [f"Ch{i}" for i in range(arr.shape[0])]
                except:
                    channel_names_for_plots = [f"Ch{i}" for i in range(arr.shape[0])]

            data_payload = {
                "source": first_evt,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "matrix": arr.tolist(),
                "ts": time.time(),
                "class_filter": selected_class,
                "n_events": res.get("n_events", 0),
                "file_name": file_name.replace('.npy', ''),  # Sin extensión
                "session": session_info,
                "duration_sec": round(duration_sec, 3),
                "sfreq": sfreq,
                "selected_channels": selected_channels,  # ✨ Canales seleccionados
                "n_channels_selected": len(selected_channels) if selected_channels else arr.shape[0],
                "channel_names": channel_names_for_plots  # ✨ Nombres de canales para plots
            }

            if selected_channels:
                print(f"[filtros] ✅ Cargado evento de clase '{selected_class}' con {len(selected_channels)} canales: {first_evt}")
                print(f"[filtros] 🔍 Canales: {selected_channels}")
            else:
                print(f"[filtros] ✅ Cargado evento de clase '{selected_class}': {first_evt}")

            print(f"[filtros] 📊 Sesión: {session_info}, Duración: {duration_sec:.3f}s, Muestras: {n_samples}")
    except Exception as e:
        print(f"[filtros] ERROR cargando evento .npy: {e}")

    return payload, data_payload


# CALLBACK: Seleccionar todas las clases (limpiar filtro)
@callback(
    Output(SELECTED_CLASS_STORE, "data", allow_duplicate=True),
    Input("btn-all-classes", "n_clicks"),
    prevent_initial_call=True
)
def select_all_classes(n_clicks):
    if not n_clicks:
        return no_update
    print("[filtros] 📊 Seleccionado: TODAS las clases")
    return None  # None significa "todas las clases"


# CALLBACK: Seleccionar clase específica
@callback(
    Output(SELECTED_CLASS_STORE, "data", allow_duplicate=True),
    Input({'type': 'btn-filter-class', 'index': ALL}, 'n_clicks'),
    State("selected-dataset", "data"),
    prevent_initial_call=True
)
def select_specific_class(n_clicks_list, selected_dataset):
    if not any(n_clicks_list):
        return no_update

    # Obtener metadata para saber qué clase corresponde al botón clickeado
    try:
        meta = get_dataset_metadata(selected_dataset)
        classes = meta.get("classes", [])
    except Exception:
        return no_update

    # Encontrar cuál botón fue clickeado
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == '':
        return no_update

    import json
    button_id = json.loads(triggered_id)
    class_index = button_id.get('index', -1)

    if 0 <= class_index < len(classes):
        selected_class = classes[class_index]
        print(f"[filtros] 🎯 Seleccionado clase: {selected_class}")
        return selected_class

    return no_update


# CALLBACK: Actualizar estilo del botón "Todas"
@callback(
    Output("btn-all-classes", "style"),
    Input(SELECTED_CLASS_STORE, "data")
)
def update_all_button_style(selected_class):
    is_selected = selected_class is None
    return {
        "padding": "3px 6px",
        "flex": "1",
        "borderRadius": "var(--radius-sm)",
        "border": f"1px solid {'var(--accent-1)' if is_selected else 'var(--border-weak)'}",
        "background": "var(--accent-1)" if is_selected else "var(--card-bg)",
        "color": "var(--text)",
        "cursor": "pointer",
        "fontSize": "10px",
        "fontWeight": "600" if is_selected else "500",
        "opacity": "1" if is_selected else "0.7",
        "whiteSpace": "nowrap"
    }


# CALLBACK: Actualizar estilos de botones de clase
@callback(
    Output({'type': 'btn-filter-class', 'index': ALL}, 'style'),
    Input(SELECTED_CLASS_STORE, "data"),
    State("selected-dataset", "data")
)
def update_class_buttons_style(selected_class, selected_dataset):
    try:
        meta = get_dataset_metadata(selected_dataset)
        classes = meta.get("classes", [])
    except Exception:
        classes = []

    styles = []
    for cls in classes:
        is_selected = selected_class == cls
        styles.append({
            "padding": "3px 6px",
            "flex": "1",
            "borderRadius": "var(--radius-sm)",
            "border": f"1px solid {'var(--accent-1)' if is_selected else 'var(--border-weak)'}",
            "background": "var(--accent-1)" if is_selected else "var(--card-bg)",
            "color": "var(--text)",
            "cursor": "pointer",
            "fontSize": "10px",
            "fontWeight": "600" if is_selected else "500",
            "opacity": "1" if is_selected else "0.7",
            "whiteSpace": "nowrap"
        })

    return styles


# CALLBACK: Habilitar/deshabilitar botón "Todas"
@callback(
    Output("btn-all-classes", "disabled"),
    Input(DATA_STORE_ID, "data")
)
def enable_all_button(signal_data):
    # Habilitar solo si hay datos cargados
    return signal_data is None or not isinstance(signal_data, dict)


# CALLBACK: Habilitar/deshabilitar botones de clase
@callback(
    Output({'type': 'btn-filter-class', 'index': ALL}, 'disabled'),
    Input(DATA_STORE_ID, "data"),
    State("selected-dataset", "data")
)
def enable_class_buttons(signal_data, selected_dataset):
    try:
        meta = get_dataset_metadata(selected_dataset)
        classes = meta.get("classes", [])
    except Exception:
        classes = []

    # Habilitar solo si hay datos cargados
    has_data = signal_data is not None and isinstance(signal_data, dict)
    return [not has_data] * len(classes)


# CLIENTSIDE: Renderiza plots con WebGL + limpieza de contextos
clientside_callback(
    """
    function(storeData, selectedPathRaw, signalData, filteredData, channelRange) {
      try {
        // ===== ⚙️ CONFIGURACIÓN PRINCIPAL =====
        
        // Tipo de renderizado
        const USE_WEBGL = false;  
        // true  = WebGL (más rápido, límite ~8 gráficos simultáneos)
        // false = SVG (sin límite, pero más lento con muchos puntos)
        // Rango recomendado: true o false
        
        // Downsampling (reducción de puntos)
        const USE_DOWNSAMPLING = false;  
        // true  = Activa reducción de puntos (RECOMENDADO)
        // false = Muestra todos los puntos (puede ser muy lento)
        // Rango recomendado: true
        
        const DS_FACTOR = 2;  
        // Factor de reducción: toma 1 de cada N puntos
        // Rango recomendado: 1-20
        //   1  = Sin reducción (todos los puntos)
        //   2  = Muestra 1 de cada 2 puntos (50% de datos)
        //   4  = Muestra 1 de cada 4 puntos (25% de datos) ← RECOMENDADO
        //   8  = Muestra 1 de cada 8 puntos (12.5% de datos)
        //   16 = Muestra 1 de cada 16 puntos (6.25% de datos)
        // Nota: A mayor factor, más rápido pero menos detalle
        
        const MAX_POINTS = 15000;  
        // Límite máximo de puntos por canal después del downsampling
        // Rango recomendado: 2000-15000
        //   2000  = Muy rápido, poco detalle
        //   4000  = Rápido, buen balance
        //   8000  = Balance óptimo ← RECOMENDADO
        //   12000 = Más detalle, algo más lento
        //   15000 = Máximo detalle con WebGL
        // Nota: Si DS_FACTOR ya reduce suficiente, este límite puede no aplicarse
        
        const CHANNELS_PER_PAGE = 16;  
        // Número de canales a mostrar por página
        // Rango recomendado: 4-16
        //   4  = Muy pocos, muchas páginas
        //   6  = Seguro para WebGL
        //   8  = Balance óptimo ← RECOMENDADO
        //   12 = Más canales, puede causar problemas con WebGL
        //   16 = Máximo recomendado solo con SVG (USE_WEBGL=false)
        // Nota: Con WebGL=true, no exceder de 8-10 para evitar errores de contexto
        
        // Limpieza de contextos WebGL previos
        if (window.plotlyGraphRefs && USE_WEBGL) {
          window.plotlyGraphRefs.forEach(ref => {
            try {
              if (ref && ref._fullLayout && ref._fullLayout._glcontainer) {
                const gl = ref._fullLayout._glcontainer.querySelector('canvas');
                if (gl) {
                  const context = gl.getContext('webgl') || gl.getContext('experimental-webgl');
                  if (context) {
                    const loseContext = context.getExtension('WEBGL_lose_context');
                    if (loseContext) loseContext.loseContext();
                  }
                }
              }
            } catch(e) { /* silenciar */ }
          });
          window.plotlyGraphRefs = [];
        }

        function downsampling(xArr, yArr, opts) {
          if (!Array.isArray(yArr) || yArr.length === 0) return { x: xArr, y: yArr };
          const factor = Math.max(1, (opts && opts.factor) ? opts.factor : 1);
          const maxPts = Math.max(0, (opts && opts.maxPoints) ? opts.maxPoints : 0);
          let eff = factor;
          if (maxPts > 0 && yArr.length > maxPts) eff = Math.max(eff, Math.ceil(yArr.length / maxPts));
          if (eff <= 1) return { x: xArr, y: yArr };
          const xd = [], yd = [];
          for (let i = 0; i < yArr.length; i += eff) { 
            yd.push(yArr[i]); 
            xd.push(xArr ? xArr[i] : i); 
          }
          return { x: xd, y: yd };
        }

        if (!(signalData && Array.isArray(signalData.matrix) && Array.isArray(signalData.matrix[0]))) {
          return [];
        }

        const total = signalData.matrix.length;
        const cols = signalData.matrix[0].length;
        const xFull = Array.from({length: cols}, (_, i) => i);

        // ✨ Obtener nombres de canales
        const channelNames = signalData.channel_names || [];
        const hasChannelNames = channelNames.length > 0;

        // Obtener rango de canales a mostrar
        const channelStart = (channelRange && channelRange.start) || 0;
        const channelCount = Math.min(CHANNELS_PER_PAGE, total - channelStart);

        const graphsOriginal = [];
        const graphsFiltered = [];

        // Renderizar plots para ambas columnas
        for (let i = 0; i < channelCount; i++) {
          const ch = channelStart + i;
          const yRaw = signalData.matrix[ch];
          if (!Array.isArray(yRaw)) continue;

          // ✨ Obtener nombre del canal (ej: "A1", "A2", "B5")
          const channelLabel = hasChannelNames && ch < channelNames.length
            ? channelNames[ch]
            : 'Ch ' + ch;

          const xy = USE_DOWNSAMPLING
            ? downsampling(xFull, yRaw, { factor: DS_FACTOR, maxPoints: MAX_POINTS })
            : { x: xFull, y: yRaw };

          // Plot original (columna izquierda)
          const figOriginal = {
            data: [{
              type: USE_WEBGL ? 'scattergl' : 'scatter',
              mode: 'lines',
              x: xy.x,
              y: xy.y,
              line: { width: 1, color: '#3b82f6' },
              hoverinfo: 'skip',
              name: channelLabel
            }],
            layout: {
              margin: { l: 50, r: 10, t: 24, b: 24 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              showlegend: false,
              xaxis: { showgrid: false, zeroline: false, fixedrange: true, title: 'muestras' },
              yaxis: {
                showgrid: true,
                gridcolor: 'rgba(128,128,128,0.25)',
                zeroline: false,
                fixedrange: true,
                title: channelLabel,
                titlefont: { size: 14, weight: 'bold' }
              },
              height: 320,
              autosize: true,
              uirevision: 'mp-const-orig-' + ch,
              annotations: [{
                text: channelLabel,
                xref: 'paper',
                yref: 'paper',
                x: 0.02,
                y: 0.98,
                xanchor: 'left',
                yanchor: 'top',
                showarrow: false,
                font: {
                  size: 18,
                  color: '#3b82f6',
                  weight: 'bold'
                },
                bgcolor: 'rgba(0,0,0,0.7)',
                borderpad: 6
              }]
            }
          };

          graphsOriginal.push({
            props: {
              id: `pg-multi-orig-${ch}`,
              figure: figOriginal,
              responsive: true,
              className: 'plot-item',
              style: { height: '320px', width: '100%', minHeight: 0, marginBottom: '12px' },
              config: {
                displaylogo: false,
                responsive: true,
                modeBarButtonsToRemove: [
                  'zoom','pan','select','lasso2d','zoomIn2d','zoomOut2d',
                  'autoScale2d','resetScale2d','toImage'
                ]
              }
            },
            type: 'Graph',
            namespace: 'dash_core_components'
          });

          // Plot filtrado (columna derecha) - usar datos filtrados si existen
          const hasFilteredData = filteredData && Array.isArray(filteredData.matrix) && Array.isArray(filteredData.matrix[ch]);
          const yFiltered = hasFilteredData ? filteredData.matrix[ch] : xy.y.map(() => 0);

          const xyFiltered = USE_DOWNSAMPLING && hasFilteredData
            ? downsampling(xFull, yFiltered, { factor: DS_FACTOR, maxPoints: MAX_POINTS })
            : { x: xFull, y: yFiltered };

          // Crear anotaciones para plot filtrado
          const filteredAnnotations = [
            // Siempre mostrar nombre del canal
            {
              text: channelLabel,
              xref: 'paper',
              yref: 'paper',
              x: 0.02,
              y: 0.98,
              xanchor: 'left',
              yanchor: 'top',
              showarrow: false,
              font: {
                size: 18,
                color: hasFilteredData ? '#a855f7' : '#888',
                weight: 'bold'
              },
              bgcolor: 'rgba(0,0,0,0.7)',
              borderpad: 6
            }
          ];

          // Agregar mensaje "Sin filtro aplicado" si no hay datos filtrados
          if (!hasFilteredData) {
            filteredAnnotations.push({
              text: 'Sin filtro aplicado',
              xref: 'paper',
              yref: 'paper',
              x: 0.5,
              y: 0.5,
              showarrow: false,
              font: { size: 12, color: 'rgba(255,255,255,0.3)' }
            });
          }

          const figFiltered = {
            data: [{
              type: USE_WEBGL ? 'scattergl' : 'scatter',
              mode: 'lines',
              x: xyFiltered.x,
              y: xyFiltered.y,
              line: { width: 1, color: hasFilteredData ? '#a855f7' : '#555' },
              hoverinfo: 'skip',
              name: 'Filtrado ' + channelLabel
            }],
            layout: {
              margin: { l: 50, r: 10, t: 24, b: 24 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              showlegend: false,
              xaxis: { showgrid: false, zeroline: false, fixedrange: true, title: 'muestras' },
              yaxis: {
                showgrid: true,
                gridcolor: 'rgba(128,128,128,0.25)',
                zeroline: false,
                fixedrange: true,
                title: channelLabel,
                titlefont: { size: 14, weight: 'bold' }
              },
              height: 320,
              autosize: true,
              uirevision: 'mp-const-filt-' + ch,
              annotations: filteredAnnotations
            }
          };

          graphsFiltered.push({
            props: {
              id: `pg-multi-filt-${ch}`,
              figure: figFiltered,
              responsive: true,
              className: 'plot-item',
              style: { height: '320px', width: '100%', minHeight: 0, marginBottom: '12px' },
              config: {
                displaylogo: false,
                responsive: true,
                modeBarButtonsToRemove: [
                  'zoom','pan','select','lasso2d','zoomIn2d','zoomOut2d',
                  'autoScale2d','resetScale2d','toImage'
                ]
              }
            },
            type: 'Graph',
            namespace: 'dash_core_components'
          });
        }

        // Guardar referencias para limpieza futura
        if (USE_WEBGL) {
          setTimeout(() => {
            if (!window.plotlyGraphRefs) window.plotlyGraphRefs = [];
            for (let i = 0; i < channelCount; i++) {
              const ch = channelStart + i;
              const elOrig = document.getElementById(`pg-multi-orig-${ch}`);
              const elFilt = document.getElementById(`pg-multi-filt-${ch}`);
              if (elOrig && elOrig._fullData) window.plotlyGraphRefs.push(elOrig);
              if (elFilt && elFilt._fullData) window.plotlyGraphRefs.push(elFilt);
            }
            window.dispatchEvent(new Event('resize'));
          }, 100);
        } else {
          setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 0);
        }

        // Retornar estructura de dos columnas
        return {
          props: {
            children: [
              {
                props: {
                  children: [
                    {
                      props: {
                        children: 'Señal Original',
                        style: {
                          fontSize: '14px',
                          fontWeight: '600',
                          color: 'var(--text)',
                          marginBottom: '12px',
                          paddingBottom: '8px',
                          borderBottom: '2px solid #3b82f6'
                        }
                      },
                      type: 'Div',
                      namespace: 'dash_html_components'
                    },
                    ...graphsOriginal
                  ],
                  style: {
                    flex: 1,
                    paddingRight: '8px',
                    minWidth: 0
                  }
                },
                type: 'Div',
                namespace: 'dash_html_components'
              },
              {
                props: {
                  children: [
                    {
                      props: {
                        children: 'Señal Filtrada',
                        style: {
                          fontSize: '14px',
                          fontWeight: '600',
                          color: 'var(--text)',
                          marginBottom: '12px',
                          paddingBottom: '8px',
                          borderBottom: '2px solid #a855f7'
                        }
                      },
                      type: 'Div',
                      namespace: 'dash_html_components'
                    },
                    ...graphsFiltered
                  ],
                  style: {
                    flex: 1,
                    paddingLeft: '8px',
                    minWidth: 0
                  }
                },
                type: 'Div',
                namespace: 'dash_html_components'
              }
            ],
            style: {
              display: 'flex',
              gap: '16px',
              width: '100%'
            }
          },
          type: 'Div',
          namespace: 'dash_html_components'
        };
      } catch (e) {
        console.error('[clientside:hybrid] ERROR:', e);
        return window.dash_clientside.no_update;
      }
    }
    """,
    Output('plots-container', 'children'),
    [
        Input(EVENTS_STORE_ID, 'data'),
        Input('selected-file-path', 'data'),
        Input(DATA_STORE_ID, 'data'),
        Input(FILTERED_DATA_STORE_ID, 'data'),
        Input(CHANNEL_RANGE_STORE, 'data')
    ],
    prevent_initial_call=True
)


# CALLBACK: Navegación de canales (Anterior)
clientside_callback(
    """
    function(n_clicks, currentRange) {
      if (!n_clicks || n_clicks === 0) return window.dash_clientside.no_update;
      const CHANNELS_PER_PAGE = 8;
      const currentStart = (currentRange && currentRange.start) || 0;
      const newStart = Math.max(0, currentStart - CHANNELS_PER_PAGE);
      return {start: newStart, count: CHANNELS_PER_PAGE};
    }
    """,
    Output(CHANNEL_RANGE_STORE, 'data', allow_duplicate=True),
    Input('btn-prev-channels', 'n_clicks'),
    State(CHANNEL_RANGE_STORE, 'data'),
    prevent_initial_call=True
)


# CALLBACK: Navegación de canales (Siguiente)
clientside_callback(
    """
    function(n_clicks, currentRange, signalData) {
      if (!n_clicks || n_clicks === 0) return window.dash_clientside.no_update;
      if (!(signalData && Array.isArray(signalData.matrix))) return window.dash_clientside.no_update;
      
      const CHANNELS_PER_PAGE = 8;
      const total = signalData.matrix.length;
      const currentStart = (currentRange && currentRange.start) || 0;
      const newStart = Math.min(total - CHANNELS_PER_PAGE, currentStart + CHANNELS_PER_PAGE);
      return {start: newStart, count: CHANNELS_PER_PAGE};
    }
    """,
    Output(CHANNEL_RANGE_STORE, 'data', allow_duplicate=True),
    Input('btn-next-channels', 'n_clicks'),
    [State(CHANNEL_RANGE_STORE, 'data'), State(DATA_STORE_ID, 'data')],
    prevent_initial_call=True
)


# CALLBACK: Actualizar texto de información de canales
clientside_callback(
    """
    function(channelRange, signalData) {
      if (!(signalData && Array.isArray(signalData.matrix))) {
        return "Canales 0 - 0 de 0";
      }

      const CHANNELS_PER_PAGE = 8;
      const total = signalData.matrix.length;
      const start = (channelRange && channelRange.start) || 0;
      const count = Math.min(CHANNELS_PER_PAGE, total - start);
      const end = start + count - 1;

      return `Canales ${start} - ${end} de ${total}`;
    }
    """,
    Output('channel-nav-info', 'children'),
    [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE_ID, 'data')]
)


# CALLBACK: Actualizar estilo botón anterior
clientside_callback(
    """
    function(channelRange, signalData) {
      if (!(signalData && Array.isArray(signalData.matrix))) {
        return {
          padding: '3px 8px',
          borderRadius: 'var(--radius-sm)',
          border: 'none',
          background: 'var(--card-bg)',
          color: 'var(--text)',
          cursor: 'not-allowed',
          fontSize: '10px',
          fontWeight: '500',
          opacity: '0.5',
          flex: '1'
        };
      }

      const start = (channelRange && channelRange.start) || 0;
      const isDisabled = start === 0;

      return {
        padding: '3px 8px',
        borderRadius: 'var(--radius-sm)',
        border: 'none',
        background: isDisabled ? 'var(--card-bg)' : 'var(--accent-1)',
        color: 'var(--text)',
        cursor: isDisabled ? 'not-allowed' : 'pointer',
        fontSize: '10px',
        fontWeight: '500',
        opacity: isDisabled ? '0.5' : '1',
        flex: '1'
      };
    }
    """,
    Output('btn-prev-channels', 'style'),
    [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE_ID, 'data')]
)


# CALLBACK: Actualizar estilo botón siguiente
clientside_callback(
    """
    function(channelRange, signalData) {
      if (!(signalData && Array.isArray(signalData.matrix))) {
        return {
          padding: '3px 8px',
          borderRadius: 'var(--radius-sm)',
          border: 'none',
          background: 'var(--card-bg)',
          color: 'var(--text)',
          cursor: 'not-allowed',
          fontSize: '10px',
          fontWeight: '500',
          opacity: '0.5',
          flex: '1'
        };
      }

      const CHANNELS_PER_PAGE = 8;
      const total = signalData.matrix.length;
      const start = (channelRange && channelRange.start) || 0;
      const isDisabled = start + CHANNELS_PER_PAGE >= total;

      return {
        padding: '3px 8px',
        borderRadius: 'var(--radius-sm)',
        border: 'none',
        background: isDisabled ? 'var(--card-bg)' : 'var(--accent-1)',
        color: 'var(--text)',
        cursor: isDisabled ? 'not-allowed' : 'pointer',
        fontSize: '10px',
        fontWeight: '500',
        opacity: isDisabled ? '0.5' : '1',
        flex: '1'
      };
    }
    """,
    Output('btn-next-channels', 'style'),
    [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE_ID, 'data')]
)


# CALLBACK: Actualizar disabled botón anterior
clientside_callback(
    """
    function(channelRange, signalData) {
      if (!(signalData && Array.isArray(signalData.matrix))) return true;
      const start = (channelRange && channelRange.start) || 0;
      return start === 0;
    }
    """,
    Output('btn-prev-channels', 'disabled'),
    [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE_ID, 'data')]
)

# CALLBACK: Actualizar disabled botón siguiente
clientside_callback(
    """
    function(channelRange, signalData) {
      if (!(signalData && Array.isArray(signalData.matrix))) return true;
      const CHANNELS_PER_PAGE = 8;
      const total = signalData.matrix.length;
      const start = (channelRange && channelRange.start) || 0;
      return start + CHANNELS_PER_PAGE >= total;
    }
    """,
    Output('btn-next-channels', 'disabled'),
    [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE_ID, 'data')]
)