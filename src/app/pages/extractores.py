# extractores.py - Transformadas con vista de dos columnas
from pathlib import Path
import time
import numpy as np
from dash import html, dcc, register_page, callback, Output, Input, State, clientside_callback, no_update, ALL, ctx
from shared.fileUtils import get_dataset_metadata
import dash_bootstrap_components as dbc

from app.components.PageContainer import get_page_container
from app.components.PlayGround import get_playGround
from app.components.RigthComlumn import get_rightColumn
from app.components.SideBar import get_sideBar

from backend.classes.dataset import Dataset

register_page(__name__, path="/extractores", name="Transformadas")

GRAPH_ID = "pg-main-plot-extractores"
EVENTS_STORE_ID = "events-store-extractores"
DATA_STORE_ID = "signal-store-extractores"
TRANSFORMED_DATA_STORE_ID = "transformed-signal-store-extractores"
CHANNEL_RANGE_STORE = "channel-range-store-extractores"
SELECTED_CLASS_STORE = "selected-class-store-extractores"
SELECTED_CHANNELS_STORE = "selected-channels-store-extractores"

layout = html.Div(
    [
        html.Div(
            id="sidebar-wrapper",
            children=[get_sideBar("Data")],
            className="sideBar-container",
            style={"width": "260px", "padding": "1rem"},
        ),
        html.Div(
            id="pg-wrapper-extractores",
            children=get_playGround("Extractores", "Description", {}, {}, graph_id=GRAPH_ID, multi=True, plots_container_id="plots-container-extractores"),
            style={"flex": "1", "padding": "1rem"},
        ),
        html.Div(
            get_rightColumn("featureExtracture"),
            style={"width": "340px", "padding": "1rem"},
        ),
        dcc.Store(id=EVENTS_STORE_ID),
        dcc.Store(id=DATA_STORE_ID),
        dcc.Store(id=TRANSFORMED_DATA_STORE_ID),
        dcc.Store(id=CHANNEL_RANGE_STORE, data={"start": 0, "count": 8}),
        dcc.Store(id=SELECTED_CLASS_STORE, data=None),
        dcc.Store(id=SELECTED_CHANNELS_STORE, data=None),
    ],
    style={"display": "flex"},
)

def create_metadata_section(meta: dict):
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
        # Navegación de canales (sin título)
        html.Div([
            html.Div(
                id='channel-nav-info-extractores',
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
                    id='btn-prev-channels-extractores',
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
                    id='btn-next-channels-extractores',
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
                    id='btn-all-classes-extractores',
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
                    id={'type': 'btn-filter-class-extractores', 'index': idx},
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
                    html.Button("Todos", id="btn-select-all-channels-extractores", n_clicks=0, style={
                        "padding": "2px 6px",
                        "fontSize": "8px",
                        "borderRadius": "3px",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer",
                        "marginRight": "4px"
                    }),
                    html.Button("Limpiar", id="btn-clear-channels-extractores", n_clicks=0, style={
                        "padding": "2px 6px",
                        "fontSize": "8px",
                        "borderRadius": "3px",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text)",
                        "cursor": "pointer",
                        "marginRight": "4px"
                    }),
                    html.Button("Solo EEG", id="btn-only-eeg-channels-extractores", n_clicks=0, style={
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
                    id='checklist-channel-selection-extractores',
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
            html.Div(id="channel-count-display-extractores", children="0 canales seleccionados", style={
                "fontSize": "8px",
                "color": "var(--text-muted)",
                "marginTop": "4px",
                "textAlign": "right"
            })
        ])
    ])


@callback(
    Output("pg-wrapper-extractores", "children"),
    Input("selected-dataset", "data")
)
def update_playground_desc(selected_dataset):
    desc = selected_dataset or "Selecciona un dataset en 'Cargar Datos'"
    if not selected_dataset:
        return get_playGround("Extractores", desc, {}, {}, graph_id=GRAPH_ID, multi=True, plots_container_id="plots-container-extractores")
    try:
        meta = get_dataset_metadata(selected_dataset)
    except Exception as e:
        return get_playGround("Extractores", f"{desc} (sin metadata: {e})", {}, {}, graph_id=GRAPH_ID, multi=True, plots_container_id="plots-container-extractores")

    meta_dict, custom_dict = create_metadata_section(meta)
    nav_controls = create_navigation_controls(meta)
    return get_playGround("Extractores", desc, meta_dict, custom_dict, graph_id=GRAPH_ID, multi=True, navigation_controls=nav_controls, plots_container_id="plots-container-extractores")


@callback(
    [
        Output(EVENTS_STORE_ID, "data"),
        Output(DATA_STORE_ID, "data"),
    ],
    [
        Input("selected-file-path", "data"),
        Input(SELECTED_CLASS_STORE, "data"),
        Input(SELECTED_CHANNELS_STORE, "data")
    ],
    State("selected-dataset", "data"),
    prevent_initial_call=True
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
                print(f"[extractores] Cargando evento con {len(selected_channels)} canales específicos")
                result = Dataset.load_event_with_channels(first_evt, selected_channels, dataset_name)
                arr = result["data"]
                print(f"[extractores] Evento filtrado shape: {arr.shape} (canales: {result['channel_names']})")
            else:
                print(f"[extractores] Cargando evento completo (todos los canales)")
                arr = np.load(first_evt, allow_pickle=False)

            # Extraer nombre del archivo (ej: "abajo[439.357]{441.908}.npy")
            import os
            import re
            file_name = os.path.basename(first_evt)

            # Extraer clase del nombre del archivo (antes del corchete)
            class_match = file_name.split('[')[0] if '[' in file_name else file_name.replace('.npy', '')
            event_class = class_match.strip()

            session_match = re.search(r'(sub-\d+)/(ses-\d+)', first_evt)
            session_info = f"{session_match.group(1)}/{session_match.group(2)}" if session_match else "Unknown"

            # Calcular duración (necesitamos la frecuencia de muestreo)
            # Intentar obtener metadata para la frecuencia y colores de clases
            try:
                from shared.fileUtils import get_dataset_metadata
                meta = get_dataset_metadata(candidate.split('/')[0])  # "nieto_inner_speech"
                sfreq = meta.get("sampling_frequency_hz", 1024.0)
                # Obtener mapa de colores de clases
                class_color_map, _ = create_metadata_section(meta)
            except:
                sfreq = 1024.0  # Default
                class_color_map = {}

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
                "event_class": event_class,  # ✨ Clase extraída del nombre del archivo
                "class_name": event_class,  # ✨ Alias para compatibilidad
                "session": session_info,
                "duration_sec": round(duration_sec, 3),
                "sfreq": sfreq,
                "selected_channels": selected_channels,  # ✨ Canales seleccionados
                "n_channels_selected": len(selected_channels) if selected_channels else arr.shape[0],
                "channel_names": channel_names_for_plots,  # ✨ Nombres de canales para plots
                "class_colors": class_color_map  # ✨ Mapa de colores por clase
            }

            if selected_channels:
                print(f"[extractores] ✅ Cargado evento de clase '{selected_class}' con {len(selected_channels)} canales: {first_evt}")
                print(f"[extractores] 🔍 Canales: {selected_channels}")
            else:
                print(f"[extractores] ✅ Cargado evento de clase '{selected_class}': {first_evt}")

            print(f"[extractores] 📊 Sesión: {session_info}, Duración: {duration_sec:.3f}s, Muestras: {n_samples}")
    except Exception as e:
        print(f"[extractores] ERROR cargando primer evento .npy: {e}")

    return payload, data_payload


# CLIENTSIDE: Renderiza plots con dos columnas (Original + Transformada)
clientside_callback(
    """
    function(storeData, selectedPathRaw, signalData, transformedData, channelRange, selectedClass, selectedChannels) {
      try {
        // ===== ⚙️ CONFIGURACIÓN PRINCIPAL =====

        const USE_WEBGL = false;
        const USE_DOWNSAMPLING = false;
        const DS_FACTOR = 2;
        const MAX_POINTS = 15000;
        const CHANNELS_PER_PAGE = 16;

        // Limpieza de contextos WebGL previos
        if (window.plotlyGraphRefsExtractores && USE_WEBGL) {
          window.plotlyGraphRefsExtractores.forEach(ref => {
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
          window.plotlyGraphRefsExtractores = [];
        }

        // Función para oscurecer un color HSL reduciendo la luminosidad
        function darkenHSL(hslColor, amount = 20) {
          // Parsear HSL: "hsl(210, 75%, 55%)" -> [210, 75, 55]
          const match = hslColor.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
          if (!match) return hslColor; // Si no es HSL válido, devolver original

          const h = parseInt(match[1]);
          const s = parseInt(match[2]);
          const l = parseInt(match[3]);

          // Reducir luminosidad, asegurando que no baje de 0
          const newL = Math.max(0, l - amount);

          return `hsl(${h}, ${s}%, ${newL}%)`;
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

        // ===== FILTRADO POR CANALES ESPECÍFICOS =====
        // NOTA: El filtrado por clase ya se hace en el servidor con Dataset.get_events_by_class(),
        // no necesitamos verificar aquí
        let channelIndices = [];
        if (selectedChannels && Array.isArray(selectedChannels) && selectedChannels.length > 0) {
          // Usuario seleccionó canales específicos
          channelIndices = selectedChannels.map(chName => {
            const idx = channelNames.indexOf(chName);
            return idx >= 0 ? idx : -1;
          }).filter(idx => idx >= 0);

          if (channelIndices.length === 0) {
            // No hay canales válidos seleccionados
            return [];
          }

          console.log(`[extractores] Mostrando solo ${channelIndices.length} canales seleccionados`);
        } else {
          // Mostrar todos los canales (con paginación)
          const channelStart = (channelRange && channelRange.start) || 0;
          const channelCount = Math.min(CHANNELS_PER_PAGE, total - channelStart);
          channelIndices = Array.from({length: channelCount}, (_, i) => channelStart + i);
        }

        const graphsOriginal = [];
        const graphsTransformed = [];

        // Obtener información de archivo y clase para colores consistentes
        const fileName = signalData.file_name || 'Sin archivo';
        const sessionInfo = signalData.session || '';

        // Extraer solo la clase del nombre del archivo (antes del corchete)
        const classNameMatch = fileName.match(/^([^\[]+)/);
        const className = classNameMatch ? classNameMatch[1] : fileName;

        // Obtener color de la clase desde el mapa centralizado
        const classColors = signalData.class_colors || {};
        const classColor = classColors[className] || '#3b82f6';  // Fallback: azul

        // Renderizar plots para ambas columnas
        for (let i = 0; i < channelIndices.length; i++) {
          const ch = channelIndices[i];
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
              line: { width: 1, color: classColor },
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
              uirevision: 'mp-const-orig-ext-' + ch,
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
                  color: classColor,
                  weight: 'bold'
                },
                bgcolor: 'rgba(0,0,0,0.7)',
                borderpad: 6
              }]
            }
          };

          graphsOriginal.push({
            props: {
              id: `pg-multi-orig-ext-${ch}`,
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

          // Plot transformada (columna derecha) - usar datos transformados si existen
          const hasTransformedData = transformedData && Array.isArray(transformedData.matrix) && Array.isArray(transformedData.matrix[ch]);
          const yTransformed = hasTransformedData ? transformedData.matrix[ch] : xy.y.map(() => 0);

          const xyTransformed = USE_DOWNSAMPLING && hasTransformedData
            ? downsampling(xFull, yTransformed, { factor: DS_FACTOR, maxPoints: MAX_POINTS })
            : { x: xFull, y: yTransformed };

          const figTransformed = {
            data: [{
              type: USE_WEBGL ? 'scattergl' : 'scatter',
              mode: 'lines',
              x: xyTransformed.x,
              y: xyTransformed.y,
              line: { width: 1, color: hasTransformedData ? darkenHSL(classColor, 20) : '#555' },
              hoverinfo: 'skip',
              name: 'Transform ' + channelLabel
            }],
            layout: {
              margin: { l: 50, r: 10, t: 24, b: 24 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              showlegend: false,
              xaxis: { showgrid: false, zeroline: false, fixedrange: true, title: 'frecuencia/nivel' },
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
              uirevision: 'mp-const-trans-ext-' + ch,
              annotations: hasTransformedData ? [
                // Nombre del canal en esquina superior (con datos transformados)
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
                    color: darkenHSL(classColor, 20),
                    weight: 'bold'
                  },
                  bgcolor: 'rgba(0,0,0,0.7)',
                  borderpad: 6
                }
              ] : [
                // Nombre del canal en esquina superior (sin datos)
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
                    color: '#888',
                    weight: 'bold'
                  },
                  bgcolor: 'rgba(0,0,0,0.7)',
                  borderpad: 6
                },
                // Mensaje "Sin transformada aplicada" en el centro
                {
                  text: 'Sin transformada aplicada',
                  xref: 'paper',
                  yref: 'paper',
                  x: 0.5,
                  y: 0.5,
                  showarrow: false,
                  font: { size: 12, color: 'rgba(255,255,255,0.3)' }
                }
              ]
            }
          };

          graphsTransformed.push({
            props: {
              id: `pg-multi-trans-ext-${ch}`,
              figure: figTransformed,
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
            if (!window.plotlyGraphRefsExtractores) window.plotlyGraphRefsExtractores = [];
            for (let i = 0; i < channelIndices.length; i++) {
              const ch = channelIndices[i];
              const elOrig = document.getElementById(`pg-multi-orig-ext-${ch}`);
              const elTrans = document.getElementById(`pg-multi-trans-ext-${ch}`);
              if (elOrig && elOrig._fullData) window.plotlyGraphRefsExtractores.push(elOrig);
              if (elTrans && elTrans._fullData) window.plotlyGraphRefsExtractores.push(elTrans);
            }
            window.dispatchEvent(new Event('resize'));
          }, 100);
        } else {
          setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 0);
        }

        // Crear títulos dinámicos con información del archivo
        // Función para crear título estilizado con elementos HTML
        function createStyledTitle(session, eventClass, type, color) {
          const parts = [];

          // Sesión
          if (session) {
            parts.push({
              props: {
                children: 'Sesión: ',
                style: {
                  fontSize: '11px',
                  fontWeight: '600',
                  color: 'rgba(255,255,255,0.5)',
                  marginRight: '4px'
                }
              },
              type: 'Span',
              namespace: 'dash_html_components'
            });
            parts.push({
              props: {
                children: session,
                style: {
                  fontSize: '11px',
                  fontWeight: '500',
                  color: 'rgba(255,255,255,0.7)',
                  marginRight: '12px'
                }
              },
              type: 'Span',
              namespace: 'dash_html_components'
            });
          }

          // Clase
          parts.push({
            props: {
              children: 'Clase: ',
              style: {
                fontSize: '11px',
                fontWeight: '600',
                color: 'rgba(255,255,255,0.5)',
                marginRight: '4px'
              }
            },
            type: 'Span',
            namespace: 'dash_html_components'
          });
          parts.push({
            props: {
              children: eventClass,
              style: {
                fontSize: '13px',
                fontWeight: '700',
                color: color,
                marginRight: '12px'
              }
            },
            type: 'Span',
            namespace: 'dash_html_components'
          });

          // Tipo
          parts.push({
            props: {
              children: 'Tipo: ',
              style: {
                fontSize: '11px',
                fontWeight: '600',
                color: 'rgba(255,255,255,0.5)',
                marginRight: '4px'
              }
            },
            type: 'Span',
            namespace: 'dash_html_components'
          });
          parts.push({
            props: {
              children: type,
              style: {
                fontSize: '11px',
                fontWeight: '600',
                color: 'rgba(255,255,255,0.8)',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }
            },
            type: 'Span',
            namespace: 'dash_html_components'
          });

          return parts;
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
                        children: createStyledTitle(sessionInfo, className, 'Original', classColor),
                        style: {
                          marginBottom: '12px',
                          paddingBottom: '8px',
                          borderBottom: '2px solid ' + classColor,
                          display: 'flex',
                          alignItems: 'center',
                          overflow: 'hidden'
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
                        children: createStyledTitle(sessionInfo, className, 'Transformada', classColor),
                        style: {
                          marginBottom: '12px',
                          paddingBottom: '8px',
                          borderBottom: '2px solid ' + classColor,
                          display: 'flex',
                          alignItems: 'center',
                          overflow: 'hidden'
                        }
                      },
                      type: 'Div',
                      namespace: 'dash_html_components'
                    },
                    ...graphsTransformed
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

        return result;

      } catch (e) {
        console.error('[clientside:extractores] ERROR:', e);
        return window.dash_clientside.no_update;
      }
    }
    """,
    Output('plots-container-extractores', 'children'),
    [
        Input(EVENTS_STORE_ID, 'data'),
        Input('selected-file-path', 'data'),
        Input(DATA_STORE_ID, 'data'),
        Input(TRANSFORMED_DATA_STORE_ID, 'data'),
        Input(CHANNEL_RANGE_STORE, 'data'),
        Input(SELECTED_CLASS_STORE, 'data'),
        Input(SELECTED_CHANNELS_STORE, 'data')
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
    Input('btn-prev-channels-extractores', 'n_clicks'),
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
    Input('btn-next-channels-extractores', 'n_clicks'),
    [State(CHANNEL_RANGE_STORE, 'data'), State(DATA_STORE_ID, 'data')],
    prevent_initial_call=True
)


# CALLBACK: Actualizar texto de información de canales
clientside_callback(
    """
    function(channelRange, signalData, selectedChannels) {
      if (!(signalData && Array.isArray(signalData.matrix))) {
        return "Canales 0 - 0 de 0";
      }

      // Si hay canales específicos seleccionados, mostrar eso
      if (selectedChannels && Array.isArray(selectedChannels) && selectedChannels.length > 0) {
        return `${selectedChannels.length} canales seleccionados`;
      }

      // Si no, mostrar rango paginado
      const CHANNELS_PER_PAGE = 8;
      const total = signalData.matrix.length;
      const start = (channelRange && channelRange.start) || 0;
      const count = Math.min(CHANNELS_PER_PAGE, total - start);
      const end = start + count - 1;

      return `Canales ${start} - ${end} de ${total}`;
    }
    """,
    Output('channel-nav-info-extractores', 'children'),
    [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE_ID, 'data'), Input(SELECTED_CHANNELS_STORE, 'data')]
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
    Output('btn-prev-channels-extractores', 'style'),
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
    Output('btn-next-channels-extractores', 'style'),
    [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE_ID, 'data')]
)


# CALLBACK: Actualizar disabled botón anterior
clientside_callback(
    """
    function(channelRange, signalData, selectedChannels) {
      if (!(signalData && Array.isArray(signalData.matrix))) return true;

      // Si hay canales específicos seleccionados, deshabilitar navegación
      if (selectedChannels && Array.isArray(selectedChannels) && selectedChannels.length > 0) {
        return true;
      }

      const start = (channelRange && channelRange.start) || 0;
      return start === 0;
    }
    """,
    Output('btn-prev-channels-extractores', 'disabled'),
    [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE_ID, 'data'), Input(SELECTED_CHANNELS_STORE, 'data')]
)


# CALLBACK: Actualizar disabled botón siguiente
clientside_callback(
    """
    function(channelRange, signalData, selectedChannels) {
      if (!(signalData && Array.isArray(signalData.matrix))) return true;

      // Si hay canales específicos seleccionados, deshabilitar navegación
      if (selectedChannels && Array.isArray(selectedChannels) && selectedChannels.length > 0) {
        return true;
      }

      const CHANNELS_PER_PAGE = 8;
      const total = signalData.matrix.length;
      const start = (channelRange && channelRange.start) || 0;
      return start + CHANNELS_PER_PAGE >= total;
    }
    """,
    Output('btn-next-channels-extractores', 'disabled'),
    [Input(CHANNEL_RANGE_STORE, 'data'), Input(DATA_STORE_ID, 'data'), Input(SELECTED_CHANNELS_STORE, 'data')]
)


# ===== CALLBACKS PARA SELECTOR DE CANALES =====

@callback(
    Output('checklist-channel-selection-extractores', 'options'),
    Input('selected-dataset', 'data')
)
def populate_channel_checklist_extractores(selected_dataset):
    """Llena el checklist de canales cuando se selecciona un dataset"""
    if not selected_dataset:
        return []

    try:
        # Obtener nombres de canales usando la nueva función
        channel_names = Dataset.get_all_channel_names(selected_dataset)

        if not channel_names:
            print(f"[populate_channel_checklist_extractores] No se encontraron canales para {selected_dataset}")
            return []

        # Crear opciones para el checklist
        options = [{"label": ch, "value": ch} for ch in channel_names]

        print(f"[populate_channel_checklist_extractores] Cargados {len(options)} canales para {selected_dataset}")
        return options

    except Exception as e:
        print(f"[populate_channel_checklist_extractores] ERROR: {e}")
        return []


@callback(
    Output(SELECTED_CHANNELS_STORE, 'data'),
    Input('checklist-channel-selection-extractores', 'value')
)
def save_selected_channels_extractores(selected_channels):
    """Guarda los canales seleccionados en el store"""
    if not selected_channels or len(selected_channels) == 0:
        print("[save_selected_channels_extractores] Ningún canal seleccionado, mostrando todos")
        return None  # None = mostrar todos los canales

    print(f"[save_selected_channels_extractores] Canales seleccionados: {selected_channels}")
    return selected_channels


@callback(
    Output('channel-count-display-extractores', 'children'),
    Input('checklist-channel-selection-extractores', 'value')
)
def update_channel_count_extractores(selected_channels):
    """Actualiza el contador de canales seleccionados"""
    count = len(selected_channels) if selected_channels else 0
    if count == 0:
        return "Todos los canales"
    elif count == 1:
        return "1 canal seleccionado"
    else:
        return f"{count} canales seleccionados"


@callback(
    Output('checklist-channel-selection-extractores', 'value'),
    [
        Input('btn-select-all-channels-extractores', 'n_clicks'),
        Input('btn-clear-channels-extractores', 'n_clicks'),
        Input('btn-only-eeg-channels-extractores', 'n_clicks')
    ],
    [
        State('checklist-channel-selection-extractores', 'options'),
        State('checklist-channel-selection-extractores', 'value')
    ],
    prevent_initial_call=True
)
def handle_channel_buttons_extractores(n_all, n_clear, n_eeg, options, current_value):
    """Maneja los botones de ayuda para selección de canales"""
    if not ctx.triggered:
        return no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-select-all-channels-extractores':
        # Seleccionar todos los canales
        all_channels = [opt['value'] for opt in options]
        print(f"[handle_channel_buttons_extractores] Seleccionados todos ({len(all_channels)} canales)")
        return all_channels

    elif button_id == 'btn-clear-channels-extractores':
        # Limpiar selección
        print("[handle_channel_buttons_extractores] Limpiados todos los canales")
        return []

    elif button_id == 'btn-only-eeg-channels-extractores':
        # Solo canales EEG (excluir Status y otros)
        eeg_channels = [opt['value'] for opt in options if opt['value'] != 'Status' and not opt['value'].startswith('EXG')]
        print(f"[handle_channel_buttons_extractores] Seleccionados solo EEG ({len(eeg_channels)} canales)")
        return eeg_channels

    return no_update


# ===== CALLBACKS PARA FILTRO POR CLASE =====

# CALLBACK: Seleccionar todas las clases (limpiar filtro)
@callback(
    Output(SELECTED_CLASS_STORE, "data", allow_duplicate=True),
    Input("btn-all-classes-extractores", "n_clicks"),
    prevent_initial_call=True
)
def select_all_classes_extractores(n_clicks):
    if not n_clicks:
        return no_update
    print("[extractores] 📊 Seleccionado: TODAS las clases")
    return None  # None significa "todas las clases"


# CALLBACK: Seleccionar clase específica
@callback(
    Output(SELECTED_CLASS_STORE, "data", allow_duplicate=True),
    Input({'type': 'btn-filter-class-extractores', 'index': ALL}, 'n_clicks'),
    State("selected-dataset", "data"),
    prevent_initial_call=True
)
def select_specific_class_extractores(n_clicks_list, selected_dataset):
    if not any(n_clicks_list):
        return no_update

    # Obtener metadata para saber qué clase corresponde al botón clickeado
    try:
        meta = get_dataset_metadata(selected_dataset)
        classes = meta.get("classes", [])
    except Exception:
        return no_update

    # Encontrar cuál botón fue clickeado
    import dash
    ctx_obj = dash.callback_context
    if not ctx_obj.triggered:
        return no_update

    triggered_id = ctx_obj.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == '':
        return no_update

    import json
    button_id = json.loads(triggered_id)
    class_index = button_id.get('index', -1)

    if 0 <= class_index < len(classes):
        selected_class = classes[class_index]
        print(f"[extractores] 🎯 Seleccionado clase: {selected_class}")
        return selected_class

    return no_update


# CALLBACK: Actualizar estilo del botón "Todas"
@callback(
    Output("btn-all-classes-extractores", "style"),
    Input(SELECTED_CLASS_STORE, "data")
)
def update_all_button_style_extractores(selected_class):
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
    Output({'type': 'btn-filter-class-extractores', 'index': ALL}, 'style'),
    Input(SELECTED_CLASS_STORE, "data"),
    State("selected-dataset", "data")
)
def update_class_buttons_style_extractores(selected_class, selected_dataset):
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
    Output("btn-all-classes-extractores", "disabled"),
    Input(DATA_STORE_ID, "data")
)
def enable_all_button_extractores(signal_data):
    # Habilitar solo si hay datos cargados
    return signal_data is None or not isinstance(signal_data, dict)


# CALLBACK: Habilitar/deshabilitar botones de clase
@callback(
    Output({'type': 'btn-filter-class-extractores', 'index': ALL}, 'disabled'),
    Input(DATA_STORE_ID, "data"),
    State("selected-dataset", "data")
)
def enable_class_buttons_extractores(signal_data, selected_dataset):
    try:
        meta = get_dataset_metadata(selected_dataset)
        classes = meta.get("classes", [])
    except Exception:
        classes = []

    # Habilitar solo si hay datos cargados
    has_data = signal_data is not None and isinstance(signal_data, dict)
    return [not has_data] * len(classes)

