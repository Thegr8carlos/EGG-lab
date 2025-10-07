# filtros.py - SOLUCIÓN HÍBRIDA: WebGL con controles de navegación
from pathlib import Path
import time
import numpy as np
from dash import html, dcc, register_page, callback, Output, Input, State, clientside_callback, no_update
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
                    disabled=True,
                    style={
                        "padding": "3px 6px",
                        "flex": "1",
                        "borderRadius": "var(--radius-sm)",
                        "border": "1px solid var(--accent-1)",
                        "background": "var(--accent-1)",
                        "color": "var(--text)",
                        "cursor": "not-allowed",
                        "fontSize": "10px",
                        "fontWeight": "500",
                        "opacity": "0.8",
                        "whiteSpace": "nowrap"
                    }
                ),
            ] + [
                html.Button(
                    str(cls),
                    id={'type': 'btn-filter-class', 'index': idx},
                    n_clicks=0,
                    disabled=True,
                    style={
                        "padding": "3px 6px",
                        "flex": "1",
                        "borderRadius": "var(--radius-sm)",
                        "border": "1px solid var(--border-weak)",
                        "background": "var(--card-bg)",
                        "color": "var(--text-muted)",
                        "cursor": "not-allowed",
                        "fontSize": "10px",
                        "fontWeight": "500",
                        "opacity": "0.5",
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

        # Selector de canales específicos (dummy)
        html.Div([
            html.Div("Canales específicos", style={
                "fontSize": "10px",
                "fontWeight": "500",
                "color": "var(--text-muted)",
                "marginBottom": "4px",
                "opacity": "0.7"
            }),
            dcc.Input(
                id='input-channel-selection',
                type='text',
                placeholder='ej: 0,5,10-15',
                disabled=True,
                style={
                    "width": "100%",
                    "padding": "4px 8px",
                    "borderRadius": "var(--radius-sm)",
                    "border": "1px solid var(--border-weak)",
                    "background": "var(--card-bg)",
                    "color": "var(--text-muted)",
                    "fontSize": "10px",
                    "opacity": "0.5"
                }
            )
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
    [
        Output(EVENTS_STORE_ID, "data"),
        Output(DATA_STORE_ID, "data"),
    ],
    Input("selected-file-path", "data"),
    prevent_initial_call=True
)
def pass_selected_path(selected_file_path):
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
        res = Dataset.load_events(candidate)
        first_evt = res.get("first_event_file") if isinstance(res, dict) else None
        if first_evt:
            arr = np.load(first_evt, allow_pickle=False)
            data_payload = {
                "source": first_evt,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "matrix": arr.tolist(),
                "ts": time.time(),
            }
    except Exception as e:
        print(f"[filtros] ERROR cargando primer evento .npy: {e}")

    return payload, data_payload


# CLIENTSIDE: Renderiza plots con WebGL + limpieza de contextos
clientside_callback(
    """
    function(storeData, selectedPathRaw, signalData, channelRange) {
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
              name: 'Ch ' + ch
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
                title: 'Ch ' + ch
              },
              height: 320,
              autosize: true,
              uirevision: 'mp-const-orig-' + ch
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

          // Plot filtrado (columna derecha) - por ahora con datos vacíos/placeholder
          const figFiltered = {
            data: [{
              type: USE_WEBGL ? 'scattergl' : 'scatter',
              mode: 'lines',
              x: xy.x,
              y: xy.y.map(() => 0), // Placeholder: valores en cero
              line: { width: 1, color: '#a855f7' },
              hoverinfo: 'skip',
              name: 'Filtrado Ch ' + ch
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
                title: 'Filt Ch ' + ch
              },
              height: 320,
              autosize: true,
              uirevision: 'mp-const-filt-' + ch,
              annotations: [{
                text: 'Sin filtro aplicado',
                xref: 'paper',
                yref: 'paper',
                x: 0.5,
                y: 0.5,
                showarrow: false,
                font: { size: 12, color: 'rgba(255,255,255,0.3)' }
              }]
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