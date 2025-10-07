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
    return get_playGround("Filtros", desc, meta_dict, custom_dict, graph_id=GRAPH_ID, multi=True)


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

        const graphs = [];
        
        // Controles de navegación
        const navControls = {
          props: {
            id: 'channel-nav-controls',
            children: [
              {
                props: {
                  children: `Canales ${channelStart} - ${channelStart + channelCount - 1} de ${total}`,
                  style: {
                    fontSize: '14px',
                    fontWeight: '600',
                    color: 'rgba(255,255,255,0.9)',
                    marginBottom: '12px',
                    textAlign: 'center'
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
                        id: 'btn-prev-channels',
                        children: '← Anteriores',
                        n_clicks: 0,
                        disabled: channelStart === 0,
                        style: {
                          padding: '8px 16px',
                          marginRight: '8px',
                          borderRadius: '6px',
                          border: 'none',
                          background: channelStart === 0 ? 'rgba(128,128,128,0.2)' : 'rgba(59,130,246,0.8)',
                          color: 'white',
                          cursor: channelStart === 0 ? 'not-allowed' : 'pointer',
                          fontSize: '13px',
                          fontWeight: '500'
                        }
                      },
                      type: 'Button',
                      namespace: 'dash_html_components'
                    },
                    {
                      props: {
                        id: 'btn-next-channels',
                        children: 'Siguientes →',
                        n_clicks: 0,
                        disabled: channelStart + channelCount >= total,
                        style: {
                          padding: '8px 16px',
                          borderRadius: '6px',
                          border: 'none',
                          background: channelStart + channelCount >= total ? 'rgba(128,128,128,0.2)' : 'rgba(59,130,246,0.8)',
                          color: 'white',
                          cursor: channelStart + channelCount >= total ? 'not-allowed' : 'pointer',
                          fontSize: '13px',
                          fontWeight: '500'
                        }
                      },
                      type: 'Button',
                      namespace: 'dash_html_components'
                    }
                  ],
                  style: {
                    display: 'flex',
                    justifyContent: 'center',
                    gap: '8px',
                    marginBottom: '16px'
                  }
                },
                type: 'Div',
                namespace: 'dash_html_components'
              }
            ],
            style: {
              padding: '12px',
              background: 'rgba(0,0,0,0.2)',
              borderRadius: '8px',
              marginBottom: '16px',
              border: '1px solid rgba(255,255,255,0.1)'
            }
          },
          type: 'Div',
          namespace: 'dash_html_components'
        };
        
        graphs.push(navControls);
        
        // Renderizar plots
        for (let i = 0; i < channelCount; i++) {
          const ch = channelStart + i;
          const yRaw = signalData.matrix[ch];
          if (!Array.isArray(yRaw)) continue;

          const xy = USE_DOWNSAMPLING
            ? downsampling(xFull, yRaw, { factor: DS_FACTOR, maxPoints: MAX_POINTS })
            : { x: xFull, y: yRaw };

          const fig = {
            data: [{
              type: USE_WEBGL ? 'scattergl' : 'scatter',
              mode: 'lines', 
              x: xy.x, 
              y: xy.y,
              line: { width: 1 }, 
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
              uirevision: 'mp-const-' + ch
            }
          };

          graphs.push({
            props: {
              id: `pg-multi-${ch}`,
              figure: fig,
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
              const el = document.getElementById(`pg-multi-${channelStart + i}`);
              if (el && el._fullData) window.plotlyGraphRefs.push(el);
            }
            window.dispatchEvent(new Event('resize'));
          }, 100);
        } else {
          setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 0);
        }

        return graphs;
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