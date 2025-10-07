# app/pages/p300.py
from pathlib import Path
import time
import numpy as np

from dash import html, dcc, register_page, callback
from dash import Output, Input, State, clientside_callback, no_update

from shared.fileUtils import get_dataset_metadata
from app.components.PlayGroundP300 import get_playGroundP300
from app.components.RigthComlumn import get_rightColumn
from app.components.SideBar import get_sideBar  # si quieres barra lateral

from backend.classes.dataset import Dataset

# Registrar ruta
register_page(__name__, path="/p300", name="Modelado p300")

# IDs únicos para ESTA PÁGINA
GRAPH_ID        = "pg-main-plot-p300"
EVENTS_STORE_ID = "events-store-p300"
DATA_STORE_ID   = "signal-store-p300"

# ----- Layout -----
rigthColumn = get_rightColumn("clasificationModelsP300")

layout = html.Div(
    [
        # (Opcional) Sidebar: descomenta si la quieres aquí también
        # html.Div(
        #     id="sidebar-wrapper",
        #     children=[get_sideBar("Data")],
        #     className="sideBar-container",
        #     style={"width": "260px", "padding": "1rem"},
        # ),

        html.Div(
            id="pg-wrapper-p300",
            children=get_playGroundP300(
                "Modelado P300",
                "Aquí puedes visualizar la señal P300 y elegir el modelo a implementar.",
                {}, {}, graph_id=GRAPH_ID, multi=True
            ),
            style={"flex": "1", "padding": "1rem"},
        ),

        html.Div([rigthColumn], style={"width": "340px", "padding": "1rem"}),

        # Stores propios de la página
        dcc.Store(id=EVENTS_STORE_ID),
        dcc.Store(id=DATA_STORE_ID),
    ],
    style={"display": "flex"},
)

# =========================
# Helpers de normalización
# =========================
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
        or ((meta.get("unique_sfreqs") or [None])[0]
            if isinstance(meta.get("unique_sfreqs"), (list, tuple)) else None)
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

# ==========================================================
# 1) Re-render PlayGround con metadata (server side)
# ==========================================================
@callback(
    Output("pg-wrapper-p300", "children"),
    Input("selected-dataset", "data")
)
def update_playground_desc_p300(selected_dataset):
    desc = selected_dataset or "Selecciona un dataset en 'Cargar Datos'"
    if not selected_dataset:
        print("[p300] selected-dataset vacío -> render básico")
        return get_playGroundP300("Modelado P300", desc, {}, {}, graph_id=GRAPH_ID, multi=True)
    try:
        meta = get_dataset_metadata(selected_dataset)
        print(f"[p300] get_dataset_metadata OK: {selected_dataset}")
    except Exception as e:
        print(f"[p300] get_dataset_metadata ERROR: {e}")
        return get_playGroundP300("Modelado P300", f"{desc} (sin metadata: {e})", {}, {}, graph_id=GRAPH_ID, multi=True)

    meta_dict, custom_dict = create_metadata_section(meta)
    return get_playGroundP300("Modelado P300", desc, meta_dict, custom_dict, graph_id=GRAPH_ID, multi=True)

# ==========================================================
# 2) Backend: devolver (a) path+ts y (b) matriz del primer evento .npy
# ==========================================================
@callback(
    [Output(EVENTS_STORE_ID, "data"),
     Output(DATA_STORE_ID,   "data")],
    Input("selected-file-path", "data"),
    prevent_initial_call=True
)
def pass_selected_path_p300(selected_file_path):
    print(f"[p300] pass_selected_path: {selected_file_path!r}")

    if selected_file_path is None:
        return no_update, no_update

    # Normaliza (acepta str o dict con 'path'/'file')
    if isinstance(selected_file_path, dict):
        candidate = selected_file_path.get("path") or selected_file_path.get("file") or ""
    else:
        candidate = str(selected_file_path)
    candidate = candidate.strip()
    if not candidate:
        return no_update, no_update

    payload = {"path": candidate, "ts": time.time()}

    # Localizar Events y cargar primer .npy completo
    data_payload = no_update
    try:
        res = Dataset.load_events(candidate)  # mismo helper que usas en filtros
        print(f"[p300] load_events -> {res}")

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
            print(f"[p300] DATA_STORE listo. shape={arr.shape}, dtype={arr.dtype}")
        else:
            print("[p300] No se encontró first_event_file; DATA_STORE omitido.")
    except Exception as e:
        print(f"[p300] ERROR cargando primer evento .npy: {e}")

    return payload, data_payload

# ==========================================================
# 3) Frontend (clientside): plot único stacked (tu script)
# ==========================================================
clientside_callback(
    """
    function(storeData, selectedPathRaw, signalData, currentFigure) {
      try {
        // ====== ⚙️ CONFIG ======
        const CHANNELS_START = 0;     // 0..total-1
        const CHANNELS_COUNT = 32;    // 1..total
        const PER_ROW_PX     = 220;   // 80..220
        const GAP            = 0.10;  // 0..0.10

        const USE_DOWNSAMPLING = true;
        const DS_FACTOR        = 4;   // 1 = sin DS
        const MAX_POINTS       = 6500;// 0 = sin límite

        function downsampling(xArr, yArr, opts) {
          if (!Array.isArray(yArr) || yArr.length === 0) return { x: xArr, y: yArr };
          const factor = Math.max(1, (opts && opts.factor) ? opts.factor : 1);
          const maxPts = Math.max(0, (opts && opts.maxPoints) ? opts.maxPoints : 0);
          let eff = factor;
          if (maxPts > 0 && yArr.length > maxPts) eff = Math.max(eff, Math.ceil(yArr.length / maxPts));
          if (eff <= 1) return { x: xArr, y: yArr };
          const xd = [], yd = [];
          for (let i = 0; i < yArr.length; i += eff) { yd.push(yArr[i]); xd.push(xArr ? xArr[i] : i); }
          return { x: xd, y: yd };
        }

        const pathFromStore = (storeData && typeof storeData.path !== "undefined") ? ("" + storeData.path) : "";
        const pathFromInput = (selectedPathRaw && typeof selectedPathRaw === "object")
                              ? (selectedPathRaw.path || selectedPathRaw.file || "")
                              : ("" + (selectedPathRaw || ""));
        const selectedPath  = (pathFromStore || pathFromInput).trim();

        let fig = currentFigure || {};
        fig = JSON.parse(JSON.stringify(fig || {}));
        fig.data   = [];
        fig.layout = fig.layout || {};
        if (typeof fig.config !== "undefined") delete fig.config;

        if (signalData && Array.isArray(signalData.matrix) && Array.isArray(signalData.matrix[0])) {
          const total = signalData.matrix.length;
          const cols  = signalData.matrix[0].length;
          const xFull = Array.from({length: cols}, (_, i) => i);

          const start = Math.max(0, Math.min(CHANNELS_START, total - 1));
          const filas = Math.min(Math.max(1, CHANNELS_COUNT), total - start);

          // limpia ejes previos
          for (const k in fig.layout) {
            if (/^yaxis\\d*$/.test(k) || /^xaxis\\d*$/.test(k)) delete fig.layout[k];
          }

          // eje X compartido
          fig.layout.xaxis = { title: "muestras", showgrid: false, zeroline: false, fixedrange: true };

          // dominios verticales (de ARRIBA hacia ABAJO)
          const gap  = Math.min(Math.max(GAP, 0), 0.10);
          const slot = (1 - gap * (filas - 1)) / filas;

          for (let localIdx = 0; localIdx < filas; localIdx++) {
            const ch   = start + localIdx;
            const yRaw = signalData.matrix[ch];
            if (!Array.isArray(yRaw)) continue;

            const xy = USE_DOWNSAMPLING
              ? downsampling(xFull, yRaw, { factor: DS_FACTOR, maxPoints: MAX_POINTS })
              : { x: xFull, y: yRaw };

            const bottom = 1 - (localIdx + 1) * slot - localIdx * gap;
            const top    = bottom + slot;

            const yaxisName = (localIdx === 0) ? "yaxis" : ("yaxis" + (localIdx + 1));
            const yref      = (localIdx === 0) ? "y"     : ("y"     + (localIdx + 1));

            fig.layout[yaxisName] = {
              domain: [bottom, top],
              title: "Ch " + ch,
              titlefont: { size: 10 },
              tickfont:  { size: 9 },
              showgrid:  true,
              gridcolor: "rgba(128,128,128,0.25)",
              zeroline:  false,
              fixedrange:true,
              anchor:    "x"
            };

            fig.data.push({
              type: "scattergl",
              mode: "lines",
              x: xy.x,
              y: xy.y,
              yaxis: yref,
              name: "Ch " + ch,
              line: { width: 1 },
              hoverinfo: "skip"
            });
          }

          fig.layout.height        = Math.max(400, PER_ROW_PX * filas);
          fig.layout.margin        = { l: 60, r: 10, t: 28, b: 30 };
          fig.layout.paper_bgcolor = "rgba(0,0,0,0)";
          fig.layout.plot_bgcolor  = "rgba(0,0,0,0)";
          fig.layout.showlegend    = false;

          if (selectedPath) {
            fig.layout.title = {
              text: `Eventos — ${selectedPath}  |  Canales: ${start}…${start + filas - 1} (total ${total})`,
              x: 0, xanchor: "left", font: { size: 14 }
            };
          }
        } else {
          if (selectedPath) fig.layout.title = { text: "Eventos — " + selectedPath, x: 0, xanchor: "left", font: { size: 14 } };
        }

        return fig;
      } catch (e) {
        console.error("[clientside:p300] ERROR:", e);
        return window.dash_clientside.no_update;
      }
    }
    """,
    Output(GRAPH_ID, "figure"),
    [Input(EVENTS_STORE_ID, "data"),
     Input("selected-file-path", "data"),
     Input(DATA_STORE_ID, "data")],
    State(GRAPH_ID, "figure"),
    prevent_initial_call=True
)