# filtros.py
from pathlib import Path
import time
import numpy as np
from dash import html, dcc, register_page, callback, Output, Input, State, clientside_callback, no_update
from shared.fileUtils import get_dataset_metadata

from app.components.PageContainer import get_page_container
from app.components.PlayGround import get_playGround
from app.components.RigthComlumn import get_rightColumn
from app.components.SideBar import get_sideBar

from backend.classes.dataset import Dataset

# Registrar página
register_page(__name__, path="/filtros", name="Filtros")

# IDs únicos para ESTA PÁGINA (evitar colisiones con otras páginas)
GRAPH_ID = "pg-main-plot-filtros"
EVENTS_STORE_ID = "events-store-filtros"   # path + ts
DATA_STORE_ID   = "signal-store-filtros"   # matriz (primer .npy de Events)

# Layout con barra lateral
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
            children=get_playGround("Filtros", "Description", {}, {}, graph_id=GRAPH_ID),
            style={"flex": "1", "padding": "1rem"},
        ),
        html.Div(
            get_rightColumn("filter"),
            style={"width": "340px", "padding": "1rem"},
        ),
        # Stores
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


# ==========================================================
# 1) Re-renderiza el PlayGround con metadata (server side)
# ==========================================================
@callback(
    Output("pg-wrapper-filtros", "children"),
    Input("selected-dataset", "data")
)
def update_playground_desc(selected_dataset):
    desc = selected_dataset or "Selecciona un dataset en 'Cargar Datos'"
    if not selected_dataset:
        print("[filtros] update_playground_desc: selected-dataset vacío -> render básico")
        return get_playGround("Filtros", desc, {}, {}, graph_id=GRAPH_ID)
    try:
        meta = get_dataset_metadata(selected_dataset)
        print(f"[filtros] get_dataset_metadata OK para: {selected_dataset}")
    except Exception as e:
        print(f"[filtros] get_dataset_metadata ERROR para {selected_dataset}: {e}")
        return get_playGround("Filtros", f"{desc} (sin metadata: {e})", {}, {}, graph_id=GRAPH_ID)

    meta_dict, custom_dict = create_metadata_section(meta)
    return get_playGround("Filtros", desc, meta_dict, custom_dict, graph_id=GRAPH_ID)


# ==========================================================
# 2) Backend: devolver (a) path+ts y (b) matriz del primer evento .npy
# ==========================================================
@callback(
    [
        Output(EVENTS_STORE_ID, "data"),
        Output(DATA_STORE_ID, "data"),
    ],
    Input("selected-file-path", "data"),
    prevent_initial_call=True
)
def pass_selected_path(selected_file_path):
    """
    - Normaliza y devuelve el 'path' seleccionado (con ts) en EVENTS_STORE_ID.
    - Ubica la carpeta Events junto al .npy mapeado en Aux y carga el PRIMER .npy completo.
      Ese array se envía serializado (list) en DATA_STORE_ID.
    """
    print(f"[filtros] pass_selected_path: raw value={selected_file_path!r} (type={type(selected_file_path).__name__})")

    if selected_file_path is None:
        print("[filtros] pass_selected_path: None -> no_update")
        return no_update, no_update

    # Normaliza (acepta str o dict con 'path'/'file')
    if isinstance(selected_file_path, dict):
        candidate = selected_file_path.get("path") or selected_file_path.get("file") or ""
    else:
        candidate = str(selected_file_path)

    candidate = candidate.strip()
    if not candidate:
        print("[filtros] pass_selected_path: vacío tras normalizar -> no_update")
        return no_update, no_update

    payload = {"path": candidate, "ts": time.time()}
    print(f"[filtros] pass_selected_path: payload -> {payload}")

    # Localizar Events y cargar el primer .npy completo
    data_payload = no_update
    try:
        res = Dataset.load_events(candidate)  # helper que ubica Aux/Events
        print(f"[filtros] load_events result: {res}")

        first_evt = res.get("first_event_file") if isinstance(res, dict) else None
        if first_evt:
            arr = np.load(first_evt, allow_pickle=False)
            data_payload = {
                "source": first_evt,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "matrix": arr.tolist(),  # serialize para JSON
                "ts": time.time(),
            }
            print(f"[filtros] DATA_STORE payload listo. shape={arr.shape}, dtype={arr.dtype}")
        else:
            print("[filtros] No se encontró first_event_file; DATA_STORE no se enviará.")
    except Exception as e:
        print(f"[filtros] ERROR cargando primer evento .npy: {e}")

    return payload, data_payload


# ==========================================================
# 3) Frontend (clientside): 
#    - Confirma recepción (logs)
#    - PLOTEA con WebGL la PRIMERA FILA vs todas sus columnas
# ==========================================================
clientside_callback(
    """
    function(storeData, selectedPathRaw, signalData, currentFigure) {
        try {
            console.log("[clientside] storeData =", storeData,
                        " selectedPathRaw =", selectedPathRaw,
                        " signalData =", signalData,
                        " currentFigure =", currentFigure);

            // 1) Lee path desde Store y/o input directo
            var pathFromStore = (storeData && typeof storeData.path !== "undefined")
                ? ("" + storeData.path) : "";
            var pathFromInput = (typeof selectedPathRaw !== "undefined" && selectedPathRaw !== null)
                ? (typeof selectedPathRaw === "object"
                    ? (selectedPathRaw.path || selectedPathRaw.file || "")
                    : ("" + selectedPathRaw))
                : "";
            var selectedPath = (pathFromStore || pathFromInput);
            selectedPath = (typeof selectedPath === "string") ? selectedPath.trim() : ("" + selectedPath);

            // 2) Base figure (usar figure actual como punto de partida)
            var fig = currentFigure || {};
            fig = JSON.parse(JSON.stringify(fig || {})); // deep copy

            // Estructura mínima
            fig.data = Array.isArray(fig.data) ? fig.data : [];
            fig.layout = (typeof fig.layout === "object" && fig.layout !== null) ? fig.layout : {};
            if (typeof fig.config !== "undefined") delete fig.config; // evitar warnings

            // 3) Si llegó la matriz -> plotea PRIMERA FILA con Scattergl (línea)
            if (signalData && signalData.matrix && Array.isArray(signalData.matrix)) {
                // shape esperada: [n_canales, n_time]
                var y = signalData.matrix[0]; // primera fila
                if (Array.isArray(y)) {
                    var x = Array.from({length: y.length}, (_, i) => i);

                    fig.data = [{
                        type: "scattergl",
                        mode: "lines",
                        x: x,
                        y: y,
                        name: "Canal 0 (primer evento)",
                        line: { width: 1 },
                        hoverinfo: "skip"
                    }];

                    // logs de confirmación
                    console.log("[clientside] ✅ matriz recibida y ploteada",
                                { source: signalData.source,
                                  shape: signalData.shape,
                                  dtype: signalData.dtype,
                                  filas: signalData.matrix.length,
                                  columnas: signalData.matrix[0].length
                                });
                } else {
                    console.warn("[clientside] matrix[0] no es array, no se puede plotear.");
                }
            } else {
                // si aún no hay matriz, solo actualiza el título
                console.warn("[clientside] (sin matriz aún o payload vacío)");
            }

            // 4) Título (si hay selectedPath)
            if (selectedPath) {
                fig.layout.title = {
                    text: "Plot (WebGL) — " + selectedPath,
                    x: 0, xanchor: "left",
                    font: { size: 14 }
                };
            }

            // 5) Ajustes de layout mínimos (por si no estaban)
            fig.layout.margin = fig.layout.margin || {l:10, r:10, t:10, b:10};
            fig.layout.paper_bgcolor = fig.layout.paper_bgcolor || "rgba(0,0,0,0)";
            fig.layout.plot_bgcolor  = fig.layout.plot_bgcolor  || "rgba(0,0,0,0)";
            fig.layout.showlegend = false;
            fig.layout.xaxis = Object.assign({title:"muestras", showgrid:false, zeroline:false, fixedrange:true}, fig.layout.xaxis||{});
            fig.layout.yaxis = Object.assign({title:"amplitud", showgrid:true, gridcolor:"rgba(128,128,128,0.25)", zeroline:false, fixedrange:true}, fig.layout.yaxis||{});

            return fig;
        } catch (err) {
            console.error("[clientside] ERROR en callback:", err);
            return window.dash_clientside.no_update;
        }
    }
    """,
    Output(GRAPH_ID, "figure"),
    [
        Input(EVENTS_STORE_ID, "data"),        # path+ts
        Input("selected-file-path", "data"),   # respaldo directo
        Input(DATA_STORE_ID, "data"),          # dispara cuando llega la matriz
    ],
    State(GRAPH_ID, "figure"),
    prevent_initial_call=True
)
