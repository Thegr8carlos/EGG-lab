# app/pages/modelado_p300.py
import time, random, math
from dash import html, dcc, register_page, callback, Output, Input, State, no_update
from shared.fileUtils import get_dataset_metadata

register_page(__name__, path="/p300", name="Modelado P300")

# ---------- IDs ----------
TRAIN_CONFIG_STORE_ID  = "train-config-p300"
TRAIN_STATUS_STORE_ID  = "train-status-p300"
TRAIN_METRICS_STORE_ID = "train-metrics-p300"
TRAIN_INTERVAL_ID      = "train-interval-p300"

BTN_TRAIN_ID     = "btn-train-p300"
DD_CLASSES_ID    = "dd-classes-p300"
DD_CHANNELS_ID   = "dd-channels-p300"
IN_SPLIT_ID      = "in-split-p300"
IN_KFOLDS_ID     = "in-kfolds-p300"

STATUS_VIEW_ID   = "train-status-view-p300"
METRICS_VIEW_ID  = "train-metrics-view-p300"
DATASET_LABEL_ID = "ds-name-p300"

# ---------- UI helpers ----------
def _badge(text, kind="info"):
    return html.Span(text, className=f"badge {kind}")

def _progress(percent: float):
    pct = max(0, min(100, float(percent)))
    return html.Div([html.Div(className="progress__fill", style={"width": f"{pct:.0f}%"} )], className="progress")

def _training_card():
    return html.Div(
        [
            html.H2("Entrenamiento del modelo P300"),
            html.Div(id=DATASET_LABEL_ID, className="train-meta"),

            html.Label("Clases a incluir", className="train-label"),
            dcc.Dropdown(id=DD_CLASSES_ID, options=[], value=[], multi=True,
                         placeholder="Selecciona clases…", className="train-input"),
            html.Div(style={"height":".5rem"}),

            html.Label("Canales a incluir", className="train-label"),
            dcc.Dropdown(id=DD_CHANNELS_ID, options=[], value=[], multi=True,
                         placeholder="Selecciona canales…", className="train-input"),
            html.Div(style={"height":".5rem"}),

            html.Label("Test split (%)", className="train-label"),
            dcc.Input(id=IN_SPLIT_ID, type="number", min=5, max=50, step=1, value=20, className="train-input"),
            html.Div(style={"height":".5rem"}),

            html.Label("K-Folds", className="train-label"),
            dcc.Input(id=IN_KFOLDS_ID, type="number", min=2, max=10, step=1, value=5, className="train-input"),
            html.Div(style={"height":".75rem"}),

            html.Button("Entrenar", id=BTN_TRAIN_ID, n_clicks=0, className="btn-primary"),

            html.Hr(className="hr-soft"),
            html.Div(id=STATUS_VIEW_ID, style={"marginBottom":".5rem"}),
            dcc.Loading(html.Div(id=METRICS_VIEW_ID)),

            # Stores + Interval (mock)
            dcc.Store(id=TRAIN_CONFIG_STORE_ID),
            dcc.Store(id=TRAIN_STATUS_STORE_ID),
            dcc.Store(id=TRAIN_METRICS_STORE_ID),
            dcc.Interval(id=TRAIN_INTERVAL_ID, interval=1000, disabled=True),
        ],
        className="train-card"
    )

layout = html.Div([_training_card()], className="page-wrap")

# ---------- Callbacks ----------

@callback(
    [Output(DD_CLASSES_ID, "options"),
     Output(DD_CHANNELS_ID, "options"),
     Output(DATASET_LABEL_ID, "children")],
    Input("selected-dataset", "data")
)
def fill_meta_options(selected_dataset):
    if not selected_dataset:
        return [], [], "Dataset: —"
    try:
        meta = get_dataset_metadata(selected_dataset)
        classes = [{"label": str(c), "value": str(c)} for c in (meta.get("classes") or [])]
        chans   = [{"label": ch, "value": ch} for ch in (meta.get("channel_names") or meta.get("channel_name_union") or [])]
        return classes, chans, f"Dataset: {meta.get('dataset_name') or selected_dataset}"
    except Exception:
        return [], [], f"Dataset: {selected_dataset} (sin metadata)"

@callback(
    [Output(TRAIN_CONFIG_STORE_ID, "data"),
     Output(TRAIN_STATUS_STORE_ID, "data"),
     Output(TRAIN_INTERVAL_ID, "disabled")],
    Input(BTN_TRAIN_ID, "n_clicks"),
    [State(DD_CLASSES_ID, "value"),
     State(DD_CHANNELS_ID, "value"),
     State(IN_SPLIT_ID, "value"),
     State(IN_KFOLDS_ID, "value")],
    prevent_initial_call=True
)
def start_training(n, classes, channels, split, kfolds):
    cfg = {
        "classes": classes or [],
        "channels": channels or [],
        "test_split": float(split or 20.0),
        "kfolds": int(kfolds or 5),
        "ts": time.time(),
        "n_steps": 8
    }
    status = {"status": "running", "step": 0, "t0": time.time()}
    return cfg, status, False  # habilita Interval

@callback(
    [Output(TRAIN_STATUS_STORE_ID, "data", allow_duplicate=True),
     Output(TRAIN_METRICS_STORE_ID, "data"),
     Output(TRAIN_INTERVAL_ID, "disabled", allow_duplicate=True)],
    Input(TRAIN_INTERVAL_ID, "n_intervals"),
    [State(TRAIN_STATUS_STORE_ID, "data"),
     State(TRAIN_CONFIG_STORE_ID, "data")],
    prevent_initial_call=True
)
def tick_training(n, status, cfg):
    if not status or status.get("status") != "running":
        return no_update, no_update, True

    step = int(status.get("step", 0)) + 1
    n_steps = int((cfg or {}).get("n_steps", 8))

    if step < n_steps:
        return {"status": "running", "step": step, "t0": status.get("t0")}, no_update, False

    # terminar
    if random.random() < 0.8:
        acc = round(random.uniform(0.75, 0.94), 3)
        f1  = round(random.uniform(0.70, 0.92), 3)
        classes = (cfg or {}).get("classes") or ["target","non-target"]
        per_class = [{"class": str(c),
                      "precision": round(random.uniform(0.7,0.95),3),
                      "recall":    round(random.uniform(0.7,0.95),3),
                      "f1":        round(random.uniform(0.7,0.95),3)} for c in classes]
        metrics = {"summary": {"accuracy": acc, "f1_macro": f1}, "per_class": per_class}
        return {"status": "finished", "step": step, "t0": status.get("t0")}, metrics, True
    else:
        return {"status": "error", "step": step, "t0": status.get("t0"),
                "message":"Fallo de entrenamiento simulado"}, no_update, True

@callback(Output(STATUS_VIEW_ID, "children"),
          Input(TRAIN_STATUS_STORE_ID, "data"),
          State(TRAIN_CONFIG_STORE_ID, "data"))
def render_status(status, cfg):
    if not status:
        return html.Div("Esperando a iniciar entrenamiento…", style={"opacity":0.7})

    s = status.get("status")
    step = int(status.get("step", 0))
    n_steps = int((cfg or {}).get("n_steps", 8))
    pct = (step / max(1, n_steps)) * 100.0

    if s == "running":
        return html.Div([
            _badge("Entrenando… va bien", "info"),
            _progress(pct)
        ])
    if s == "finished":
        return html.Div([_badge("Entrenamiento terminado", "ok")])
    if s == "error":
        return html.Div([
            _badge("Error", "error"),
            html.Div(status.get("message") or "Ocurrió un error.", style={"marginTop":".25rem"})
        ])
    return html.Div("—")

@callback(Output(METRICS_VIEW_ID, "children"),
          Input(TRAIN_METRICS_STORE_ID, "data"))
def render_metrics(m):
    if not m:
        return html.Div("Esperando resultados…", style={"opacity":0.7, "minHeight":"1rem"})
    summary = m.get("summary", {})
    per_class = m.get("per_class", [])
    table = html.Table(
        [
            html.Thead(html.Tr([html.Th("Clase"), html.Th("Precisión"), html.Th("Recall"), html.Th("F1")])),
            html.Tbody([
                html.Tr([html.Td(r["class"]), html.Td(r["precision"]), html.Td(r["recall"]), html.Td(r["f1"])])
                for r in per_class
            ])
        ],
        style={"width":"100%","borderCollapse":"collapse"}
    )
    return html.Div([
        html.Div(f"Accuracy: {summary.get('accuracy','—')}  |  F1-macro: {summary.get('f1_macro','—')}",
                 style={"fontWeight":"700","margin":"0 0 .5rem 0"}),
        table
    ])
