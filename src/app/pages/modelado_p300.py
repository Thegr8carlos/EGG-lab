# app/pages/modelado_p300.py
import time, random, math, json
from dash import html, dcc, register_page, callback, Output, Input, State, no_update, ALL
from shared.fileUtils import get_dataset_metadata

# ← NUEVO: traemos los esquemas desde el backend al arrancar
from backend.classes.ClasificationModel.ClassifierSchemaFactory import ClassifierSchemaFactory


# ← NUEVO: cards de configuración interactiva
from app.components.model_config_cards import (
    create_config_card,
    create_model_selector_card
)

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

# ← NUEVO: layout y navegación de modelos
SCHEMAS_STORE_ID       = "classifier-schemas-store"
SELECTED_MODEL_STORE_ID= "selected-model-store"
SIDEBAR_ID             = "models-sidebar"
MAIN_VIEW_ID           = "model-config-container"

# ---------- Carga de esquemas al inicio ----------
# Se calcula una vez al importar el módulo
CLASSIFIER_SCHEMAS = ClassifierSchemaFactory.get_all_classifier_schemas() or {}
MODEL_NAMES = list(CLASSIFIER_SCHEMAS.keys())

# ---------- UI helpers ----------
def _badge(text, kind="info"):
    return html.Span(text, className=f"badge {kind}")

def _progress(percent: float):
    pct = max(0, min(100, float(percent)))
    return html.Div([html.Div(className="progress__fill", style={"width": f"{pct:.0f}%"} )], className="progress")

# ← Barra lateral derecha con cards de modelos
def _models_sidebar(models: list[str]) -> html.Div:
    """Crea la barra lateral derecha con las cards de modelos."""
    model_cards = [
        create_model_selector_card(model, idx)
        for idx, model in enumerate(models)
    ]

    return html.Div([
        html.H2("Modelos de clasificación", className="right-panel-title"),
        html.Div(model_cards, className="right-panel-container")
    ], id=SIDEBAR_ID, className="models-sidebar", style={
        "height": "100%",
        "overflowY": "auto",
        "overflowX": "hidden"
    })



# ← Layout maestro con 2 columnas: configuración (izq/centro) | modelos (der)
layout = html.Div([
    # Stores para esquemas y estado
    dcc.Store(id=SCHEMAS_STORE_ID, data=CLASSIFIER_SCHEMAS),
    dcc.Store(id=SELECTED_MODEL_STORE_ID),
    dcc.Store(id=TRAIN_CONFIG_STORE_ID),
    dcc.Store(id=TRAIN_STATUS_STORE_ID),
    dcc.Store(id=TRAIN_METRICS_STORE_ID),
    dcc.Interval(id=TRAIN_INTERVAL_ID, interval=1000, disabled=True),

    # Contenedor principal con 2 columnas
    html.Div([
        # Columna izquierda: Configuración de modelo (scrolleable) - 85%
        html.Div([
            html.Div(id=MAIN_VIEW_ID, children=[
                html.Div([
                    html.H3("Bienvenido", className="text-center mb-3", style={"color": "white"}),
                    html.P(
                        "Selecciona un modelo de la barra derecha para configurarlo de manera interactiva.",
                        className="text-center",
                        style={"color": "rgba(255,255,255,0.7)"}
                    ),
                    html.Div([
                        html.I(className="fas fa-brain fa-5x", style={"color": "#4A90E2"})
                    ], className="text-center mt-5")
                ], className="welcome-message", style={
                    "padding": "60px 20px",
                    "borderRadius": "8px",
                    "minHeight": "400px"
                })
            ], style={
                "height": "100%",
                "overflowY": "auto",
                "overflowX": "hidden"
            })
        ], className="center-column", style={
            "width": "85%",
            "padding": "20px 20px 20px 20px",
            "maxHeight": "calc(100vh - 140px)"
        }),

        # Columna derecha: Lista de modelos - pegada a la derecha
        html.Div(_models_sidebar(MODEL_NAMES), className="right-column", style={
            "width": "15%",
            "minWidth": "280px",
            "maxWidth": "350px",
            "position": "fixed",
            "right": "0",
            "top": "100px",
            "height": "calc(100vh - 120px)",
            "paddingRight": "20px"
        })
    ], style={
        "display": "flex",
        "position": "relative",
        "width": "100%",
        "height": "calc(100vh - 100px)"
    })
], className="page-wrap")

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

    # terminar (mock)
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

# =========================
# Navegación de modelos
# =========================

# 1) Al hacer click en cualquier botón de "Configurar" de un modelo
@callback(
    Output(SELECTED_MODEL_STORE_ID, "data"),
    Input({"type": "model-selector-btn", "index": ALL, "model": ALL}, "n_clicks"),
    State(SELECTED_MODEL_STORE_ID, "data"),
    prevent_initial_call=True
)
def select_model(n_clicks_list, current_sel):
    """Guarda el modelo seleccionado cuando se hace click en 'Configurar'."""
    if not n_clicks_list or not any(n_clicks_list):
        return no_update

    from dash import callback_context as ctx
    if not ctx.triggered:
        return no_update

    # Obtener el botón que disparó el callback
    trig = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        btn_id = json.loads(trig)
        model_name = btn_id.get("model")
        return {"name": model_name}
    except Exception as e:
        print(f"Error al seleccionar modelo: {e}")
        return no_update


# 2) Renderiza la card de configuración según el modelo seleccionado
@callback(
    Output(MAIN_VIEW_ID, "children"),
    [Input(SELECTED_MODEL_STORE_ID, "data"),
     Input(SCHEMAS_STORE_ID, "data")]
)
def render_config_card(selected, schemas):
    """Renderiza la card de configuración del modelo seleccionado."""
    # Si no hay selección, muestra mensaje de bienvenida
    if not selected or not selected.get("name"):
        return html.Div([
            html.H3("Bienvenido", className="text-center mb-3", style={"color": "white"}),
            html.P(
                "Selecciona un modelo de la barra derecha para configurarlo de manera interactiva.",
                className="text-center",
                style={"color": "rgba(255, 255, 255, 0.7)"}
            ),
            html.Div([
                html.I(className="fas fa-brain fa-5x", style={"color": "#4A90E2"})
            ], className="text-center mt-5")
        ], className="welcome-message", style={
            "padding": "60px 20px",
            "borderRadius": "8px",
            "minHeight": "400px"
        })

    model_name = selected["name"]
    schema = (schemas or {}).get(model_name, {})

    if not schema:
        return html.Div([
            html.H4(f"Error: No se encontró esquema para {model_name}", style={"color": "#ff6b6b"})
        ])

    # Crear card de configuración interactiva
    return create_config_card(model_name, schema)


# 3) Botón "Volver" dentro de la card de configuración
@callback(
    Output(SELECTED_MODEL_STORE_ID, "data", allow_duplicate=True),
    Input("config-back-btn", "n_clicks"),
    prevent_initial_call=True
)
def back_to_welcome(_):
    """Limpia la selección para volver a la vista de bienvenida."""
    return None
