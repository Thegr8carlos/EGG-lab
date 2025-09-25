from dash import html, dcc, register_page, callback, Input, Output, ALL, ctx
from dash.exceptions import PreventUpdate
from app.components.PageContainer import get_page_container
from shared.fileUtils import get_data_folders
from backend.classes.dataset import Dataset

register_page(__name__, path="/cargardatos", name="Cargar Datos")

BUTTON_STYLE = {
    "display": "inline-block",
    "padding": "0.5rem 1rem",
    "margin": "0.5rem",
    "backgroundColor": "var(--accent-3)",
    "color": "var(--text)",
    "textDecoration": "none",
    "border": "none",
    "borderRadius": "8px",
    "cursor": "pointer",
    "fontWeight": "bold",
    "boxShadow": "0 4px 6px color-mix(in srgb, var(--shadow-base) 35%, transparent)",
}

layout = get_page_container(
    "Cargando datos hehe",
    "Presiona el botón para mostrar las opciones",
    html.Div([
        # ❌ NO declares aquí otro Store con el mismo id.
        dcc.Location(id="redirector", refresh=True),

        html.Button(
            "Cargar dataset",
            id="cargar-btn",
            n_clicks=0,
            style={
                "padding": "0.5rem 1rem",
                "backgroundColor": "var(--surface-2)",
                "color": "var(--text)",
                "border": "none",
                "borderRadius": "4px",
                "cursor": "pointer"
            }
        ),

        html.Div(id="lista-opciones", style={"marginTop": "1rem", "textAlign": "center"})
    ], style={"textAlign": "center"})
)

@callback(
    Output("lista-opciones", "children"),
    Input("cargar-btn", "n_clicks")
)
def mostrar_opciones(n):
    if n == 0:
        return ""
    datasets = get_data_folders()
    return html.Div([
        html.Button(
            nombre,
            id={"type": "dataset-btn", "index": nombre},
            n_clicks=0,
            style=BUTTON_STYLE
        )
        for nombre in datasets
    ], style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap"})

# ✅ Escribe en el Store GLOBAL declarado en main.py (storage_type="local")
@callback(
    Output("selected-dataset", "data"),
    Input({"type": "dataset-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def guardar_dataset_seleccionado(n_clicks_list):
    if not any(n_clicks_list):
        raise PreventUpdate
    return ctx.triggered_id.get("index")

@callback(
    Output("redirector", "pathname"),
    Input({"type": "dataset-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def redirigir_dataset(n_clicks_list):
    if not any(n_clicks_list):
        raise PreventUpdate

    triggered = ctx.triggered_id
    nombre = triggered.get("index")
    print(f"👉 Dataset seleccionado: {nombre}")

    dataset = Dataset(f"Data/{nombre}", nombre)
    dataset.upload_dataset(dataset.path)

    return f"/dataset"
