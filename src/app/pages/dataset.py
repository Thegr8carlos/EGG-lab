import plotly.express as px
from dash import html, dcc, register_page, callback, Output, Input
from app.components.PageContainer import get_page_container
from backend.file_reader import is_data_loaded

# registrar página
register_page(__name__, path="/dataset", name="Dataset")

# layout con contenedor dinámico
layout = html.Div(id="dataset-view")

# callback que se dispara al entrar a la ruta /dataset
@callback(
    Output("dataset-view", "children"),
    Input("url", "pathname")
)
def render_dataset_page(pathname):
    if pathname != "/dataset":
        raise PreventUpdate

    if is_data_loaded():
        return get_page_container(
            "Dataset cargado",
            "Aquí irá la vista de los datos cargados."
        )
    else:
        return get_page_container(
            "Sin dataset",
            "Por favor, sube un archivo en la sección de Cargar Datos."
        )
