import plotly.express as px
from dash import html, dcc, register_page, callback, Output, Input
import plotly.express as px


from app.components.PageContainer import get_page_container
from app.components.DashBoard import get_dashboard_container

#from shared.file_reader import is_data_loaded

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
    fig1 = px.line(x=[1, 2, 3, 4], y=[4, 1, 3, 2], title="Gráfico de Línea")
    fig2 = px.scatter(x=[1, 2, 3, 4], y=[2, 4, 3, 1], title="Gráfico de Dispersión")
    fig3 = px.density_heatmap(
        x=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        y=[1, 2, 3, 2, 3, 4, 1, 2, 3, 4],
        title="Heatmap de Densidad"
    )

    dashboard = get_dashboard_container([fig1, fig2, fig3], columns_per_row=2)

    return get_page_container("Prueba de dataset", "Verificación de tipos de gráficos", dashboard)
    # if pathname != "/dataset":
    #     raise PreventUpdate

    # if is_data_loaded():
    #     return get_page_container(
    #         "Dataset cargado",
    #         "Aquí irá la vista de los datos cargados."
    #     )
    # else:
    #     return get_page_container(
    #         "Sin dataset",
    #         "Por favor, sube un archivo en la sección de Cargar Datos."
    #     )
