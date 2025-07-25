import plotly.express as px
from dash import html, dcc, register_page, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go

from app.components.PageContainer import get_page_container
from app.components.DashBoard import get_dashboard_container

from backend.classes.dataset import Dataset

#from shared.file_reader import is_data_loaded

# registrar página
register_page(__name__, path="/dataset", name="Dataset")

# layout con contenedor dinámico
layout = html.Div(id="dataset-view")

# callback que se dispara al entrar a la ruta /dataset
@callback(
    Output("dataset-view", "children"),
    Input("url", "pathname"),
    Input("selected-file-path", "data")
)
def render_dataset_page(pathname, selected_file_path):
    if selected_file_path != None:
        print(f'📂 Archivo seleccionado: {selected_file_path.split(":")[0]}')
        dataset = Dataset("Data/inner_speech", "nieto_dataset")
        data = dataset.read_npy(f'{selected_file_path.split(":")[0]}')
        print(data.shape)

        figures = []
        for ch in range(min(20, data.shape[0])):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=data[ch],
                mode='lines',
                name=f"Canal {ch+1}",
                line=dict(width=1)
            ))
            fig.update_layout(
                title=f"Canal {ch+1}",
                height=250,
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            figures.append(fig)

        dashboard = get_dashboard_container(figures, columns_per_row=5)


        # fig = go.Figure()
        # for ch in range(5):
        #     fig.add_trace(go.Scatter(
        #         y=data[ch],
        #         mode='lines',
        #         name=f"Canal {ch+1}",
        #         line=dict(width=1)
        #     ))

        # fig.update_layout(
        #     title="Señales EEG de todos los canales",
        #     xaxis_title="Tiempo (muestras)",
        #     yaxis_title="Amplitud",
        #     height=600,
        #     showlegend=False
        # )
        # dashboard = get_dashboard_container([fig], columns_per_row=1)
        return get_page_container("Visualizacion de datos", "", dashboard)
    else : 
        fig1 = px.line(x=[1, 2, 3, 4], y=[4, 1, 3, 2], title="Gráfico de Línea")
        fig2 = px.scatter(x=[1, 2, 3, 4], y=[2, 4, 3, 1], title="Gráfico de Dispersión")
        fig3 = px.density_heatmap(
            x=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
            y=[1, 2, 3, 2, 3, 4, 1, 2, 3, 4],
            title="Heatmap de Densidad"
        )
        fig4 = px.density_heatmap(
            x=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
            y=[13, 2, 13, 452, 53, 4, 1, 2, 3, 4],
            title="Heatmap de Densidad"
        )
        fig5 = px.density_heatmap(
            x=[1, 2, 2, 3, 3, 3, 44, 4, 4, 4],
            y=[15232, 242, 5433, 3122, 133, 2344, 421, 672, 3, 4],
            title="Heatmap de Densidad"
        )

        dashboard = get_dashboard_container([fig1, fig2, fig3, fig4, fig5], columns_per_row=3)
        return get_page_container("Visualizacion de datos", "", dashboard)
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
