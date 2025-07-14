import plotly.express as px
from dash import html, dcc, register_page, callback, Output, Input
import plotly.express as px
import numpy as np

from app.components.PageContainer import get_page_container
from app.components.DashBoard import get_dashboard_container

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    
    print(pathname)
    
    
    #Simulating the signal for now will actually read it in from the proper file later on 
    
    signal = np.random.randn(1000,8)
    
    channel_figures = create_channel_plots(signal)
    
    # Put them into your dashboard container
    # dashboard = get_dashboard_container(channel_figures, columns_per_row=3)

    scrollable_graph = html.Div(
        dcc.Graph(figure=channel_figures, config={"displayModeBar": True}),
        style={
            "height": "80vh",
            "overflowY": "scroll",
            "border": "1px solid #ccc",
            "padding": "10px",
            "backgroundColor": "#fafafa"
        }
    )
     



    return get_page_container(
        "Señales Multicanal",
        "Gráfico desplazable con canales apilados.",
        scrollable_graph
    )
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



def create_channel_plots(signal_ndarray):
    num_channels = signal_ndarray.shape[1]
    time = np.arange(signal_ndarray.shape[0])

    # Create subplots: 1 column, N rows, shared X-axis
    fig = make_subplots(
        rows=num_channels,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,  # tighter spacing
        subplot_titles=[f"Channel {i+1}" for i in range(num_channels)]
    )

    for i in range(num_channels):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal_ndarray[:, i],
                mode="lines",
                name=f"Channel {i+1}"
            ),
            row=i+1,
            col=1
        )
        # Optional: hide Y axis ticks if needed
        fig.update_yaxes(title_text=f"Ch {i+1}", row=i+1, col=1)

    fig.update_layout(
        height=250 * num_channels,  # adjust for scrollable height
        showlegend=False,
        title="Señales Multicanal (Apiladas)",
        margin=dict(t=40, b=40)
    )
    
    return fig
