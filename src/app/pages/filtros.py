import plotly.express as px
from dash import html, dcc, register_page, callback, Output, Input
from backend.classes.dataset import Dataset
from app.components.PageContainer import get_page_container
from app.components.RigthComlumn import get_rightColumn
from app.components.DataView import (
    get_dataset_view,
    register_dataset_clientside,
    register_dataset_legend,
)

# Registrar página
register_page(__name__, path="/filtros", name="Filtros")

# Barra lateral
rightColumn = get_rightColumn("filter")

# ---- IDs con sufijo -filters ----
CONTAINER_ID = "dataset-view-filters"
STORE_ID     = "full-signal-data-filters"
LABEL_STORE  = "label-color-store-filters"
LEGEND_ID    = "dynamic-color-legend-filters"
GRAPH_ID     = "signal-graph-filters"
INTERVAL_ID  = "interval-component-filters"

# Contenido principal (viewer de señales + leyenda + interval)
viewer = get_dataset_view(
    container_id=CONTAINER_ID,
    full_signal_store_id=STORE_ID,
    label_color_store_id=LABEL_STORE,
    legend_container_id=LEGEND_ID,
    graph_id=GRAPH_ID,
    interval_id=INTERVAL_ID,
)

# Layout con barra lateral
layout = html.Div(
    [
        html.Div(
            get_page_container("Filtros", "Description", viewer),
            style={"flex": "1", "padding": "1rem"},
        ),
        html.Div(rightColumn),  # barra lateral
    ],
    style={"display": "flex"},
)

# --- Callback: carga datos -> escribe Store y habilita Interval ---
@callback(
    Output(STORE_ID, "data"),
    Output(INTERVAL_ID, "disabled"),
    Input("selected-file-path", "data"),  # asegúrate de que exista este Store en tu app
)
def load_signal_data_filters(selected_file_path):
    return Dataset.load_signal_data(selected_file_path)

# --- Registrar callbacks (clientside para la figura, server para la leyenda) ---
register_dataset_clientside(graph_id=GRAPH_ID, interval_id=INTERVAL_ID, store_id=STORE_ID)
register_dataset_legend(legend_container_id=LEGEND_ID, store_id=STORE_ID)
