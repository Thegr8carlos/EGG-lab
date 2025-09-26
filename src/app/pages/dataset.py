from dash import html, register_page, callback, Output, Input
from backend.classes.dataset import Dataset
from app.components.DataView import (
    get_dataset_view,
    register_dataset_clientside,
    register_dataset_legend,
)
from app.components.SideBar import get_sideBar

register_page(__name__, path="/dataset", name="Dataset")

# IDs con sufijo -dataset
CONTAINER_ID = "dataset-view-dataset"
STORE_ID     = "full-signal-data-dataset"
LABEL_STORE  = "label-color-store-dataset"
LEGEND_ID    = "dynamic-color-legend-dataset"
GRAPH_ID     = "signal-graph-dataset"
INTERVAL_ID  = "interval-component-dataset"

layout = html.Div(
    [
        html.Div(
            id="sidebar-wrapper",
            children=[get_sideBar("Data")],
            className="sideBar-container",
            style={"width": "260px", "padding": "1rem"},
        ),
        html.Div(
            # get_dataset_view ya trae el grid 2-panel
            get_dataset_view(
                container_id=CONTAINER_ID,
                full_signal_store_id=STORE_ID,
                label_color_store_id=LABEL_STORE,
                legend_container_id=LEGEND_ID,
                graph_id=GRAPH_ID,
                interval_id=INTERVAL_ID,
            ),
            style={
                "flex": "1",
                "padding": "1rem",
                "maxWidth": "1400px",
                "margin": "0 auto",
            },
        ),
    ],
    style={"display": "flex", "minHeight": "100vh"},
)

# Callback de carga (sin cambios)
@callback(
    Output(STORE_ID, "data"),
    Output(INTERVAL_ID, "disabled"),
    Input("selected-file-path", "data"),
)
def load_signal_data_dataset(selected_file_path):
    return Dataset.load_signal_data(selected_file_path)

register_dataset_clientside(graph_id=GRAPH_ID, interval_id=INTERVAL_ID, store_id=STORE_ID)
register_dataset_legend(legend_container_id=LEGEND_ID, store_id=STORE_ID)
