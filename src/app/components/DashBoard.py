from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objs as go
from typing import List

def get_dashboard_container(figures: List[go.Figure], columns_per_row: int = 2):
    rows = []
    for i in range(0, len(figures), columns_per_row):
        cols = [
            dbc.Col(dcc.Graph(figure=fig), width=int(12/columns_per_row))
            for fig in figures[i:i+columns_per_row]
        ]
        rows.append(dbc.Row(cols, className="mb-4"))

    return html.Div(
        className="dashboard-scroll-wrapper",
        children=[
            dbc.Container(
                fluid=True,
                className="dashboard-container",
                children=rows
            )
        ]
    )



def get_dashboard_container_dynamic(graph_id="signal-graph"):
    return html.Div(
        className="dashboard-scroll-wrapper",
        # ⬅️ quita límites y scroll interno del wrapper
        style={"width": "100%", "overflow": "visible"},
        children=[
            dbc.Container(
                fluid=True,
                className="dashboard-container",
                # ⬅️ quita padding lateral del container bootstrap
                style={"paddingLeft": 0, "paddingRight": 0},
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(
                                    id=graph_id,
                                    figure=None,
                                    # ⬅️ tamaño real del lienzo
                                    style={"width": "100%", "height": "82vh"},
                                    config={"displayModeBar": True, "responsive": True},
                                ),
                                width=12,
                                # ⬅️ sin padding en la columna
                                style={"paddingLeft": 0, "paddingRight": 0},
                            )
                        ],
                        className="mb-0",
                        # ⬅️ sin márgenes de fila
                        style={"marginLeft": 0, "marginRight": 0},
                    )
                ],
            )
        ],
    )