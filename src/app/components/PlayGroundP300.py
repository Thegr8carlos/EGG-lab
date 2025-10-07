# app/components/PlayGroundP300.py
import dash_bootstrap_components as dbc
from dash import html, dcc

def _meta_card(meta_dict: dict, custom_dict: dict):
    # Render compacto de metadata (puedes estilizar a gusto)
    items_left  = []
    if isinstance(meta_dict, dict) and meta_dict:
        for k, v in meta_dict.items():
            items_left.append(html.Div([html.Code(str(k)), " → ", html.Span(str(v))]))

    items_right = []
    if isinstance(custom_dict, dict) and custom_dict:
        for k, v in custom_dict.items():
            items_right.append(html.Div([html.B(str(k) + ": "), html.Span(str(v))]))

    return dbc.Row(
        [
            dbc.Col(dbc.Card(dbc.CardBody(items_left), className="pg-card pg-card--meta"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody(items_right), className="pg-card pg-card--custom"), md=6),
        ],
        className="mb-3",
    )

def get_playGroundP300(title: str, description: str, meta_dict: dict, custom_dict: dict, graph_id: str, multi: bool = True):
    """
    Layout:
      - Fila 1: Metadata (izq) + Custom (der)
      - Fila 2: Plot stacked único (dcc.Graph id=graph_id)
    """
    return dbc.Container(
        fluid=True,
        className="page-container pg-wrap",
        children=[
            html.H2(title, className="page-title"),
            html.P(description, className="page-description"),

            _meta_card(meta_dict, custom_dict),

            # Plot card (scroll vertical lo da la altura del figure)
            dbc.Card(
                dbc.CardBody(
                    dcc.Graph(
                        id=graph_id,
                        figure={},
                        className="pg-card pg-card--plot",
                        style={"width": "100%"},
                        config={
                            "displaylogo": False,
                            "responsive": True,
                            "modeBarButtonsToRemove": [
                                "zoom","pan","select","lasso2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d","toImage"
                            ],
                        },
                    )
                ),
                className="pg-card",
            ),
        ],
    )
