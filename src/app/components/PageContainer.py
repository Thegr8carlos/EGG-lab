import dash_bootstrap_components as dbc
from dash import html

def get_page_container(title: str, description: str, *children):
    
    return dbc.Container(
        fluid=True,
        className="page-container",
        children=[
            html.H2(title, className="page-title"),
             html.P(description, className="page-description"),
            *children
        ],
    )


