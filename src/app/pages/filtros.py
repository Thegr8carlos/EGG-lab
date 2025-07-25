import plotly.express as px
from dash import html, dcc, register_page

from app.components.PageContainer import get_page_container
from app.components.RigthComlumn import get_rightColumn


# registry of the the page 
register_page(__name__, path="/filtros", name="Filtros")



rigthColumn = get_rightColumn("filter")


layout = html.Div([
    html.Div([  # Contenido principal
        get_page_container("Filtros", "Description")
    ], style={"flex": "1", "padding": "1rem"}),

    html.Div([  # Barra lateral
        rigthColumn
    ])  # Asegúrate de que coincida con tu CSS
], style={"display": "flex"})
