import plotly.express as px
from dash import html, dcc, register_page

# registry of the the page 
register_page(__name__, path="/filtros", name="Filtros")

layout = html.Div([
    html.H2("FILTROS page", style={"textAlign": "center"}),
])
