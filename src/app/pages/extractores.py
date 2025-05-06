import plotly.express as px
from dash import html, dcc, register_page

# registry of the the page 
register_page(__name__, path="/extractores", name="Extractores de Caracteristicas")

layout = html.Div([
    html.H2("EXTRACTORES page", style={"textAlign": "center"}),
])
