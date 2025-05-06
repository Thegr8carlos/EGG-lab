import plotly.express as px
from dash import html, dcc, register_page

# registry of the the page 
register_page(__name__, path="/", name="archivo")

layout = html.Div([
    html.H2("Inicio de la aplicacion, chance como este es el index, como una explicacion aca en corto de que se puede hacer y como", style={"textAlign": "center"}),
])
