import plotly.express as px
from dash import html, dcc, register_page

# registry of the the page 
register_page(__name__, path="/p300", name="Modelado p300")

layout = html.Div([
    html.H2("P300 page", style={"textAlign": "center"}),
])
