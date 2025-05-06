import plotly.express as px
from dash import html, dcc, register_page

# registry of the the page 
register_page(__name__, path="/hablainterna", name="Modelado Habla Interna")

layout = html.Div([
    html.H2("Modelado Habla Interna page", style={"textAlign": "center"}),
])




