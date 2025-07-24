import plotly.express as px
from dash import html, dcc, register_page

from app.components.PageContainer import get_page_container


# registry of the the page 
register_page(__name__, path="/filtros", name="Filtros")

layout = get_page_container(
    "filtros",
    "Description"
                            )
