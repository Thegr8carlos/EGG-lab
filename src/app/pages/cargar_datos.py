import plotly.express as px
from dash import html, dcc, register_page

# internal dependencies
from components.PageContainer import get_page_container


# registry of the the page 
register_page(__name__, path="/cargardatos", name="Cargar Datos")


upload_button = html.Div(
    dcc.Upload(
        # El child de Upload será nuestro botón
        children=html.Button(
            "Seleccionar archivo",
            style={
                'padding': '0.5rem 1rem',
                'backgroundColor': '#007bff',
                'color': 'white',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer'
            }
        ),
        multiple=False,
        # Aseguramos que el Upload no amplíe el botón por defecto
        style={'display': 'inline-block'}
    ),
    # Y lo alineamos a la izquierda de su contenedor padre
    style={'textAlign': 'center'}
)

# 3) Componemos todo con nuestro wrapper
layout = get_page_container(
    "Cargando datos",
    "porfavor sube datos de tipo .egg, dbf, etc",
    upload_button
)