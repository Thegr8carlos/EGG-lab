from dash import html, dcc, register_page, callback, Input, Output, ALL, ctx, Input
from dash.exceptions import PreventUpdate
from app.components.PageContainer import get_page_container
from shared.fileUtils import get_data_folders
from backend.classes.dataset import Dataset
import dash
import os
register_page(__name__, path="/cargardatos", name="Cargar Datos")

BUTTON_STYLE = {
    "display": "inline-block",
    "padding": "0.5rem 1rem",
    "margin": "0.5rem",
    "backgroundColor": "#00C8A0",
    "color": "white",
    "textDecoration": "none",
    "border": "none",
    "borderRadius": "8px",
    "cursor": "pointer",
    "fontWeight": "bold",
    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.2)"
}

layout = get_page_container(
    "Cargando datos hehe",
    "Presiona el botón para mostrar las opciones",
    html.Div([
        dcc.Location(id="redirector", refresh=True),

        html.Button(
            "Cargar dataset",
            id="cargar-btn",
            n_clicks=0,
            style={
                'padding': '0.5rem 1rem',
                'backgroundColor': '#007bff',
                'color': 'white',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer'
            }
        ),

        html.Div(id="lista-opciones", style={"marginTop": "1rem", "textAlign": "center"}), 
        
        html.Div(id="folder-upload-container", style={"marginTop": "1rem", "textAlign": "center"}),


        
    ], style={'textAlign': 'center'})
)


@callback(
    Output("lista-opciones", "children"),
    Input("cargar-btn", "n_clicks")
)
def mostrar_opciones(n):
    if n == 0:
        return ""
    datasets = get_data_folders()
    return html.Div([
        html.Button(
            nombre,
            id={"type": "dataset-btn", "index": nombre},
            n_clicks=0,
            style=BUTTON_STYLE
        )
        for nombre in datasets
    ], style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap"})


@callback(
    Output("redirector", "pathname"),
    Input({"type": "dataset-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def redirigir_dataset(n_clicks_list):
    # Evita disparos falsos cuando la lista se crea (todos n_clicks == 0)
    if not any(n_clicks_list):
        raise PreventUpdate

    # ctx.triggered_id es el dict del botón presionado
    triggered = ctx.triggered_id
    nombre = triggered.get("index")
    print(f"👉 Dataset seleccionado: {nombre}")


    dataset = Dataset(f"Data/{nombre}",nombre)
    
    dataset.upload_dataset(dataset.path)

    
    
    
    
    
    
    # redirige con query param para la página /dataset
    return f"/dataset"
