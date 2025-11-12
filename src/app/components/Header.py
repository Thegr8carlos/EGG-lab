import dash_bootstrap_components as dbc
from dash import html

def get_header(page_registry):
    header = html.Div([
        html.H1("EEG-Lab", className='header-title')
    ], className='header-container')
    return header
