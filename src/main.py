# external dependencies
from dash import Dash, dcc, html, page_container, page_registry
import dash_bootstrap_components as dbc 
# internal dependencies
from components.NavBar import get_navBar


# using this if in the future more config is needed
config = {
    "name_app": "Dashboard EEG",
    "title_app": "BCI lab - Demo"
}



app = Dash(__name__,
    use_pages=True,
    pages_folder="app/pages",
    external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
app.title = config["name_app"]



navBar = get_navBar(page_registry)

app.layout = html.Div([
    html.H1(config["title_app"], style={'textAlign': 'center'}),
    navBar, # inserts the nav bar in the main container 
    dcc.Location(id="url"),
    page_container, # container that storages pages
    #  html.Footer(               # <-- footer fixed
    #     html.Div([
    #         html.P("© 2025 BCI lab. Todos los derechos reservados.")
    #     ], style={'textAlign': 'center', 'padding': '1rem'}),
    #     style={
    #         'backgroundColor': '#111',
    #         'color': 'rgba(255,255,255,0.7)',
    #         'marginTop': '2rem'
    #     }
    # )
])


if __name__ == "__main__":
    app.run(debug=True)
    
