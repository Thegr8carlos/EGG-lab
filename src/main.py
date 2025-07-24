# ui dependencys 
from dash import Dash, dcc, html, page_container, page_registry,Input, Output, State, ALL
from dash import dcc, html, callback
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc  
import json

# internal dependencies

# ui components  
from app.components.NavBar import get_navBar
from app.components.Header import get_header


# backend dependencies 
from backend.classes.dataset import Dataset
from app.components.SideBar import get_sideBar


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


#Adding a dcc.Store for a client side memory store 
dcc.Store(id='selected-file-path', storage_type='local')  # or 'local' if you want it to persist longer



navBar = get_navBar(page_registry)
header = get_header(page_registry)
app.layout = html.Div(

    id="app-container",  # 👈 le damos un id para estilizar
    children=[
        header,
        navBar,
        # html.Button("Show sidebar", id= "toggle-sidebar-btn",n_clicks =0 ),
        html.Div(
            id= "main-content-wrapper",
            style= {"display": "flex", "flexDirection":"row"},
            children= [
                html.Div(
                    id = "sidebar-wrapper", 
                    children = get_sideBar("Data"), 
                    # style = {"transition" : "transform 0.3s ease-in-out"},
                    className = "sideBar-container",
                ),
                html.Div(
                    id="page-content",
                    children=page_container,
                    style={"flex": 1, "padding": "20px"}
                )
            ]
        ),
        
        
            
        dcc.Location(id="url"),
        dcc.Store(id="selected-file-path"),

        
    ]
)




@app.callback(
    Output("sidebar-wrapper", "className"),
    Input("gif-btn", "n_clicks"),
    State("sidebar-wrapper", "className")
)
def toggle_sidebar(n_clicks, current_class):
    
    if "hidden" in current_class: 
        
        return current_class.replace("hidden","").strip() + " shown"
    else: 
        return current_class.replace("shown","") + " hidden"
    

#Listener for list elements     
# @app.callback(
#     Output('sideBar-div', 'children', allow_duplicate=True),  # or another output like a file preview area
#     Input({'type': 'file-item', 'path': ALL}, 'n_clicks'),
#     prevent_initial_call=True
# )
@app.callback(
    Output('selected-file-path','data'),  # or another output like a file preview area
    Output('url','pathname'),
    Input({'type': 'file-item', 'path': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def on_file_click(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    # Find which file was clicked
    # triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    triggered_id = ctx.triggered[0]['prop_id'].split(".n_clicks")[0]
    triggered_id = json.loads(triggered_id)  # {'type': 'file-item', 'path': 'some/file.txt'}
    
    
    file_path = triggered_id['path']
    print(file_path)
    
    return file_path, "/dataset"


if __name__ == "__main__":
    
    app.run(debug=True) # comment this line if u want to test backend functionality 
    #print("🧐🔎🛠️💻  Backend Debug") # entry point to backend debug 
    #data = Dataset("path", "name")
    #response = data.upload_dataset("dataset/inner_speech")
    #print(response)
    
