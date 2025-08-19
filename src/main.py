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

from app.components.RigthComlumn import get_rightColumn



# backend dependencies 
from backend.classes.dataset import Dataset
from app.components.SideBar import get_sideBar



from backend.classes.Experiment import Experiment



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
                ),
                

                    
                
            ]
        ),
        
    
        
        
            
        dcc.Location(id="url"),
        dcc.Store(id="selected-file-path", storage_type = "local"), # Local: Persists across browser tabs/ windows and reloads

        
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
@app.callback(
    Output('selected-file-path','data'),  
    Output('url','pathname'),
    Input({'type': 'file-item', 'path': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def on_file_click(n_clicks_list):
    import json
    from dash import callback_context, no_update

    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = json.loads(ctx.triggered[0]['prop_id'].split(".n_clicks")[0])
    file_path = triggered_id['path']

    # ✅ Actualiza el store, pero NO cambies de ruta
    return file_path, no_update











    
if __name__ == "__main__":

    # We create a new experiment
    try:
        Experiment.create_blank_json()
    except Exception as e:
        print(f"⚠️ Error al crear experimento: {e}")
    
    
    app.run(debug=True) # comment this line if u want to test backend functionality 
    #print("🧐🔎🛠️💻  Backend Debug") # entry point to backend debug 
    #data = Dataset("path", "name")
    #response = data.upload_dataset("Data/nieto_inner_speech")
    #print(response)
    
