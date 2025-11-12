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

# Allow pages to register callbacks for components that are not present
# in the global layout (we'll let pages render `sidebar-wrapper` themselves).
app.config.suppress_callback_exceptions = True


#Adding a dcc.Store for a client side memory store 
dcc.Store(id='selected-file-path', storage_type='local')  # or 'local' if you want it to persist longer



navBar = get_navBar(page_registry)
header = get_header(page_registry)

app.layout = html.Div(
    id="app-container",
    children=[
        #header,
        navBar,
        html.Div(id="main-content-wrapper", children=page_container),
        dcc.Location(id="url"),
        dcc.Store(id="selected-file-path", storage_type="local"),
        dcc.Store(id="selected-dataset", storage_type="local"),
    ],
)


# Sidebar visibility is now controlled by pages which render the `sidebar-wrapper`.
# The old toggle callback (gif button) was removed so the sidebar stays visible.
    

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
    print(f"🧐 Selected file: {file_path}")
    # ✅ Actualiza el store, pero NO cambies de ruta
    return file_path, no_update











    
if __name__ == "__main__":

    # We create a new experiment
    try:
        Experiment.create_blank_json()
    except Exception as e:
        print(f"⚠️ Error al crear experimento: {e}")
    
    
    
    #app.run(debug=True) # comment this line if u want to test backend functionality 
    app.run(debug=True, use_reloader=False, port=8090, host="127.0.0.1")
    #print("🧐🔎🛠️💻  Backend Debug") # entry point to backend debug 
    #data = Dataset("path", "name")
    #response = data.upload_dataset("Data/nieto_inner_speech")
    #print(response)
    
