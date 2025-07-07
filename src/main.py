# ui dependencys 
from dash import Dash, dcc, html, page_container, page_registry,Input, Output, State

import dash_bootstrap_components as dbc  

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



navBar = get_navBar(page_registry)
header = get_header(page_registry)
print(get_sideBar("Data"))
app.layout = html.Div(
    id="app-container",  # 👈 le damos un id para estilizar
    children=[
        header,
        navBar,
        html.Button("Show sidebar", id= "toggle-sidebar-btn",n_clicks =0 ),
        html.Div(
            id = "sidebar-wrapper", 
            children = get_sideBar("Data"), 
            style = {"display" : "none"}
        ),
        dcc.Location(id="url"),
        page_container
    ]
)




@app.callback(
    Output("sidebar-wrapper", "style"),
    Input("toggle-sidebar-btn", "n_clicks"),
    State("sidebar-wrapper", "style")
)
def toggle_sidebar(n_clicks, current_style):
    print("toggled sidebar")
    
    
    
    
    if n_clicks and current_style.get("display") == "none":
        
        return {"display": "block"}
    return {"display": "none"}




if __name__ == "__main__":
    
    app.run(debug=True) # comment this line if u want to test backend functionality 
    #print("🧐🔎🛠️💻  Backend Debug") # entry point to backend debug 
    #data = Dataset("path", "name")
    #response = data.upload_dataset("dataset/inner_speech")
    #print(response)
    
