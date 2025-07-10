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
        
        
       
        dcc.Location(id="url")
        
    ]
)




@app.callback(
    Output("sidebar-wrapper", "className"),
    Input("gif-btn", "n_clicks"),
    State("sidebar-wrapper", "className")
)
def toggle_sidebar(n_clicks, current_class):
    print("toggled sidebar")
    
    if "hidden" in current_class: 
        
        return current_class.replace("hidden","").strip() + " shown"
    else: 
        return current_class.replace("shown","") + " hidden"
    
    
    




if __name__ == "__main__":
    
    app.run(debug=True) # comment this line if u want to test backend functionality 
    #print("🧐🔎🛠️💻  Backend Debug") # entry point to backend debug 
    #data = Dataset("path", "name")
    #response = data.upload_dataset("dataset/inner_speech")
    #print(response)
    
