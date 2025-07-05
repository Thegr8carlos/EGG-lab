# ui dependencys 
from dash import Dash, dcc, html, page_container, page_registry
import dash_bootstrap_components as dbc  

# internal dependencies

# ui components  
from app.components.NavBar import get_navBar
from app.components.Header import get_header


# backend dependencies 
from backend.classes.dataset import Dataset


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


app.layout = html.Div(
    id="app-container",  # 👈 le damos un id para estilizar
    children=[
        header,
        navBar,
        dcc.Location(id="url"),
        page_container
    ]
)



if __name__ == "__main__":
    
    app.run(debug=True) # comment this line if u want to test backend functionality 
    #print("🧐🔎🛠️💻  Backend Debug") # entry point to backend debug 
    #data = Dataset("path", "name")
    #response = data.upload_dataset("dataset/inner_speech")
    #print(response)
    
