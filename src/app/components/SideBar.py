import dash_bootstrap_components as dbc
from dash import html
import os
def parse_local_folder(path):
    """
    Recursively walks through a folder and builds a nested dict.
    """
    
    
    
    tree = {}
    try:
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                tree[entry] = parse_local_folder(full_path)
            else:
                tree[entry] = None
    except Exception as e:
        print(f"Error reading folder {path}: {e}")
    return tree



def build_file_tree(folder_structure,base_path=""):
    if not folder_structure:
        return html.Div("Empty folder.", className="file-item")

    tree = []
    for name, content in folder_structure.items():
        full_path = os.path.join(base_path,name)
        if isinstance(content, dict):
            subtree = build_file_tree(content,base_path=full_path)
            # Ensure subtree is a list
            if not isinstance(subtree, list):
                subtree = [subtree]
            tree.append(
                html.Details(
                    [html.Summary(name)] + subtree
                )
            )
        else:
            tree.append(
                html.Div(
                    name,
                    id = {'type': 'file-item', 'path': full_path.replace("\\", "/")},
                    className="file-item",
                    n_clicks=0
                    ),
                )

    return tree



def get_sideBar(folder_path = "data/"):
    folder_structure = parse_local_folder(folder_path)
    
    
    
    return html.Div(
            
            html.Div(build_file_tree(folder_structure),id= "sideBar-div", className='file-tree')
    )