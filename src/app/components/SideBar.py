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



def build_file_tree(folder_structure):
    if not folder_structure:
        return html.Div("No folder loaded.", className="file-item")

    tree = []
    for name, content in folder_structure.items():
        if isinstance(content, dict):
            tree.append(
                html.Details(
                [html.Summary(name)] + build_file_tree(content)
                )
            )
        else:
            tree.append(html.Div(name, className="file-item"))
    
    print(tree)
    
    
    return tree


def get_sideBar(folder_path = "data/"):
    folder_structure = parse_local_folder(folder_path)
    
    
    
    return html.Div(
            html.Div(build_file_tree(folder_structure), className='file-tree')
    )