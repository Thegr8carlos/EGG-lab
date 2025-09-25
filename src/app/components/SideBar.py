import dash_bootstrap_components as dbc
from dash import html
import os
from pathlib import Path

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

def build_file_tree(folder_structure, base_path=""):
    if not folder_structure:
        return html.Div("Empty folder.", className="file-item")

    tree = []
    for name, content in folder_structure.items():
        full_path = os.path.join(base_path, name)
        if isinstance(content, dict):
            subtree = build_file_tree(content, base_path=full_path)
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
                    id={'type': 'file-item', 'path': full_path.replace("\\", "/")},
                    className="file-item",
                    n_clicks=0,
                    role="button",
                    tabIndex=0,
                ),
            )

    return tree

def get_sideBar(folder_path="Data/"):
    # Resolve folder_path
    resolved = None
    if Path(folder_path).exists():
        resolved = Path(folder_path)
    else:
        for p in [Path.cwd()] + list(Path(__file__).resolve().parents):
            candidate = p / folder_path
            if candidate.exists():
                resolved = candidate
                break

    if resolved is None:
        resolved_path = folder_path
    else:
        resolved_path = str(resolved)

    folder_structure = parse_local_folder(resolved_path)

    inner = build_file_tree(folder_structure)
    if not inner:
        inner = [html.Div(f"No files found (looked at: {resolved_path})", className="file-item")]

    # Solo cambiamos colores para usar variables de la paleta
    return html.Div(
        [
            html.Div("Archivos", style={"fontWeight": "600", "color": "var(--text)", "marginBottom": "8px"}),
            html.Div(inner, id="sideBar-div", className="file-tree"),
            html.Div(f"(resolved: {resolved_path})", style={"fontSize": "11px", "color": "var(--text-muted)", "marginTop": "6px"}),
        ],
        className="sideBar-root",
        style={"background": "var(--surface-2)", "padding": "8px", "borderRadius": "4px", "minHeight": "120px"},
    )
