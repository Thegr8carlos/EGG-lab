from pathlib import Path
from dash import html, dcc, register_page, callback, Output, Input
from shared.fileUtils import get_dataset_metadata

from app.components.PageContainer import get_page_container
from app.components.PlayGround import get_playGround
from app.components.RigthComlumn import get_rightColumn
from app.components.SideBar import get_sideBar

# Registrar página
register_page(__name__, path="/filtros", name="Filtros")

# Layout con barra lateral
layout = html.Div(
    [
        html.Div(
            id="sidebar-wrapper",
            children=[get_sideBar("Data")],
            className="sideBar-container",
            style={"width": "260px", "padding": "1rem"},
        ),
        html.Div(
            id="pg-wrapper-filtros",
            children=get_playGround("Filtros", "Description", {}, {}),
            style={"flex": "1", "padding": "1rem"},
        ),
        html.Div(
            get_rightColumn("filter"),
            style={"width": "340px", "padding": "1rem"},
        ),
    ],
    style={"display": "flex"},
)

def create_metadata_section(meta: dict):
    """
    Retorna:
      - dict 'metadata' (clase -> color)
      - dict 'custom metadata' con:
          dataset_name, num_classes, sfreq, n_channels, eeg_unit
    Soporta JSON nuevo y antiguo.
    """
    if not isinstance(meta, dict):
        return {}, {}

    # 1) Colores por clase
    classes = meta.get("classes", []) or []
    class_color_map = {}
    for idx, label in enumerate(classes):
        hue = (idx * 47) % 360
        class_color_map[str(label)] = f"hsl({hue}, 70%, 50%)"

    # 2) Fs (sampling frequency) con fallback
    #   - nuevo: "sampling_frequency_hz"
    #   - antiguo: "unique_sfreqs": [..] o "sfreq"
    sfreq = (
        meta.get("sampling_frequency_hz")
        or meta.get("sfreq")
        or (
            (meta.get("unique_sfreqs") or [None])[0]
            if isinstance(meta.get("unique_sfreqs"), (list, tuple))
            else None
        )
    )
    if isinstance(sfreq, str):
        try:
            sfreq = float(sfreq)
        except Exception:
            sfreq = None

    # 3) # canales con fallback
    #   - nuevo: "n_channels" o len("channel_names")
    #   - antiguo: len("channel_name_union")
    n_channels = (
        meta.get("n_channels")
        or len(meta.get("channel_names") or [])
        or len(meta.get("channel_name_union") or [])
        or None
    )

    custom = {
        "dataset_name": meta.get("dataset_name"),
        "num_classes": meta.get("num_classes", len(classes)),
        "sfreq": float(sfreq) if isinstance(sfreq, (int, float)) else None,
        "n_channels": int(n_channels) if isinstance(n_channels, (int, float)) else None,
        "eeg_unit": meta.get("eeg_unit", "V"),
    }

    return class_color_map, custom


# Cuando cambie el Store global 'selected-dataset', re-renderiza el PlayGround
@callback(
    Output("pg-wrapper-filtros", "children"),
    Input("selected-dataset", "data")
)
def update_playground_desc(selected_dataset):
    desc = selected_dataset or "Selecciona un dataset en 'Cargar Datos'"

    # Si no hay dataset seleccionado aún, devuelve el playground vacío
    if not selected_dataset:
        return get_playGround("Filtros", desc, {}, {})

    # Lee el JSON de Aux/<dataset>/dataset_metadata.json
    try:
        meta = get_dataset_metadata(selected_dataset)
    except Exception as e:
        # Si falla la lectura del JSON, muestra el error en la descripción y no rompe la UI
        return get_playGround("Filtros", f"{desc} (sin metadata: {e})", {}, {})

    # Construye los dos dicts requeridos
    meta_dict, custom_dict = create_metadata_section(meta)

    # Pásalos como 3er y 4º argumento (ya los tienes disponibles para tu PlayGround)
    return get_playGround("Filtros", desc, meta_dict, custom_dict)
