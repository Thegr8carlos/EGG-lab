from dash import html, dcc, register_page, callback, Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from app.components.PageContainer import get_page_container
from shared.fileUtils import get_data_folders
from backend.classes.dataset import Dataset
import os
from pathlib import Path

register_page(__name__, path="/cargardatos", name="Cargar Datos")

# Estilo común para botones
BUTTON_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "gap": "8px",
    "padding": "0.65rem 1.1rem",
    "margin": "0.25rem 0.5rem",
    "backgroundColor": "var(--color-2)",
    "color": "var(--color-3)",
    "border": "1px solid color-mix(in srgb, var(--color-4) 60%, transparent)",
    "borderRadius": "10px",
    "cursor": "pointer",
    "fontWeight": 600,
    "letterSpacing": "0.2px",
    "boxShadow": "0 6px 14px rgba(0,0,0,0.25)",
    "transition": "transform 120ms ease",
}

PANEL_STYLE = {
    "maxWidth": "980px",
    "margin": "0 auto",
    "background": "linear-gradient(180deg, color-mix(in srgb, var(--color-2) 85%, transparent), var(--color-1))",
    "border": "1px solid color-mix(in srgb, var(--color-4) 35%, transparent)",
    "borderRadius": "16px",
    "padding": "1.25rem",
    "boxShadow": "0 10px 24px rgba(0,0,0,0.35)",
}

TOOLBAR_STYLE = {
    "display": "flex",
    "flexWrap": "wrap",
    "justifyContent": "center",
    "gap": "0.5rem 0.75rem",
    "padding": "0.5rem 0.25rem",
}

HELP_STYLE = {
    "textAlign": "center",
    "opacity": 0.85,
    "fontSize": "0.95rem",
    "margin": "0.25rem 0 0.75rem 0",
    "color": "var(--color-3)",
}

DATASET_BUTTON_STYLE = {
    **BUTTON_STYLE,
    "backgroundColor": "var(--color-2)",
    "border": "1px solid color-mix(in srgb, var(--color-5) 45%, transparent)",
    "boxShadow": "0 4px 10px rgba(0,0,0,0.30)",
}

PENDING_BUTTON_STYLE = {
    **BUTTON_STYLE,
    "backgroundColor": "var(--color-2)",
    "border": "1px solid color-mix(in srgb, var(--color-4) 45%, transparent)",
    "boxShadow": "0 4px 10px rgba(0,0,0,0.30)",
}

layout = get_page_container(
    "Carga y gestión de datos EEG",
    "Procesa datasets nuevos (.bdf, .edf, .vhdr) o selecciona uno ya procesado.",
    html.Div(
        style=PANEL_STYLE,
        children=[
            dcc.Location(id="redirector", refresh=True),

            # Barra de acciones - SOLO 2 BOTONES
            html.Div(
                style=TOOLBAR_STYLE,
                children=[
                    html.Button(
                        "📂 Cargar Dataset",
                        id="show-pending-btn",
                        n_clicks=0,
                        style={**BUTTON_STYLE, "borderColor": "var(--color-5)", "backgroundColor": "#00C8A0"},
                        title="Mostrar datasets en Data/ que aún no han sido procesados"
                    ),
                    html.Button(
                        "📋 Listar Datasets",
                        id="list-datasets-btn",
                        n_clicks=0,
                        style={**BUTTON_STYLE, "borderColor": "var(--color-4)"},
                        title="Mostrar datasets ya procesados en el sistema"
                    ),
                ],
            ),

            # Loading indicator
            dcc.Loading(
                id="loading-datasets",
                type="default",
                children=html.Div(id="loading-output"),
                style={"marginTop": "1rem"}
            ),

            # Feedback de procesamiento
            html.Div(id="processing-feedback", style={"marginTop": "1rem", "textAlign": "center"}),

            # Texto de ayuda
            html.Div(
                children=[
                    html.Div("¿Qué puedes hacer aquí?", style={
                        "fontWeight": 700, "letterSpacing": "0.2px",
                        "margin": "0.2rem 0 0.4rem 0", "color": "var(--color-3)"
                    }),
                    html.Ul([
                        html.Li("Cargar Dataset: Muestra datasets en Data/ que no tienen Aux/ generado. Al hacer clic, los procesa y genera archivos .npy, labels y eventos."),
                        html.Li("Listar Datasets: Muestra datasets ya procesados (con archivos en Aux/) listos para trabajar."),
                    ], style={"margin": "0 0 0.75rem 1.25rem", "color": "var(--color-3)", "opacity": 0.9}),
                ],
                style={"textAlign": "left"}
            ),

            html.Div(
                "Selecciona un dataset ya cargado para continuar con el análisis, o procesa uno nuevo.",
                style=HELP_STYLE
            ),

            # Lista de datasets (pendientes o procesados)
            html.Div(id="datasets-list", style={"marginTop": "0.5rem", "textAlign": "center"}),
        ]
    ),
)

# =============================================================================
# Helper: Obtener datasets realmente procesados (con .npy en Aux/)
# =============================================================================
def get_processed_datasets():
    """
    Retorna lista de datasets que están realmente procesados en Aux/.

    Un dataset se considera "procesado" solo si tiene:
    - Carpeta Aux/{nombre}/ existente
    - Al menos un archivo .npy procesado
    - Opcionalmente dataset_metadata.json
    """
    aux_path = Path("Aux")

    if not aux_path.exists():
        return []

    # Obtener todas las carpetas en Aux/
    aux_folders = [f.name for f in aux_path.iterdir() if f.is_dir()]

    # Verificar cuáles realmente tienen archivos .npy procesados
    truly_processed = []
    for folder_name in aux_folders:
        aux_folder = aux_path / folder_name

        # Verificar si tiene archivos .npy procesados (buscar recursivamente)
        npy_files = list(aux_folder.rglob("*.npy"))

        # Verificar si tiene dataset_metadata.json
        has_metadata = (aux_folder / "dataset_metadata.json").exists()

        # Considerar procesado si tiene .npy O metadata
        if npy_files or has_metadata:
            truly_processed.append(folder_name)
            print(f"[get_processed_datasets] ✅ {folder_name} está procesado ({len(npy_files)} .npy, metadata={has_metadata})")
        else:
            print(f"[get_processed_datasets] ⏭️ {folder_name} tiene carpeta Aux/ pero está vacío, se omite")

    print(f"[get_processed_datasets] Aux folders: {aux_folders}")
    print(f"[get_processed_datasets] Truly processed: {truly_processed}")

    return truly_processed

# =============================================================================
# Helper: Obtener datasets pendientes (en Data/ pero sin Aux/)
# =============================================================================
def get_pending_datasets():
    """
    Retorna lista de datasets que están en Data/ pero no tienen Aux/ generado correctamente.

    Un dataset se considera "válido para mostrar" solo si:
    - Tiene archivos .bdf, .edf o .vhdr en Data/{nombre}/

    Un dataset se considera "procesado" solo si tiene:
    - Carpeta Aux/{nombre}/ existente
    - Al menos un archivo .npy procesado
    - Carpeta dataset_metadata.json (opcional pero recomendado)
    """
    data_path = Path("Data")
    aux_path = Path("Aux")

    if not data_path.exists():
        return []

    # Obtener todas las carpetas en Data/
    data_folders = [f.name for f in data_path.iterdir() if f.is_dir()]

    # Verificar cuáles tienen archivos .bdf o .edf
    valid_datasets = []
    for folder_name in data_folders:
        data_folder = data_path / folder_name

        # Buscar archivos .bdf y .edf recursivamente
        bdf_files = list(data_folder.rglob("*.bdf"))
        edf_files = list(data_folder.rglob("*.edf"))
        vhdr_files = list(data_folder.rglob("*.vhdr"))

        total_eeg_files = len(bdf_files) + len(edf_files) + len(vhdr_files)

        if total_eeg_files > 0:
            valid_datasets.append(folder_name)
            print(f"[get_pending_datasets] 📁 {folder_name} tiene {total_eeg_files} archivos EEG (.bdf: {len(bdf_files)}, .edf: {len(edf_files)}, .vhdr: {len(vhdr_files)})")
        else:
            print(f"[get_pending_datasets] ⏭️ {folder_name} no tiene archivos .bdf/.edf/.vhdr, se omite")

    # Verificar cuáles realmente están procesadas (con archivos .npy)
    truly_processed = []
    for folder_name in valid_datasets:
        aux_folder = aux_path / folder_name

        # Verificar si existe Aux/{nombre}/
        if not aux_folder.exists():
            continue

        # Verificar si tiene archivos .npy procesados (buscar recursivamente)
        npy_files = list(aux_folder.rglob("*.npy"))

        # Verificar si tiene dataset_metadata.json
        has_metadata = (aux_folder / "dataset_metadata.json").exists()

        # Considerar procesado si tiene .npy O metadata
        if npy_files or has_metadata:
            truly_processed.append(folder_name)
            print(f"[get_pending_datasets] ✅ {folder_name} está procesado ({len(npy_files)} .npy, metadata={has_metadata})")
        else:
            print(f"[get_pending_datasets] ⚠️ {folder_name} tiene Aux/ pero está vacío")

    # Filtrar: solo las que tienen archivos EEG pero NO están realmente procesadas
    pending = [name for name in valid_datasets if name not in truly_processed]

    print(f"[get_pending_datasets] Data folders: {data_folders}")
    print(f"[get_pending_datasets] Valid datasets (with .bdf/.edf/.vhdr): {valid_datasets}")
    print(f"[get_pending_datasets] Truly processed: {truly_processed}")
    print(f"[get_pending_datasets] Pending: {pending}")

    return pending

# =============================================================================
# Callback 1: Mostrar datasets pendientes de procesar
# =============================================================================
@callback(
    [Output("datasets-list", "children"),
     Output("processing-feedback", "children"),
     Output("loading-output", "children")],
    Input("show-pending-btn", "n_clicks")
)
def show_pending_datasets(n):
    """Muestra datasets en Data/ que no tienen Aux/ generado"""
    if n == 0:
        return "", "", None

    pending = get_pending_datasets()

    if not pending:
        return html.Div(
            "✅ No hay datasets pendientes. Todos los datasets en Data/ ya han sido procesados.",
            style={
                "color": "var(--text-muted)",
                "padding": "1rem",
                "animation": "fadeIn 0.5s ease-in"
            }
        ), "", None

    return html.Div([
        html.Div(
            f"📦 Datasets pendientes de procesar ({len(pending)}):",
            style={
                "fontWeight": "bold",
                "marginBottom": "0.75rem",
                "color": "var(--text)",
                "animation": "fadeIn 0.5s ease-in"
            }
        ),
        html.Div([
            html.Button(
                [
                    html.Span("🔄 ", style={"marginRight": "0.5rem"}),
                    html.Span(nombre)
                ],
                id={"type": "pending-dataset-btn", "index": nombre},
                n_clicks=0,
                style={
                    **PENDING_BUTTON_STYLE,
                    "animation": f"slideIn 0.3s ease-out {i * 0.1}s both"
                },
                className="dataset-btn-hover"
            )
            for i, nombre in enumerate(pending)
        ], style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap", "gap": "0.5rem"})
    ]), "", None

# =============================================================================
# Callback 2: Listar datasets ya procesados
# =============================================================================
@callback(
    [Output("datasets-list", "children", allow_duplicate=True),
     Output("processing-feedback", "children", allow_duplicate=True),
     Output("loading-output", "children", allow_duplicate=True)],
    Input("list-datasets-btn", "n_clicks"),
    prevent_initial_call=True
)
def list_processed_datasets(n):
    """Muestra datasets que realmente tienen archivos .npy procesados en Aux/"""
    if n == 0:
        return "", "", None

    datasets = get_processed_datasets()  # ✅ Ahora usa la nueva función

    if not datasets:
        return html.Div(
            "No se encontraron datasets procesados. Usa 'Cargar Dataset' para procesar uno nuevo.",
            style={
                "color": "var(--text-muted)",
                "padding": "1rem",
                "animation": "fadeIn 0.5s ease-in"
            }
        ), "", None

    return html.Div([
        html.Div(
            f"✅ Datasets procesados ({len(datasets)}):",
            style={
                "fontWeight": "bold",
                "marginBottom": "0.75rem",
                "color": "var(--text)",
                "animation": "fadeIn 0.5s ease-in"
            }
        ),
        html.Div([
            html.Button(
                [
                    html.Span("✓ ", style={"marginRight": "0.5rem", "color": "#38FF97"}),
                    html.Span(nombre)
                ],
                id={"type": "dataset-btn", "index": nombre},
                n_clicks=0,
                style={
                    **DATASET_BUTTON_STYLE,
                    "animation": f"slideIn 0.3s ease-out {i * 0.1}s both"
                },
                className="dataset-btn-hover"
            )
            for i, nombre in enumerate(datasets)
        ], style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap", "gap": "0.5rem"})
    ]), "", None

# =============================================================================
# Callback 3: Procesar dataset pendiente
# =============================================================================
@callback(
    [Output("datasets-list", "children", allow_duplicate=True),
     Output("processing-feedback", "children", allow_duplicate=True),
     Output("loading-output", "children", allow_duplicate=True)],
    Input({"type": "pending-dataset-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def process_pending_dataset(n_clicks_list):
    """Procesa un dataset pendiente cuando se hace clic en él"""
    if not any(n_clicks_list):
        raise PreventUpdate

    triggered = ctx.triggered_id
    dataset_name = triggered.get("index")

    print(f"\n[PROCESAR DATASET] Iniciando procesamiento de: {dataset_name}")

    # Construir ruta
    dataset_path = f"Data/{dataset_name}"

    # Validar que existe
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        return "", html.Div([
            html.Span("❌ Error: ", style={"fontWeight": "bold", "color": "#FF235A"}),
            html.Span(f"La carpeta Data/{dataset_name} no existe o no es válida.")
        ], style={
            "color": "var(--text)",
            "padding": "0.5rem",
            "backgroundColor": "rgba(255, 35, 90, 0.1)",
            "borderRadius": "8px",
            "animation": "shake 0.5s ease-in-out"
        }), None

    # Validar que contenga archivos .bdf, .edf o .vhdr
    folder_path_obj = Path(dataset_path)
    bdf_files = list(folder_path_obj.rglob("*.bdf"))
    edf_files = list(folder_path_obj.rglob("*.edf"))
    vhdr_files = list(folder_path_obj.rglob("*.vhdr"))
    total_files = len(bdf_files) + len(edf_files) + len(vhdr_files)

    if total_files == 0:
        return "", html.Div([
            html.Span("⚠️ Advertencia: ", style={"fontWeight": "bold", "color": "#FFD400"}),
            html.Span(f"No se encontraron archivos .bdf, .edf o .vhdr en Data/{dataset_name}."),
        ], style={
            "color": "var(--text)",
            "padding": "0.5rem",
            "backgroundColor": "rgba(255, 212, 0, 0.1)",
            "borderRadius": "8px",
            "animation": "fadeIn 0.5s ease-in"
        }), None

    # Procesar dataset
    try:
        print(f"[PROCESAR DATASET] Encontrados {len(bdf_files)} .bdf, {len(edf_files)} .edf, {len(vhdr_files)} .vhdr")

        dataset = Dataset(dataset_path, dataset_name)
        result = dataset.upload_dataset(dataset_path)

        if result.get("status") == 200:
            num_files = len(result.get("files", []))

            # Actualizar lista de pendientes
            remaining_pending = get_pending_datasets()

            pending_list = ""
            if remaining_pending:
                pending_list = html.Div([
                    html.Div(
                        f"📦 Datasets pendientes ({len(remaining_pending)}):",
                        style={
                            "fontWeight": "bold",
                            "marginBottom": "0.75rem",
                            "color": "var(--text)",
                            "animation": "fadeIn 0.5s ease-in"
                        }
                    ),
                    html.Div([
                        html.Button(
                            [
                                html.Span("🔄 ", style={"marginRight": "0.5rem"}),
                                html.Span(nombre)
                            ],
                            id={"type": "pending-dataset-btn", "index": nombre},
                            n_clicks=0,
                            style={
                                **PENDING_BUTTON_STYLE,
                                "animation": f"slideIn 0.3s ease-out {i * 0.1}s both"
                            },
                            className="dataset-btn-hover"
                        )
                        for i, nombre in enumerate(remaining_pending)
                    ], style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap", "gap": "0.5rem"})
                ])
            else:
                pending_list = html.Div(
                    "✅ No hay más datasets pendientes.",
                    style={
                        "color": "var(--text-muted)",
                        "padding": "1rem",
                        "animation": "fadeIn 0.5s ease-in"
                    }
                )

            return pending_list, html.Div([
                html.Div([
                    html.Span("✅ ", style={"fontSize": "1.5rem", "marginRight": "0.5rem"}),
                    html.Span("Éxito", style={"fontWeight": "bold", "color": "#38FF97", "fontSize": "1.2rem"})
                ], style={"marginBottom": "0.5rem"}),
                html.Span(f"Dataset '{dataset_name}' procesado correctamente."),
                html.Br(),
                html.Span(f"Archivos procesados: {num_files} (.bdf: {len(bdf_files)}, .edf: {len(edf_files)}, .vhdr: {len(vhdr_files)})",
                         style={"fontSize": "0.85rem", "opacity": "0.8"}),
                html.Br(),
                html.Span(f"Archivos .npy, Labels y Events generados en Aux/{dataset_name}/",
                         style={"fontSize": "0.85rem", "opacity": "0.8"}),
                html.Br(),
                html.Br(),
                html.Div([
                    html.Span("💡 ", style={"marginRight": "0.5rem"}),
                    html.Span("Ahora puedes hacer clic en 'Listar Datasets' para seleccionarlo.",
                             style={"fontStyle": "italic", "fontSize": "0.9rem"})
                ])
            ], style={
                "color": "var(--text)",
                "padding": "1rem",
                "backgroundColor": "rgba(56, 255, 151, 0.1)",
                "borderRadius": "var(--radius-md)",
                "border": "1px solid rgba(56, 255, 151, 0.3)",
                "animation": "successPulse 0.6s ease-in-out"
            }), None
        else:
            error_msg = result.get("message", "Error desconocido")
            return "", html.Div([
                html.Span("❌ Error: ", style={"fontWeight": "bold", "color": "#FF235A"}),
                html.Span(f"No se pudo procesar el dataset. {error_msg}")
            ], style={
                "color": "var(--text)",
                "padding": "0.5rem",
                "backgroundColor": "rgba(255, 35, 90, 0.1)",
                "borderRadius": "8px",
                "animation": "shake 0.5s ease-in-out"
            }), None

    except Exception as e:
        print(f"[PROCESAR DATASET] ERROR: {e}")
        import traceback
        traceback.print_exc()

        return "", html.Div([
            html.Span("❌ Error: ", style={"fontWeight": "bold", "color": "#FF235A"}),
            html.Span(f"Excepción durante el procesamiento: {str(e)}"),
            html.Br(),
            html.Span("Revisa la consola para más detalles.", style={"fontSize": "0.85rem", "opacity": "0.8"})
        ], style={
            "color": "var(--text)",
            "padding": "0.5rem",
            "backgroundColor": "rgba(255, 35, 90, 0.1)",
            "borderRadius": "8px",
            "animation": "shake 0.5s ease-in-out"
        }), None

# =============================================================================
# Callback 4: Guardar dataset procesado seleccionado en store global
# =============================================================================
@callback(
    Output("selected-dataset", "data"),
    Input({"type": "dataset-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def save_selected_dataset(n_clicks_list):
    """Guarda dataset seleccionado en store global"""
    if not any(n_clicks_list):
        raise PreventUpdate
    return ctx.triggered_id.get("index")

# =============================================================================
# Callback 5: Redirigir a vista de dataset
# =============================================================================
@callback(
    Output("redirector", "pathname"),
    Input({"type": "dataset-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def redirect_to_dataset(n_clicks_list):
    """Redirige a la vista del dataset seleccionado"""
    if not any(n_clicks_list):
        raise PreventUpdate

    triggered = ctx.triggered_id
    nombre = triggered.get("index")
    print(f"👉 Dataset seleccionado: {nombre}")

    return "/dataset"
