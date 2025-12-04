from dash import html, dcc, register_page, callback, Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from app.components.PageContainer import get_page_container
from shared.fileUtils import get_data_folders
from backend.classes.dataset import Dataset
from shared.datasetUploader import upload_dataset_to_server
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

            # ===== Stores para manejo de baseline selection =====
            dcc.Store(id='detected-classes-store', data=None),
            dcc.Store(id='dataset-pending-upload-store', data=None),

            # Barra de acciones
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
                    html.Button(
                        "🧹 Limpiar Caché",
                        id="clear-cache-btn",
                        n_clicks=0,
                        style={**BUTTON_STYLE, "borderColor": "rgba(255, 107, 107, 0.6)", "backgroundColor": "rgba(255, 107, 107, 0.15)"},
                        title="Eliminar archivos de caché del pipeline para liberar espacio"
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

            # ===== Diálogo de selección de baseline =====
            html.Div(
                id='baseline-selection-dialog',
                style={'display': 'none', 'marginTop': '1.5rem'},
                children=[
                    html.Div(
                        style={
                            **PANEL_STYLE,
                            "maxWidth": "700px",
                            "background": "linear-gradient(180deg, rgba(0, 200, 160, 0.15), color-mix(in srgb, var(--color-2) 90%, transparent))",
                            "border": "2px solid rgba(0, 200, 160, 0.4)"
                        },
                        children=[
                            html.H3("⚙️ Configuración de Clases Baseline", style={
                                "color": "#00C8A0",
                                "marginBottom": "1rem",
                                "textAlign": "center"
                            }),
                            html.Div(id='detected-classes-message', style={
                                "marginBottom": "1rem",
                                "fontSize": "0.95rem",
                                "color": "var(--color-3)",
                                "textAlign": "center",
                                "lineHeight": "1.6"
                            }),
                            html.Div([
                                html.Label("¿Qué clases representan baseline/reposo (rest)?", style={
                                    "fontWeight": "600",
                                    "marginBottom": "0.5rem",
                                    "display": "block",
                                    "color": "var(--color-3)"
                                }),
                                html.Div("Puedes seleccionar múltiples clases (ej: Rest, Wait, WarmUp)", style={
                                    "fontSize": "0.85rem",
                                    "opacity": "0.8",
                                    "marginBottom": "0.5rem",
                                    "color": "var(--color-3)"
                                }),
                                dcc.Dropdown(
                                    id='baseline-class-dropdown',
                                    placeholder="Selecciona una o más clases (o deja vacío para generar rest del background)",
                                    style={'marginBottom': '1.5rem'},
                                    clearable=True,  # Permitir limpiar selecciones
                                    multi=True  # ← Permitir selección múltiple
                                ),
                            ]),
                            html.Div([
                                html.Label("Duración de ventana por evento (milisegundos):", style={
                                    "fontWeight": "600",
                                    "marginBottom": "0.5rem",
                                    "display": "block",
                                    "color": "var(--color-3)"
                                }),
                                html.Div("Recomendado: 800ms para P300/ERP, 3200ms para Inner Speech", style={
                                    "fontSize": "0.85rem",
                                    "opacity": "0.8",
                                    "marginBottom": "0.5rem",
                                    "color": "var(--color-3)"
                                }),
                                dcc.Input(
                                    id='duration-ms-input',
                                    type='number',
                                    value=3200,  # Default: mantiene comportamiento actual (3.2s)
                                    min=100,
                                    max=10000,
                                    step=100,
                                    placeholder="Duración en ms (ej: 800)",
                                    style={
                                        'width': '100%',
                                        'padding': '0.5rem',
                                        'marginBottom': '1.5rem',
                                        'borderRadius': '4px',
                                        'border': '1px solid rgba(255,255,255,0.2)'
                                    }
                                ),
                            ]),
                            html.Div([
                                html.Label("Porcentaje del dataset a subir a Azure (%):", style={
                                    "fontWeight": "600",
                                    "marginBottom": "0.5rem",
                                    "display": "block",
                                    "color": "var(--color-3)"
                                }),
                                html.Div("Solo aplica si Azure está habilitado. 0% = no subir, 100% = subir todos los archivos", style={
                                    "fontSize": "0.85rem",
                                    "opacity": "0.8",
                                    "marginBottom": "0.5rem",
                                    "color": "var(--color-3)"
                                }),
                                dcc.Input(
                                    id='azure-percentage-input',
                                    type='number',
                                    value=100,  # Default: subir todo
                                    min=0,  # 0 = no subir nada
                                    max=100,
                                    step=1,
                                    placeholder="Porcentaje (0-100)",
                                    style={
                                        'width': '100%',
                                        'padding': '0.5rem',
                                        'marginBottom': '1.5rem',
                                        'borderRadius': '4px',
                                        'border': '1px solid rgba(255,255,255,0.2)'
                                    }
                                ),
                            ]),
                            html.Div([
                                html.Button(
                                    "✓ Continuar con la carga",
                                    id='confirm-baseline-button',
                                    n_clicks=0,
                                    style={
                                        **BUTTON_STYLE,
                                        "backgroundColor": "#00C8A0",
                                        "borderColor": "#00C8A0",
                                        "width": "100%",
                                        "justifyContent": "center",
                                        "fontSize": "1rem"
                                    }
                                )
                            ], style={"textAlign": "center"})
                        ]
                    )
                ]
            ),

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
                        html.Li("Limpiar Caché: Elimina archivos intermedios del pipeline (filtros + transforms) para liberar espacio. Los datos originales NO se eliminan."),
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
# Callback 3: Detectar clases y mostrar diálogo de selección de baseline
# =============================================================================
@callback(
    [Output("baseline-selection-dialog", "style"),
     Output("baseline-class-dropdown", "options"),
     Output("baseline-class-dropdown", "value"),
     Output("detected-classes-message", "children"),
     Output("dataset-pending-upload-store", "data"),
     Output("processing-feedback", "children", allow_duplicate=True)],
    Input({"type": "pending-dataset-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def detect_classes_and_show_dialog(n_clicks_list):
    """Detecta clases del dataset y muestra diálogo de selección de baseline"""
    if not any(n_clicks_list):
        raise PreventUpdate

    triggered = ctx.triggered_id
    dataset_name = triggered.get("index")

    print(f"\n[DETECT] Detectando clases de dataset: {dataset_name}")

    # Construir ruta
    dataset_path = f"Data/{dataset_name}"

    # Detectar clases
    try:
        dataset = Dataset(dataset_path, dataset_name)
        detection_result = dataset.detect_classes_preview(dataset_path)

        if detection_result.get("status") != 200:
            error_msg = detection_result.get("message", "Error desconocido")
            return (
                {'display': 'none'},  # Ocultar diálogo
                [],  # Sin opciones
                None,  # Sin valor seleccionado
                "",  # Sin mensaje
                None,  # No guardar en store
                html.Div([
                    html.Span("❌ Error: ", style={"fontWeight": "bold", "color": "#FF235A"}),
                    html.Span(f"No se pudieron detectar clases: {error_msg}")
                ], style={
                    "color": "var(--text)",
                    "padding": "0.5rem",
                    "backgroundColor": "rgba(255, 35, 90, 0.1)",
                    "borderRadius": "8px",
                    "animation": "shake 0.5s ease-in-out"
                })
            )

        # Extraer información
        classes = detection_result.get("classes", [])
        dataset_type = detection_result.get("dataset_type", "generic")

        print(f"[DETECT] Detectadas {len(classes)} clases: {classes}")
        print(f"[DETECT] Tipo de dataset: {dataset_type}")

        # Preparar opciones del dropdown (sin "Ninguna", el vacío lo representa)
        options = [{"label": cls, "value": cls} for cls in classes]

        # Mensaje personalizado según tipo de dataset
        if dataset_type == "inner_speech":
            message = [
                html.Div(f"📊 Dataset de Inner Speech (Nieto) detectado", style={"fontWeight": "600", "marginBottom": "0.5rem"}),
                html.Div(f"Clases detectadas: {', '.join(classes)}"),
                html.Div("Este dataset típicamente NO tiene clase rest marcada en los datos.",
                         style={"fontSize": "0.9rem", "opacity": "0.85", "marginTop": "0.5rem"}),
                html.Div("💡 Deja vacío para generar rest automáticamente del background.",
                         style={"fontSize": "0.9rem", "opacity": "0.85", "fontStyle": "italic", "color": "#00C8A0"})
            ]
        else:
            message = [
                html.Div(f"📊 Dataset genérico detectado", style={"fontWeight": "600", "marginBottom": "0.5rem"}),
                html.Div(f"Clases detectadas: {', '.join(classes)}"),
                html.Div("Selecciona una o más clases que representen baseline/reposo (ej: Rest, Wait, WarmUp).",
                         style={"fontSize": "0.9rem", "opacity": "0.85", "marginTop": "0.5rem"}),
                html.Div("💡 Deja vacío para generar rest automáticamente del background.",
                         style={"fontSize": "0.9rem", "opacity": "0.85", "fontStyle": "italic", "color": "#00C8A0"})
            ]

        # Guardar información del dataset en store
        dataset_info = {
            "name": dataset_name,
            "path": dataset_path,
            "type": dataset_type,
            "classes": classes
        }

        return (
            {'display': 'block', 'marginTop': '1.5rem'},  # Mostrar diálogo
            options,  # Opciones del dropdown
            [],  # Valor por defecto: lista vacía (ninguna seleccionada)
            message,  # Mensaje explicativo
            dataset_info,  # Guardar en store
            ""  # Limpiar feedback
        )

    except Exception as e:
        print(f"[DETECT] ERROR: {e}")
        import traceback
        traceback.print_exc()

        return (
            {'display': 'none'},
            [],
            None,
            "",
            None,
            html.Div([
                html.Span("❌ Error: ", style={"fontWeight": "bold", "color": "#FF235A"}),
                html.Span(f"Excepción detectando clases: {str(e)}"),
                html.Br(),
                html.Span("Revisa la consola para más detalles.", style={"fontSize": "0.85rem", "opacity": "0.8"})
            ], style={
                "color": "var(--text)",
                "padding": "0.5rem",
                "backgroundColor": "rgba(255, 35, 90, 0.1)",
                "borderRadius": "8px",
                "animation": "shake 0.5s ease-in-out"
            })
        )

# =============================================================================
# Callback 4: Procesar dataset con baseline seleccionada
# =============================================================================
@callback(
    [Output("datasets-list", "children", allow_duplicate=True),
     Output("processing-feedback", "children", allow_duplicate=True),
     Output("baseline-selection-dialog", "style", allow_duplicate=True),
     Output("loading-output", "children", allow_duplicate=True)],
    Input("confirm-baseline-button", "n_clicks"),
    [State("baseline-class-dropdown", "value"),
     State("duration-ms-input", "value"),
     State("azure-percentage-input", "value"),
     State("dataset-pending-upload-store", "data")],
    prevent_initial_call=True
)
def process_dataset_with_baseline(n_clicks, baseline_classes, duration_ms, azure_percentage, dataset_info):
    """Procesa el dataset con las clases baseline seleccionadas"""
    if not n_clicks or not dataset_info:
        raise PreventUpdate

    dataset_name = dataset_info["name"]
    dataset_path = dataset_info["path"]

    # baseline_classes es una lista (puede estar vacía, tener 1 o múltiples elementos)
    # Si está vacía o es None, convertir a None para el backend
    if not baseline_classes or len(baseline_classes) == 0:
        baseline_classes_param = None
        print(f"\n[UPLOAD] Procesando dataset '{dataset_name}' sin baseline (generará rest del background)")
    else:
        baseline_classes_param = baseline_classes  # Lista de strings
        print(f"\n[UPLOAD] Procesando dataset '{dataset_name}' con baseline classes: {baseline_classes_param}")

    # Validar duration_ms (si está vacío, usar default de 3200ms)
    if not duration_ms or duration_ms < 100:
        duration_ms = 3200  # Default: mantiene comportamiento actual
        print(f"\n[UPLOAD] Usando duración por defecto: {duration_ms}ms (3.2s)")
    else:
        print(f"\n[UPLOAD] Duración de ventana configurada: {duration_ms}ms")

    # Validar azure_percentage (0 = no subir nada, 1-100 = porcentaje a subir)
    if azure_percentage is None or azure_percentage < 0 or azure_percentage > 100:
        azure_percentage = 100  # Default: subir todo
        print(f"\n[UPLOAD] Usando porcentaje por defecto para Azure: 100%")
    elif azure_percentage == 0:
        print(f"\n[UPLOAD] ⚠️ Porcentaje 0%: NO se subirá nada a Azure")
    else:
        print(f"\n[UPLOAD] Porcentaje para Azure configurado: {azure_percentage}%")

    try:
        # Procesar dataset con baseline y duración de ventana configurada
        dataset = Dataset(dataset_path, dataset_name)
        result = dataset.upload_dataset(
            dataset_path,
            baseline_classes=baseline_classes_param,
            duration_ms=duration_ms
        )

        if result.get("status") == 200:
            # ===== SUBIR ARCHIVOS AL SERVIDOR EXTERNO =====
            # Leer configuración de variables de entorno
            SERVER_URL = os.getenv("EXTERNAL_SERVER_URL", "https://xx8mx485.usw3.devtunnels.ms:8000/")
            ENABLE_UPLOAD = os.getenv("ENABLE_EXTERNAL_UPLOAD", "true").lower() == "true"

            if ENABLE_UPLOAD:
                print(f"\n[UPLOAD-SERVER] Iniciando subida de archivos al servidor externo...")
                print(f"[UPLOAD-SERVER] Servidor: {SERVER_URL}")
                print(f"[UPLOAD-SERVER] Porcentaje a subir: {azure_percentage}%")
                try:
                    upload_result = upload_dataset_to_server(
                        dataset_name=dataset_name,
                        server_url=SERVER_URL,
                        aux_folder="Aux",
                        description=f"Dataset procesado: {dataset_name}",
                        percentage=azure_percentage  # ← Pasar porcentaje
                    )

                    if upload_result["success"]:
                        print(f"[UPLOAD-SERVER] ✅ {upload_result['message']}")
                    else:
                        print(f"[UPLOAD-SERVER] ⚠️ {upload_result['message']}")
                        # No fallar el procesamiento si falla la subida al servidor
                        # Solo log el error

                except Exception as upload_error:
                    print(f"[UPLOAD-SERVER] ❌ Error subiendo al servidor: {upload_error}")
                    import traceback
                    traceback.print_exc()
                    # Continuar con el procesamiento normal aunque falle la subida
            else:
                print(f"[UPLOAD-SERVER] ⏭️ Subida al servidor externo deshabilitada (ENABLE_EXTERNAL_UPLOAD=false)")
            # ===== FIN SUBIDA AL SERVIDOR =====

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

            # Mensaje de éxito personalizado
            success_msg = [
                html.Div([
                    html.Span("✅ ", style={"fontSize": "1.5rem", "marginRight": "0.5rem"}),
                    html.Span("Éxito", style={"fontWeight": "bold", "color": "#38FF97", "fontSize": "1.2rem"})
                ], style={"marginBottom": "0.5rem"}),
                html.Span(f"Dataset '{dataset_name}' procesado correctamente."),
                html.Br(),
                html.Span(f"Archivos procesados: {num_files}", style={"fontSize": "0.85rem", "opacity": "0.8"}),
                html.Br()
            ]

            # Agregar info de baseline
            if baseline_classes_param:
                classes_list = ', '.join(baseline_classes_param)
                success_msg.extend([
                    html.Span(f"✓ Clases {classes_list} mapeadas a 'rest'", style={"fontSize": "0.9rem", "color": "#00C8A0"}),
                    html.Br(),
                    html.Span(f"✓ NO se generó 'rest' del background", style={"fontSize": "0.9rem", "color": "#00C8A0"}),
                    html.Br()
                ])
            else:
                success_msg.extend([
                    html.Span(f"✓ Clase 'rest' generada automáticamente del background", style={"fontSize": "0.9rem", "color": "#00C8A0"}),
                    html.Br()
                ])

            success_msg.extend([
                html.Span(f"Archivos .npy, Labels y Events generados en Aux/{dataset_name}/",
                         style={"fontSize": "0.85rem", "opacity": "0.8"}),
                html.Br(),
                html.Br(),
                html.Div([
                    html.Span("💡 ", style={"marginRight": "0.5rem"}),
                    html.Span("Ahora puedes hacer clic en 'Listar Datasets' para seleccionarlo.",
                             style={"fontStyle": "italic", "fontSize": "0.9rem"})
                ])
            ])

            return (
                pending_list,
                html.Div(success_msg, style={
                    "color": "var(--text)",
                    "padding": "1rem",
                    "backgroundColor": "rgba(56, 255, 151, 0.1)",
                    "borderRadius": "var(--radius-md)",
                    "border": "1px solid rgba(56, 255, 151, 0.3)",
                    "animation": "successPulse 0.6s ease-in-out"
                }),
                {'display': 'none'},  # Ocultar diálogo
                None
            )
        else:
            error_msg = result.get("message", "Error desconocido")
            return (
                "",
                html.Div([
                    html.Span("❌ Error: ", style={"fontWeight": "bold", "color": "#FF235A"}),
                    html.Span(f"No se pudo procesar el dataset. {error_msg}")
                ], style={
                    "color": "var(--text)",
                    "padding": "0.5rem",
                    "backgroundColor": "rgba(255, 35, 90, 0.1)",
                    "borderRadius": "8px",
                    "animation": "shake 0.5s ease-in-out"
                }),
                {'display': 'none'},  # Ocultar diálogo
                None
            )

    except Exception as e:
        print(f"[UPLOAD] ERROR: {e}")
        import traceback
        traceback.print_exc()

        return (
            "",
            html.Div([
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
            }),
            {'display': 'none'},  # Ocultar diálogo
            None
        )

# =============================================================================
# Callback 5: Guardar dataset procesado seleccionado en store global
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

# =============================================================================
# Callback 6: Limpiar caché del pipeline
# =============================================================================
@callback(
    Output("processing-feedback", "children", allow_duplicate=True),
    Output("loading-output", "children", allow_duplicate=True),
    Input("clear-cache-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_pipeline_cache(n_clicks):
    """
    Limpia el caché del pipeline (archivos intermedios de filtros + transforms).
    NO elimina datos originales ni subsets generados.
    """
    if not n_clicks:
        raise PreventUpdate

    try:
        from backend.classes.Experiment import Experiment

        print("\n🧹 [LIMPIEZA] Iniciando limpieza de caché del pipeline...")

        # Llamar a la función de limpieza
        result = Experiment.clear_pipeline_cache()

        files_deleted = result.get("files_deleted", 0)
        space_freed_mb = result.get("space_freed_mb", 0.0)
        experiments_affected = result.get("experiments_affected", [])

        print(f"✅ [LIMPIEZA] Eliminados {files_deleted} archivos")
        print(f"✅ [LIMPIEZA] Liberados {space_freed_mb:.2f} MB")
        print(f"✅ [LIMPIEZA] Experimentos afectados: {len(experiments_affected)}")

        # Mensaje de éxito
        feedback = html.Div([
            html.Div([
                html.Span("✅ Limpieza completada", style={"fontWeight": "bold", "color": "#00C8A0", "fontSize": "16px"}),
            ], className="mb-2"),
            html.Div([
                html.Div([
                    html.I(className="fas fa-trash-alt me-2", style={"fontSize": "12px"}),
                    html.Span(f"Archivos eliminados: {files_deleted:,}", style={"fontSize": "14px"})
                ], className="mb-1"),
                html.Div([
                    html.I(className="fas fa-hdd me-2", style={"fontSize": "12px"}),
                    html.Span(f"Espacio liberado: {space_freed_mb:.2f} MB", style={"fontSize": "14px"})
                ], className="mb-1"),
                html.Div([
                    html.I(className="fas fa-flask me-2", style={"fontSize": "12px"}),
                    html.Span(f"Experimentos afectados: {len(experiments_affected)}", style={"fontSize": "14px"})
                ])
            ])
        ], style={
            "color": "var(--text)",
            "padding": "1rem",
            "backgroundColor": "rgba(0, 200, 160, 0.15)",
            "border": "1px solid rgba(0, 200, 160, 0.3)",
            "borderRadius": "10px",
            "marginTop": "1rem",
            "textAlign": "left",
            "maxWidth": "500px",
            "margin": "1rem auto"
        })

        return feedback, None

    except Exception as e:
        import traceback
        print(f"❌ [LIMPIEZA] Error: {e}")
        traceback.print_exc()

        # Mensaje de error
        error_feedback = html.Div([
            html.Span("❌ Error: ", style={"fontWeight": "bold", "color": "#FF235A"}),
            html.Span(f"No se pudo limpiar el caché: {str(e)}"),
            html.Br(),
            html.Span("Revisa la consola para más detalles.", style={"fontSize": "0.85rem", "opacity": "0.8"})
        ], style={
            "color": "var(--text)",
            "padding": "0.5rem",
            "backgroundColor": "rgba(255, 35, 90, 0.1)",
            "borderRadius": "8px",
            "marginTop": "1rem"
        })

        return error_feedback, None
