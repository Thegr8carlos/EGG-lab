from dash import html, dcc, register_page, callback, Input, Output, ALL, ctx, clientside_callback
from dash.exceptions import PreventUpdate
from app.components.PageContainer import get_page_container
from shared.fileUtils import get_data_folders
from backend.classes.dataset import Dataset

register_page(__name__, path="/cargardatos", name="Cargar Datos")

# Estilo común para botones (paleta sobria para app científica)
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

layout = get_page_container(
    "Carga y gestión de datos EEG",
    "Carga datasets EEG (actualmente .bdf y .edf) o elige uno ya cargado para trabajar.",
    html.Div(
        style=PANEL_STYLE,
        children=[
            # Mantén el Location para redirección
            dcc.Location(id="redirector", refresh=True),

            # Barra de acciones primaria
            html.Div(
                style=TOOLBAR_STYLE,
                children=[
                    # ⬅️ MISMO ID, sólo cambia la etiqueta visible
                    html.Button(
                        "Datasets cargados",
                        id="cargar-btn",
                        n_clicks=0,
                        style={**BUTTON_STYLE, "borderColor": "var(--color-4)"},
                        title="Mostrar los datasets disponibles que ya están en el sistema"
                    ),

                    # ➕ Nuevo botón (no altera callbacks existentes)
                    html.Button(
                        "Cargar dataset",
                        id="upload-btn",
                        n_clicks=0,
                        style={**BUTTON_STYLE, "borderColor": "var(--color-5)"},
                        title="Importar un nuevo dataset al sistema (pendiente de lógica)"
                    ),

                    # 📁 Botón para subir carpeta completa
                    html.Button(
                        "📁 Subir carpeta",
                        id="upload-folder-btn",
                        n_clicks=0,
                        style={**BUTTON_STYLE, "borderColor": "#00C8A0", "backgroundColor": "#00C8A0"},
                        title="Seleccionar y subir una carpeta completa con estructura"
                    ),
                ],
            ),

            # Contenedor para el input de carpeta (será inicializado por JS)
            html.Div(id="folder-upload-container", style={"marginTop": "1rem", "textAlign": "center"}),
                        # Texto de ayuda (no afecta funcionalidad)
            html.Div(
                children=[
                    html.Div("¿Qué puedes hacer aquí?", style={
                        "fontWeight": 700, "letterSpacing": "0.2px",
                        "margin": "0.2rem 0 0.4rem 0", "color": "var(--color-3)"
                    }),
                    html.Ul([
                        html.Li("Importar datasets EEG — por ahora se admiten archivos .bdf y .edf."),
                        html.Li("Ver la lista de datasets ya cargados en el sistema."),
                        html.Li("Seleccionar un dataset para continuar con su exploración/visualización."),
                    ], style={"margin": "0 0 0.75rem 1.25rem", "color": "var(--color-3)", "opacity": 0.9}),
                ],
                style={"textAlign": "left"}
            ),

            html.Div(
                "Selecciona un dataset ya cargado para continuar con el análisis.",
                style=HELP_STYLE
            ),

            # Contenedor donde se listan los datasets (no se cambia el id)
            html.Div(id="lista-opciones", style={"marginTop": "0.5rem", "textAlign": "center"}),
        ]
    ),
)

@callback(
    Output("lista-opciones", "children"),
    Input("cargar-btn", "n_clicks")
)
def mostrar_opciones(n):
    if n == 0:
        return ""
    datasets = get_data_folders()
    return html.Div([
        html.Button(
            nombre,
            id={"type": "dataset-btn", "index": nombre},
            n_clicks=0,
            style=DATASET_BUTTON_STYLE
        )
        for nombre in datasets
    ], style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap"})

# ✅ Escribe en el Store GLOBAL declarado en main.py (storage_type="local")
@callback(
    Output("selected-dataset", "data"),
    Input({"type": "dataset-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def guardar_dataset_seleccionado(n_clicks_list):
    if not any(n_clicks_list):
        raise PreventUpdate
    return ctx.triggered_id.get("index")

@callback(
    Output("redirector", "pathname"),
    Input({"type": "dataset-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def redirigir_dataset(n_clicks_list):
    if not any(n_clicks_list):
        raise PreventUpdate
    triggered = ctx.triggered_id
    nombre = triggered.get("index")
    print(f"👉 Dataset seleccionado: {nombre}")

    dataset = Dataset(f"Data/{nombre}", nombre)
    dataset.upload_dataset(dataset.path)  # misma lógica actual

    return "/dataset"


# Clientside callback para inicializar el botón de subir carpeta
clientside_callback(
    """
    function(n_clicks) {
        function initFolderUpload() {
            const container = document.getElementById("folder-upload-container");
            if (!container) return false;

            // Prevent double initialization
            if (container.dataset.initialized) return true;
            container.dataset.initialized = true;

            // Create hidden file input
            const input = document.createElement("input");
            input.type = "file";
            input.multiple = true;
            input.webkitdirectory = true;
            input.style.display = "none";
            container.appendChild(input);

            // Create visible styled button
            const btn = document.createElement("button");
            btn.innerText = "📂 Seleccionar carpeta";
            btn.style.padding = "0.65rem 1.1rem";
            btn.style.backgroundColor = "#00C8A0";
            btn.style.color = "white";
            btn.style.border = "none";
            btn.style.borderRadius = "10px";
            btn.style.cursor = "pointer";
            btn.style.fontWeight = "600";
            btn.style.boxShadow = "0 6px 14px rgba(0, 200, 160, 0.4)";
            btn.style.transition = "transform 120ms ease";
            btn.onmouseover = () => btn.style.transform = "scale(1.05)";
            btn.onmouseout = () => btn.style.transform = "scale(1)";
            container.appendChild(btn);

            // Trigger file input when button is clicked
            btn.addEventListener("click", () => input.click());

            // Handle folder selection and upload files individually
            input.addEventListener("change", async function () {
                const totalFiles = input.files.length;
                console.log(`Iniciando carga de ${totalFiles} archivos...`);

                let uploaded = 0;
                let failed = 0;

                for (const file of input.files) {
                    const formData = new FormData();
                    formData.append("files", file);
                    formData.append("paths", file.webkitRelativePath);

                    try {
                        const resp = await fetch(
                            "https://l2c9dqkn.usw3.devtunnels.ms:8000/upload-folder",
                            { method: "POST", body: formData }
                        );

                        if (!resp.ok) {
                            console.error(`❌ Failed: ${file.name}`);
                            failed++;
                        } else {
                            console.log(`✅ Uploaded: ${file.webkitRelativePath}`);
                            uploaded++;
                        }
                    } catch (err) {
                        console.error(`❌ Upload error for ${file.name}:`, err);
                        failed++;
                    }
                }

                alert(`Proceso completado!\\n✅ Exitosos: ${uploaded}\\n❌ Fallidos: ${failed}\\n\\nRevisa la consola para detalles.`);
            });

            return true;
        }

        // Initialize on button click
        if (n_clicks > 0) {
            const interval = setInterval(() => {
                if (initFolderUpload()) clearInterval(interval);
            }, 100);
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output("folder-upload-container", "children"),
    Input("upload-folder-btn", "n_clicks"),
    prevent_initial_call=True
)
