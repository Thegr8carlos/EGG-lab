# ======================================================================
# CONFIGURACIÓN GPU - DEBE SER LO PRIMERO
# ======================================================================
# IMPORTANTE: Configurar GPU ANTES de cualquier import que use TensorFlow
# Esto evita fragmentación de memoria y permite entrenamientos secuenciales
from backend.gpu_config import configure_gpu_for_training
_gpu_available = configure_gpu_for_training()
# ======================================================================

# ui dependencys
from dash import Dash, dcc, html, page_container, page_registry,Input, Output, State, ALL
from dash import dcc, html, callback
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc  
import json

# internal dependencies

# ui components  
from app.components.NavBar import get_navBar
from app.components.Header import get_header

from app.components.RigthComlumn import get_rightColumn



# backend dependencies 
from backend.classes.dataset import Dataset
from app.components.SideBar import get_sideBar



from backend.classes.Experiment import Experiment



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

# Allow pages to register callbacks for components that are not present
# in the global layout (we'll let pages render `sidebar-wrapper` themselves).
app.config.suppress_callback_exceptions = True


#Adding a dcc.Store for a client side memory store 
dcc.Store(id='selected-file-path', storage_type='local')  # or 'local' if you want it to persist longer



navBar = get_navBar(page_registry)
header = get_header(page_registry)

app.layout = html.Div(
    id="app-container",
    children=[
        #header,
        navBar,
        html.Div(id="main-content-wrapper", children=page_container),
        dcc.Location(id="url"),
        dcc.Store(id="selected-file-path", storage_type="local"),
        dcc.Store(id="selected-dataset", storage_type="local"),
        # Modales de confirmación
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Confirmar Nuevo Experimento")),
            dbc.ModalBody([
                html.P("¿Estás seguro que deseas crear un nuevo experimento?"),
                html.P("Esto iniciará una nueva sesión con un ID diferente.", className="text-muted"),
                html.P("El experimento actual y su cache permanecerán intactos.", className="text-muted small"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancelar", id="modal-new-exp-cancel", color="secondary", className="me-2"),
                dbc.Button("Crear Nuevo", id="modal-new-exp-confirm", color="success"),
            ]),
        ], id="modal-new-experiment", is_open=False, centered=True),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Confirmar Limpieza de Cache")),
            dbc.ModalBody([
                html.P("¿Estás seguro que deseas limpiar el cache de experimentos antiguos?"),
                html.P("Se eliminarán todos los archivos procesados de experimentos previos.", className="text-warning"),
                html.P("Esta acción NO afectará el experimento actual.", className="text-muted small"),
                html.Div(id="cache-stats-preview", children="Calculando espacio a liberar...", className="mt-3"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancelar", id="modal-clear-cache-cancel", color="secondary", className="me-2"),
                dbc.Button("Limpiar Cache", id="modal-clear-cache-confirm", color="warning"),
            ]),
        ], id="modal-clear-cache", is_open=False, centered=True),
        # Toasts para notificaciones
        dbc.Toast(
            id="toast-notification",
            header="Notificación",
            is_open=False,
            dismissable=True,
            duration=4000,
            style={"position": "fixed", "top": 80, "right": 10, "width": 350, "zIndex": 9999},
        ),
    ],
)


# Sidebar visibility is now controlled by pages which render the `sidebar-wrapper`.
# The old toggle callback (gif button) was removed so the sidebar stays visible.
    

#Listener for list elements     
@app.callback(
    Output('selected-file-path','data'),  
    Output('url','pathname'),
    Input({'type': 'file-item', 'path': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def on_file_click(n_clicks_list):
    import json
    from dash import callback_context, no_update

    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = json.loads(ctx.triggered[0]['prop_id'].split(".n_clicks")[0])
    file_path = triggered_id['path']
    print(f"🧐 Selected file: {file_path}")
    # ✅ Actualiza el store, pero NO cambies de ruta
    return file_path, no_update


# ============== Callbacks para gestión de experimentos ==============

# 1. Actualizar ID del experimento en navbar
@app.callback(
    Output('navbar-experiment-id', 'children'),
    Input('url', 'pathname')
)
def update_experiment_id(pathname):
    """Actualiza el ID del experimento actual en el navbar."""
    try:
        exp_id = Experiment._get_last_experiment_id()
        return exp_id
    except:
        return "--"


# 2. Abrir modal de nuevo experimento
@app.callback(
    Output('modal-new-experiment', 'is_open'),
    Input('btn-new-experiment', 'n_clicks'),
    Input('modal-new-exp-cancel', 'n_clicks'),
    Input('modal-new-exp-confirm', 'n_clicks'),
    State('modal-new-experiment', 'is_open'),
    prevent_initial_call=True
)
def toggle_new_experiment_modal(btn_new, btn_cancel, btn_confirm, is_open):
    """Controla apertura/cierre del modal de nuevo experimento."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'btn-new-experiment':
        return True
    elif trigger_id in ['modal-new-exp-cancel', 'modal-new-exp-confirm']:
        return False

    return is_open


# 3. Confirmar creación de nuevo experimento
@app.callback(
    Output('toast-notification', 'is_open', allow_duplicate=True),
    Output('toast-notification', 'children', allow_duplicate=True),
    Output('toast-notification', 'header', allow_duplicate=True),
    Output('toast-notification', 'icon', allow_duplicate=True),
    Output('navbar-experiment-id', 'children', allow_duplicate=True),
    Input('modal-new-exp-confirm', 'n_clicks'),
    prevent_initial_call=True
)
def create_new_experiment(n_clicks):
    """Crea un nuevo experimento al confirmar."""
    if not n_clicks:
        raise PreventUpdate

    try:
        new_id = Experiment.reset_experiment()
        return True, f"Nuevo experimento {new_id} creado exitosamente", "Éxito", "success", new_id
    except Exception as e:
        return True, f"Error al crear experimento: {str(e)}", "Error", "danger", no_update


# 4. Abrir modal de limpiar cache
@app.callback(
    Output('modal-clear-cache', 'is_open'),
    Input('btn-clear-cache', 'n_clicks'),
    Input('modal-clear-cache-cancel', 'n_clicks'),
    Input('modal-clear-cache-confirm', 'n_clicks'),
    State('modal-clear-cache', 'is_open'),
    prevent_initial_call=True
)
def toggle_clear_cache_modal(btn_clear, btn_cancel, btn_confirm, is_open):
    """Controla apertura/cierre del modal de limpiar cache."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'btn-clear-cache':
        return True
    elif trigger_id in ['modal-clear-cache-cancel', 'modal-clear-cache-confirm']:
        return False

    return is_open


# 5. Confirmar limpieza de cache
@app.callback(
    Output('toast-notification', 'is_open', allow_duplicate=True),
    Output('toast-notification', 'children', allow_duplicate=True),
    Output('toast-notification', 'header', allow_duplicate=True),
    Output('toast-notification', 'icon', allow_duplicate=True),
    Input('modal-clear-cache-confirm', 'n_clicks'),
    prevent_initial_call=True
)
def clear_old_caches(n_clicks):
    """Limpia cache de experimentos antiguos al confirmar."""
    if not n_clicks:
        raise PreventUpdate

    try:
        stats = Experiment.clear_old_caches(keep_current=True)

        if stats['experiments_cleaned'] == 0:
            return True, "No hay cache antiguo para limpiar", "Información", "info"

        message = html.Div([
            html.P(f"✅ {stats['experiments_cleaned']} experimentos limpiados"),
            html.P(f"📁 {stats['files_deleted']} archivos eliminados"),
            html.P(f"💾 {stats['space_freed_mb']:.2f} MB liberados"),
        ])

        if stats['errors']:
            message.children.append(html.P(f"⚠️ {len(stats['errors'])} errores", className="text-warning"))

        return True, message, "Cache Limpiado", "success"
    except Exception as e:
        return True, f"Error al limpiar cache: {str(e)}", "Error", "danger"









    
if __name__ == "__main__":

    # We create a new experiment
    try:
        Experiment.create_blank_json()
    except Exception as e:
        print(f"⚠️ Error al crear experimento: {e}")
    
    
    
    #app.run(debug=True) # comment this line if u want to test backend functionality 
    app.run(debug=True, use_reloader=False, port=8091, host="127.0.0.1")
    #print("🧐🔎🛠️💻  Backend Debug") # entry point to backend debug 
    #data = Dataset("path", "name")
    #response = data.upload_dataset("Data/nieto_inner_speech")
    #print(response)
    
