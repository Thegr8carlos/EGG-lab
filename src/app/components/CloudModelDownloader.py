"""
Componente para descargar modelos entrenados en la nube usando job_id
"""

from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from shared.experimentUploader import download_trained_model


def create_cloud_downloader_card():
    """
    Crea una tarjeta para descargar modelos desde la nube usando job_id.

    Returns:
        html.Div: Card con input para job_id y botón de descarga
    """
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-cloud-download-alt me-2"),
            html.Strong("Descargar Modelo desde la Nube")
        ], className="bg-primary text-white"),

        dbc.CardBody([
            html.P([
                "Si tienes un ",
                html.Code("job_id"),
                " de un modelo entrenado en la nube, ingresalo aquí para descargarlo:"
            ], className="mb-3", style={"fontSize": "14px"}),

            # Input para job_id
            dbc.Row([
                dbc.Col([
                    dbc.Label("Job ID:", className="fw-bold", style={"fontSize": "13px"}),
                    dbc.Input(
                        id="cloud-job-id-input",
                        placeholder="Ej: upbeat_eye_ns0ylw40mv",
                        type="text",
                        className="mb-2"
                    )
                ], md=8),

                dbc.Col([
                    dbc.Label("Tipo:", className="fw-bold", style={"fontSize": "13px"}),
                    dbc.Select(
                        id="cloud-model-type-select",
                        options=[
                            {"label": "Inner Speech", "value": "InnerSpeech"},
                            {"label": "P300", "value": "P300"}
                        ],
                        value="InnerSpeech",
                        className="mb-2"
                    )
                ], md=4)
            ]),

            # Botón de descarga
            dbc.Button(
                [
                    html.I(className="fas fa-download me-2"),
                    "Descargar Modelo"
                ],
                id="btn-download-cloud-model",
                color="success",
                className="w-100 mt-2",
                n_clicks=0
            ),

            # Loading spinner
            dbc.Spinner(
                html.Div(id="cloud-download-result", className="mt-3"),
                color="primary",
                size="sm",
                spinner_style={"display": "none"}
            )
        ])
    ], className="shadow-sm mb-4")


@callback(
    Output("cloud-download-result", "children"),
    Input("btn-download-cloud-model", "n_clicks"),
    State("cloud-job-id-input", "value"),
    State("cloud-model-type-select", "value"),
    prevent_initial_call=True
)
def download_cloud_model_callback(n_clicks, job_id, classifier_type):
    """
    Callback para descargar modelo desde la nube.
    """
    if not n_clicks or not job_id:
        return dbc.Alert(
            "Por favor ingresa un Job ID válido",
            color="warning",
            dismissable=True
        )

    # Limpiar job_id (quitar espacios)
    job_id = job_id.strip()

    if not job_id:
        return dbc.Alert(
            "El Job ID no puede estar vacío",
            color="warning",
            dismissable=True
        )

    # Mostrar mensaje de inicio
    print(f"\n{'='*60}")
    print(f"[CloudDownloader] Descargando modelo con Job ID: {job_id}")
    print(f"[CloudDownloader] Tipo: {classifier_type}")
    print(f"{'='*60}\n")

    # Intentar descargar
    try:
        result = download_trained_model(
            job_id=job_id,
            classifier_type=classifier_type
        )

        if result["success"]:
            metrics = result.get("metrics", {})

            return dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                html.Div([
                    html.Strong("✅ Modelo descargado exitosamente"),
                    html.Br(),
                    html.Hr(className="my-2"),
                    html.Div([
                        html.I(className="fas fa-folder me-1", style={"fontSize": "12px"}),
                        html.Small(f"Ruta: {result['model_path']}", className="text-muted")
                    ], className="mb-1"),
                    html.Div([
                        html.I(className="fas fa-chart-line me-1", style={"fontSize": "12px"}),
                        html.Small(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}" if isinstance(metrics.get('accuracy'), (int, float)) else "N/A", className="text-muted")
                    ], className="mb-1"),
                    html.Div([
                        html.I(className="fas fa-chart-bar me-1", style={"fontSize": "12px"}),
                        html.Small(f"F1-Score: {metrics.get('f1_score', 'N/A'):.4f}" if isinstance(metrics.get('f1_score'), (int, float)) else "N/A", className="text-muted")
                    ])
                ])
            ], color="success", dismissable=True, duration=10000)

        else:
            error_msg = result.get("error", "Error desconocido")
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Div([
                    html.Strong("❌ Error al descargar el modelo"),
                    html.Br(),
                    html.Hr(className="my-2"),
                    html.Small(error_msg, className="text-muted")
                ])
            ], color="danger", dismissable=True)

    except Exception as e:
        import traceback
        traceback.print_exc()

        return dbc.Alert([
            html.I(className="fas fa-times-circle me-2"),
            html.Div([
                html.Strong("❌ Error inesperado"),
                html.Br(),
                html.Hr(className="my-2"),
                html.Small(str(e), className="text-muted")
            ])
        ], color="danger", dismissable=True)
