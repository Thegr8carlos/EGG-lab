"""
Componente reutilizable para entrenamiento de modelos en la nube.
Incluye simulación de entrenamiento y visualización de métricas.
"""

from dash import html, dcc, callback, Input, Output, State, no_update, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import random
import time


def create_cloud_training_section(model_identifier: str) -> html.Div:
    """
    Crea la sección de entrenamiento en la nube con botón y visualización de métricas.

    Args:
        model_identifier: Identificador único del modelo (ej: "classic-SVM", "classic-RandomForest")

    Returns:
        Componente Dash con botón y área de métricas
    """
    return html.Div([
        # Store para validación del experimento
        dcc.Store(id={"type": "cloud-experiment-valid", "model": model_identifier}, data=False),

        # Store para estado del entrenamiento en la nube
        dcc.Store(id={"type": "cloud-training-status", "model": model_identifier}),

        # Store para métricas resultantes
        dcc.Store(id={"type": "cloud-training-metrics", "model": model_identifier}),

        # Interval para simular progreso
        dcc.Interval(
            id={"type": "cloud-training-interval", "model": model_identifier},
            interval=1000,  # 1 segundo
            disabled=True
        ),

        # Botón de entrenamiento en la nube
        html.Div([
            dbc.Button(
                [
                    html.I(className="fas fa-cloud me-2"),
                    "Entrenar Modelo en la Nube"
                ],
                id={"type": "btn-cloud-training", "model": model_identifier},
                color="success",
                size="lg",
                disabled=True,
                className="w-100",
                style={
                    "fontSize": "16px",
                    "fontWeight": "600",
                    "height": "50px",
                    "marginTop": "20px",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"
                }
            ),
            html.Small(
                "El botón se activará después de validar la configuración del modelo",
                id={"type": "cloud-training-hint", "model": model_identifier},
                className="text-muted mt-2 d-block text-center",
                style={"fontSize": "11px"}
            )
        ], className="mb-4"),

        # Área de estado de entrenamiento
        html.Div(
            id={"type": "cloud-training-status-display", "model": model_identifier},
            className="mt-3"
        ),

        # Área de visualización de métricas
        dcc.Loading(
            id={"type": "cloud-metrics-loading", "model": model_identifier},
            type="circle",
            fullscreen=False,
            color="#28a745",
            children=[
                html.Div(
                    id={"type": "cloud-metrics-display", "model": model_identifier},
                    className="mt-4"
                )
            ]
        )
    ], className="cloud-training-section")


def render_metrics_visualization(metrics: dict) -> html.Div:
    """
    Renderiza las métricas de evaluación de manera visual.

    Args:
        metrics: Diccionario con métricas del modelo (estructura de EvaluationMetrics)

    Returns:
        Componente Dash con visualización de métricas
    """
    if not metrics:
        return html.Div()

    # Extraer métricas principales
    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    f1_score = metrics.get("f1_score", 0)
    auc_roc = metrics.get("auc_roc", 0)
    confusion_matrix = metrics.get("confusion_matrix", [])
    loss_values = metrics.get("loss", [])
    eval_time = metrics.get("evaluation_time", "N/A")

    # Crear tarjetas de métricas principales
    metric_cards = html.Div([
        dbc.Row([
            dbc.Col([
                _create_metric_card("Accuracy", accuracy, "fas fa-chart-line", "#4A90E2")
            ], width=12, lg=4, className="mb-3"),
            dbc.Col([
                _create_metric_card("Precision", precision, "fas fa-crosshairs", "#F5A623")
            ], width=12, lg=4, className="mb-3"),
            dbc.Col([
                _create_metric_card("Recall", recall, "fas fa-search", "#50E3C2")
            ], width=12, lg=4, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col([
                _create_metric_card("F1-Score", f1_score, "fas fa-balance-scale", "#7ED321")
            ], width=12, lg=6, className="mb-3"),
            dbc.Col([
                _create_metric_card("AUC-ROC", auc_roc, "fas fa-chart-area", "#BD10E0")
            ], width=12, lg=6, className="mb-3"),
        ])
    ], className="mb-4")

    # Crear gráfico de matriz de confusión
    confusion_matrix_plot = _create_confusion_matrix_plot(confusion_matrix)

    # Crear gráfico de pérdida
    loss_plot = _create_loss_plot(loss_values)

    return html.Div([
        html.H4([
            html.I(className="fas fa-chart-bar me-2"),
            "Resultados del Entrenamiento"
        ], className="text-center mb-4", style={"color": "#28a745"}),

        # Tiempo de evaluación
        html.Div([
            html.I(className="fas fa-clock me-2"),
            html.Strong("Tiempo de evaluación: "),
            html.Span(eval_time)
        ], className="text-center mb-4", style={"color": "rgba(255,255,255,0.8)", "fontSize": "14px"}),

        # Métricas principales
        metric_cards,

        # Gráficos
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Matriz de Confusión", className="text-center mb-3", style={"color": "white"}),
                    dcc.Graph(
                        figure=confusion_matrix_plot,
                        config={'displayModeBar': False},
                        style={"height": "400px"}
                    )
                ], className="p-3", style={
                    "backgroundColor": "rgba(0,0,0,0.3)",
                    "borderRadius": "8px",
                    "border": "1px solid rgba(255,255,255,0.1)"
                })
            ], width=12, lg=6, className="mb-3"),

            dbc.Col([
                html.Div([
                    html.H5("Pérdida durante Entrenamiento", className="text-center mb-3", style={"color": "white"}),
                    dcc.Graph(
                        figure=loss_plot,
                        config={'displayModeBar': False},
                        style={"height": "400px"}
                    )
                ], className="p-3", style={
                    "backgroundColor": "rgba(0,0,0,0.3)",
                    "borderRadius": "8px",
                    "border": "1px solid rgba(255,255,255,0.1)"
                })
            ], width=12, lg=6, className="mb-3"),
        ])
    ], className="metrics-visualization", style={
        "backgroundColor": "rgba(0,0,0,0.2)",
        "padding": "20px",
        "borderRadius": "12px",
        "border": "1px solid rgba(255,255,255,0.1)"
    })


def _create_metric_card(label: str, value: float, icon: str, color: str) -> dbc.Card:
    """Crea una tarjeta individual para una métrica."""
    percentage = f"{value * 100:.2f}%"

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"{icon} fa-2x mb-2", style={"color": color}),
                html.H6(label, className="mb-2", style={"color": "rgba(255,255,255,0.8)", "fontSize": "13px"}),
                html.H3(percentage, className="mb-0", style={"color": "white", "fontWeight": "700"})
            ], className="text-center")
        ], style={"padding": "15px"})
    ], style={
        "backgroundColor": "rgba(0,0,0,0.4)",
        "border": f"1px solid {color}",
        "borderRadius": "8px",
        "boxShadow": f"0 4px 8px {color}33"
    })


def _create_confusion_matrix_plot(matrix: list) -> go.Figure:
    """Crea un heatmap de la matriz de confusión."""
    if not matrix or not isinstance(matrix, list):
        matrix = [[0]]

    # Generar etiquetas de clase
    n_classes = len(matrix)
    class_labels = [f"Clase {i}" for i in range(n_classes)]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=class_labels,
        y=class_labels,
        colorscale='Blues',
        text=matrix,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        hovertemplate='Verdadero: %{y}<br>Predicho: %{x}<br>Valor: %{z}<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Predicción",
        yaxis_title="Etiqueta Verdadera",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=60, r=20, t=20, b=60),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )

    return fig


def _create_loss_plot(loss_values: list) -> go.Figure:
    """Crea un gráfico de línea para la pérdida."""
    if not loss_values or not isinstance(loss_values, list):
        loss_values = []

    epochs = list(range(1, len(loss_values) + 1)) if loss_values else [1]
    loss_vals = loss_values if loss_values else [0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=loss_vals,
        mode='lines+markers',
        name='Loss',
        line=dict(color='#F5A623', width=3),
        marker=dict(size=8, color='#F5A623', line=dict(width=2, color='white')),
        hovertemplate='Época: %{x}<br>Pérdida: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Época",
        yaxis_title="Pérdida",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=60, r=20, t=20, b=60),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        hovermode='x unified'
    )

    return fig


def _generate_simulated_metrics() -> dict:
    """
    Genera métricas simuladas siguiendo la estructura de EvaluationMetrics.

    Returns:
        Diccionario con métricas simuladas
    """
    # Generar métricas principales
    accuracy = random.uniform(0.80, 0.95)
    precision = random.uniform(0.75, 0.93)
    recall = random.uniform(0.78, 0.92)
    f1_score = 2 * (precision * recall) / (precision + recall)
    auc_roc = random.uniform(0.85, 0.98)

    # Generar matriz de confusión (ejemplo 2x2 para clasificación binaria)
    # Pero puede ser más grande
    n_classes = random.choice([2, 3, 4])
    confusion_matrix = []
    for i in range(n_classes):
        row = []
        for j in range(n_classes):
            if i == j:
                # Diagonal (clasificaciones correctas)
                row.append(random.randint(80, 120))
            else:
                # Fuera de diagonal (errores)
                row.append(random.randint(5, 25))
        confusion_matrix.append(row)

    # Generar valores de pérdida (simulando épocas de entrenamiento)
    n_epochs = random.randint(15, 30)
    initial_loss = random.uniform(0.8, 1.2)
    loss_values = [initial_loss]

    for _ in range(1, n_epochs):
        # Pérdida decrece con algo de ruido
        new_loss = loss_values[-1] * random.uniform(0.85, 0.98)
        loss_values.append(max(0.1, new_loss))  # Mínimo de 0.1

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "auc_roc": auc_roc,
        "confusion_matrix": confusion_matrix,
        "loss": loss_values,
        "evaluation_time": ""  # Se llenará en el callback
    }


# ============================================
# CALLBACKS DE SIMULACIÓN (usando MATCH pattern)
# ============================================


@callback(
    Output({"type": "cloud-training-status", "model": MATCH}, "data", allow_duplicate=True),
    Output({"type": "cloud-training-interval", "model": MATCH}, "disabled", allow_duplicate=True),
    Output({"type": "cloud-training-status-display", "model": MATCH}, "children", allow_duplicate=True),
    Output({"type": "btn-cloud-training", "model": MATCH}, "disabled", allow_duplicate=True),
    Input({"type": "btn-cloud-training", "model": MATCH}, "n_clicks"),
    prevent_initial_call=True
)
def start_cloud_training(n_clicks):
    """
    Inicia la simulación de entrenamiento: reinicia progreso, habilita el intervalo y deshabilita el botón.
    """
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    status = {"running": True, "progress": 0, "started_at": time.time()}
    status_ui = dbc.Alert(
        [
            html.I(className="fas fa-cloud me-2"),
            html.Strong("Entrenando en la nube..."),
            dbc.Progress(value=0, striped=True, animated=True, className="mt-2", style={"height": "16px"})
        ],
        color="info",
        className="mb-0"
    )
    return status, False, status_ui, True


@callback(
    Output({"type": "cloud-training-status", "model": MATCH}, "data", allow_duplicate=True),
    Output({"type": "cloud-training-interval", "model": MATCH}, "disabled", allow_duplicate=True),
    Output({"type": "cloud-training-metrics", "model": MATCH}, "data", allow_duplicate=True),
    Output({"type": "cloud-metrics-display", "model": MATCH}, "children", allow_duplicate=True),
    Output({"type": "cloud-training-status-display", "model": MATCH}, "children", allow_duplicate=True),
    Output({"type": "btn-cloud-training", "model": MATCH}, "disabled", allow_duplicate=True),
    Input({"type": "cloud-training-interval", "model": MATCH}, "n_intervals"),
    State({"type": "cloud-training-status", "model": MATCH}, "data"),
    prevent_initial_call=True
)
def tick_cloud_training(n_intervals, status_data):
    """
    Avanza el progreso cada tick. Al completar, genera métricas simuladas y las muestra.
    """
    if not status_data or not status_data.get("running"):
        # Nada que hacer si no hay entrenamiento activo
        return no_update, True, no_update, no_update, no_update, no_update

    progress = status_data.get("progress", 0)
    # Incremento de 12-25% por tick
    progress += random.randint(12, 25)
    progress = min(progress, 100)
    status_data["progress"] = progress

    if progress >= 100:
        # Finalizar simulación
        metrics = _generate_simulated_metrics()
        # Tiempo de evaluación simulado
        elapsed = max(1, int(time.time() - status_data.get("started_at", time.time())))
        metrics["evaluation_time"] = f"{elapsed}s"

        metrics_ui = render_metrics_visualization(metrics)
        status_ui = dbc.Alert(
            [
                html.I(className="fas fa-check-circle me-2"),
                html.Strong("Entrenamiento completado"),
                html.Span("  - listo para revisar métricas", className="ms-2")
            ],
            color="success",
            className="mb-0"
        )

        # Marcar como no corriendo para permitir re-ejecutar
        status_data["running"] = False
        return status_data, True, metrics, metrics_ui, status_ui, False

    # Aún en progreso
    status_ui = dbc.Alert(
        [
            html.I(className="fas fa-cloud me-2"),
            html.Strong("Entrenando en la nube..."),
            dbc.Progress(value=progress, striped=True, animated=True, className="mt-2", style={"height": "16px"})
        ],
        color="info",
        className="mb-0"
    )
    return status_data, False, no_update, no_update, status_ui, True
