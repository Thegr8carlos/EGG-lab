"""
Editor matricial interactivo para definir kernels de capas convolucionales.
Permite crear múltiples filtros con matrices editables.
"""

from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, no_update, ctx
import dash_bootstrap_components as dbc
from typing import List, Dict, Any
import json


def create_kernel_matrix_editor(filter_index: int, kernel_index: int, size: tuple = (3, 3), values: List[List[float]] = None) -> html.Div:
    """
    Crea un editor de matriz para un kernel específico.

    Args:
        filter_index: Índice del filtro
        kernel_index: Índice del kernel dentro del filtro (0=R, 1=G, 2=B)
        size: Tamaño del kernel (filas, columnas)
        values: Valores iniciales de la matriz
    """
    rows, cols = size

    if values is None:
        # Inicializar con valores aleatorios pequeños
        values = [[0.0 for _ in range(cols)] for _ in range(rows)]

    kernel_names = ["R (Rojo)", "G (Verde)", "B (Azul)"]

    cells = []
    for i in range(rows):
        row_cells = []
        for j in range(cols):
            cell = dbc.Input(
                id={
                    "type": "kernel-cell",
                    "filter": filter_index,
                    "kernel": kernel_index,
                    "row": i,
                    "col": j
                },
                type="number",
                value=values[i][j],
                step=0.01,
                style={
                    "width": "60px",
                    "height": "40px",
                    "textAlign": "center",
                    "fontSize": "12px",
                    "padding": "5px"
                }
            )
            row_cells.append(html.Td(cell, style={"padding": "2px"}))
        cells.append(html.Tr(row_cells))

    return html.Div([
        html.H6(
            f"Kernel {kernel_names[kernel_index]}",
            style={"color": "white", "fontSize": "13px", "marginBottom": "8px"}
        ),
        html.Table(
            html.Tbody(cells),
            style={
                "backgroundColor": "rgba(0,0,0,0.3)",
                "padding": "5px",
                "borderRadius": "4px"
            }
        )
    ], style={"marginBottom": "15px"})


def create_filter_editor(filter_index: int, kernel_size: tuple = (3, 3), num_filters_total: int = 1) -> html.Div:
    """
    Crea el editor completo para un filtro (3 kernels: R, G, B).

    Args:
        filter_index: Índice del filtro
        kernel_size: Tamaño de cada kernel
        num_filters_total: Número total de filtros (para mostrar N de M)
    """
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Span(
                    f"Filtro {filter_index + 1} de {num_filters_total}",
                    style={"fontWeight": "bold", "flex": "1"}
                ),
                dbc.Button(
                    [html.I(className="fas fa-trash")],
                    id={"type": "delete-filter-btn", "index": filter_index},
                    color="danger",
                    size="sm",
                    outline=True
                )
            ], style={"display": "flex", "alignItems": "center", "width": "100%"})
        ], style={"backgroundColor": "#2c3e50", "color": "white"}),
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.Label("Tamaño del Kernel:", style={"color": "white", "fontSize": "13px", "marginRight": "10px"}),
                    dcc.Dropdown(
                        id={"type": "kernel-size-dropdown", "filter": filter_index},
                        options=[
                            {"label": "3×3", "value": "3x3"},
                            {"label": "5×5", "value": "5x5"},
                            {"label": "7×7", "value": "7x7"}
                        ],
                        value=f"{kernel_size[0]}x{kernel_size[1]}",
                        style={"width": "100px", "display": "inline-block"},
                        clearable=False
                    )
                ], style={"marginBottom": "15px"}),

                # Contenedor para los 3 kernels (R, G, B)
                html.Div(
                    id={"type": "kernels-container", "filter": filter_index},
                    children=[
                        create_kernel_matrix_editor(filter_index, k, kernel_size)
                        for k in range(3)
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(3, 1fr)",
                        "gap": "15px"
                    }
                ),

                # Parámetros adicionales del filtro
                html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "20px 0"}),

                html.Div([
                    dbc.Label("Stride:", style={"color": "white", "fontSize": "13px", "minWidth": "100px"}),
                    html.Div([
                        dbc.Input(
                            id={"type": "stride-h", "filter": filter_index},
                            type="number",
                            value=1,
                            min=1,
                            style={"width": "60px", "marginRight": "5px"}
                        ),
                        html.Span("×", style={"color": "white", "margin": "0 5px"}),
                        dbc.Input(
                            id={"type": "stride-w", "filter": filter_index},
                            type="number",
                            value=1,
                            min=1,
                            style={"width": "60px"}
                        )
                    ], style={"display": "flex", "alignItems": "center"})
                ], className="input-field-group"),

                html.Div([
                    dbc.Label("Padding:", style={"color": "white", "fontSize": "13px", "minWidth": "100px"}),
                    dcc.Dropdown(
                        id={"type": "padding", "filter": filter_index},
                        options=[
                            {"label": "Same (mantiene tamaño)", "value": "same"},
                            {"label": "Valid (sin padding)", "value": "valid"}
                        ],
                        value="same",
                        style={"flex": "1"},
                        clearable=False
                    )
                ], className="input-field-group"),

                html.Div([
                    dbc.Label("Activación:", style={"color": "white", "fontSize": "13px", "minWidth": "100px"}),
                    dcc.Dropdown(
                        id={"type": "activation", "filter": filter_index},
                        options=[
                            {"label": "ReLU", "value": "relu"},
                            {"label": "Tanh", "value": "tanh"},
                            {"label": "Sigmoid", "value": "sigmoid"},
                            {"label": "Linear", "value": "linear"}
                        ],
                        value="relu",
                        style={"flex": "1"},
                        clearable=False
                    )
                ], className="input-field-group")
            ])
        ])
    ], className="mb-3", style={"backgroundColor": "#34495e"})


def create_convolution_layer_config(layer_index: int) -> html.Div:
    """
    Crea la configuración completa de una capa convolucional con múltiples filtros.

    Args:
        layer_index: Índice de la capa
    """
    return html.Div([
        # Store para los filtros de esta capa
        dcc.Store(id={"type": "conv-filters-store", "layer": layer_index}, data=[]),

        # Descripción de la capa
        dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            html.Strong("Capa Convolucional: "),
            "Extrae características espaciales usando filtros deslizantes (kernels). ",
            "Cada filtro tiene 3 kernels (R, G, B) que se aplican sobre diferentes canales de la entrada."
        ], color="info", style={"fontSize": "13px"}),

        # Botón para agregar filtros
        html.Div([
            dbc.Button(
                [html.I(className="fas fa-plus me-2"), "Agregar Filtro"],
                id={"type": "add-filter-btn", "layer": layer_index},
                color="success",
                size="sm",
                className="mb-3"
            ),
            html.Span(
                id={"type": "filter-count", "layer": layer_index},
                style={"color": "rgba(255,255,255,0.7)", "marginLeft": "15px", "fontSize": "14px"}
            )
        ], style={"display": "flex", "alignItems": "center"}),

        # Contenedor de filtros
        html.Div(
            id={"type": "filters-container", "layer": layer_index},
            children=[],
            style={"marginTop": "15px"}
        )
    ])


# ============ CALLBACKS ============

def register_cnn_kernel_callbacks():
    """Registra callbacks para el editor de kernels de CNN."""

    # Callback: Agregar nuevo filtro
    @callback(
        [Output({"type": "conv-filters-store", "layer": MATCH}, "data"),
         Output({"type": "filters-container", "layer": MATCH}, "children"),
         Output({"type": "filter-count", "layer": MATCH}, "children")],
        Input({"type": "add-filter-btn", "layer": MATCH}, "n_clicks"),
        State({"type": "conv-filters-store", "layer": MATCH}, "data"),
        prevent_initial_call=True
    )
    def add_new_filter(n_clicks, current_filters):
        if not n_clicks:
            return no_update, no_update, no_update

        current_filters = current_filters or []
        new_filter_index = len(current_filters)

        # Agregar nuevo filtro a la lista
        new_filter = {
            "index": new_filter_index,
            "kernel_size": (3, 3),
            "kernels": [
                [[0.0 for _ in range(3)] for _ in range(3)] for _ in range(3)
            ],
            "stride": [1, 1],
            "padding": "same",
            "activation": "relu"
        }
        current_filters.append(new_filter)

        # Crear UI para todos los filtros
        filter_editors = [
            create_filter_editor(i, (3, 3), len(current_filters))
            for i in range(len(current_filters))
        ]

        count_text = f"Total: {len(current_filters)} filtro(s)"

        return current_filters, filter_editors, count_text


    # Callback: Eliminar filtro
    @callback(
        [Output({"type": "conv-filters-store", "layer": MATCH}, "data", allow_duplicate=True),
         Output({"type": "filters-container", "layer": MATCH}, "children", allow_duplicate=True),
         Output({"type": "filter-count", "layer": MATCH}, "children", allow_duplicate=True)],
        Input({"type": "delete-filter-btn", "index": ALL}, "n_clicks"),
        State({"type": "conv-filters-store", "layer": MATCH}, "data"),
        prevent_initial_call=True
    )
    def delete_filter(n_clicks_list, current_filters):
        if not any(n_clicks_list):
            return no_update, no_update, no_update

        triggered = ctx.triggered_id
        if not triggered:
            return no_update, no_update, no_update

        filter_index = triggered["index"]
        current_filters = current_filters or []

        if filter_index < len(current_filters):
            current_filters.pop(filter_index)

            # Reindexar
            for i, f in enumerate(current_filters):
                f["index"] = i

        # Recrear UI
        filter_editors = [
            create_filter_editor(i, (3, 3), len(current_filters))
            for i in range(len(current_filters))
        ]

        count_text = f"Total: {len(current_filters)} filtro(s)" if current_filters else "No hay filtros"

        return current_filters, filter_editors, count_text


    # Callback: Cambiar tamaño de kernel
    @callback(
        Output({"type": "kernels-container", "filter": MATCH}, "children"),
        Input({"type": "kernel-size-dropdown", "filter": MATCH}, "value"),
        State({"type": "kernel-size-dropdown", "filter": MATCH}, "id"),
        prevent_initial_call=True
    )
    def update_kernel_size(size_value, dropdown_id):
        if not size_value:
            return no_update

        filter_index = dropdown_id["filter"]
        rows, cols = map(int, size_value.split("x"))

        # Recrear matrices con nuevo tamaño
        return [
            create_kernel_matrix_editor(filter_index, k, (rows, cols))
            for k in range(3)
        ]


# Registrar callbacks al importar
register_cnn_kernel_callbacks()
