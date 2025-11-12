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
                value=round(values[i][j], 4),  # Redondear para mejor visualización
                step=0.01,
                disabled=True,  # ✅ Deshabilitar edición (generado automáticamente)
                style={
                    "width": "60px",
                    "height": "40px",
                    "textAlign": "center",
                    "fontSize": "12px",
                    "padding": "5px",
                    "backgroundColor": "#2d3748",  # Fondo más oscuro para indicar disabled
                    "cursor": "not-allowed"
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


def create_filter_editor(filter_index: int, kernel_size: tuple = (3, 3), num_filters_total: int = 1, saved_filter_data: Dict = None) -> html.Div:
    """
    Crea el editor completo para un filtro (3 kernels: R, G, B).

    Args:
        filter_index: Índice del filtro
        kernel_size: Tamaño de cada kernel
        num_filters_total: Número total de filtros (para mostrar N de M)
        saved_filter_data: Datos guardados del filtro (kernels, stride, padding, activation)
    """
    saved_filter_data = saved_filter_data or {}

    # Extraer valores guardados
    saved_kernels = saved_filter_data.get("kernels", None)
    saved_stride = saved_filter_data.get("stride", [1, 1])
    saved_padding = saved_filter_data.get("padding", "same")
    saved_activation = saved_filter_data.get("activation", "relu")
    saved_kernel_size = saved_filter_data.get("kernel_size", kernel_size)

    # Si hay kernels guardados, usar su tamaño
    if saved_kernels and len(saved_kernels) > 0 and len(saved_kernels[0]) > 0:
        saved_kernel_size = (len(saved_kernels[0]), len(saved_kernels[0][0]))

    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Span(
                    f"Filtro {filter_index + 1} de {num_filters_total} (generado aleatoriamente)",
                    style={"fontWeight": "bold", "flex": "1", "fontSize": "14px"}
                ),
                dbc.Button(
                    [html.I(className="fas fa-trash")],
                    id={"type": "delete-filter-btn", "index": filter_index},
                    color="danger",
                    size="sm",
                    outline=True
                )
            ], style={"display": "flex", "alignItems": "center", "width": "100%", "gap": "5px"})
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
                        value=f"{saved_kernel_size[0]}x{saved_kernel_size[1]}",
                        style={"width": "100px", "display": "inline-block"},
                        clearable=False,
                        disabled=True  # ✅ Disabled: todos los filtros deben tener el mismo tamaño
                    ),
                    html.Span(
                        " (tamaño fijo para toda la capa)",
                        style={"color": "#999", "fontSize": "11px", "marginLeft": "10px", "fontStyle": "italic"}
                    )
                ], style={"marginBottom": "15px"}),

                # Contenedor para los 3 kernels (R, G, B)
                html.Div(
                    id={"type": "kernels-container", "filter": filter_index},
                    children=[
                        create_kernel_matrix_editor(
                            filter_index,
                            k,
                            saved_kernel_size,
                            saved_kernels[k] if saved_kernels and k < len(saved_kernels) else None
                        )
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
                            value=saved_stride[0] if isinstance(saved_stride, list) and len(saved_stride) > 0 else 1,
                            min=1,
                            style={"width": "60px", "marginRight": "5px"}
                        ),
                        html.Span("×", style={"color": "white", "margin": "0 5px"}),
                        dbc.Input(
                            id={"type": "stride-w", "filter": filter_index},
                            type="number",
                            value=saved_stride[1] if isinstance(saved_stride, list) and len(saved_stride) > 1 else 1,
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
                        value=saved_padding,
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
                        value=saved_activation,
                        style={"flex": "1"},
                        clearable=False
                    )
                ], className="input-field-group")
            ])
        ])
    ], className="mb-3", style={"backgroundColor": "#34495e"})


def create_convolution_layer_config(layer_index: int, saved_filters: List[Dict] = None) -> html.Div:
    """
    Crea la configuración completa de una capa convolucional con múltiples filtros.

    Args:
        layer_index: Índice de la capa
        saved_filters: Filtros guardados previamente para restaurar
    """
    # Usar filtros guardados si existen, sino lista vacía
    initial_filters = saved_filters if saved_filters else []

    # Crear editores de filtros si hay filtros guardados
    initial_filter_editors = []
    if initial_filters:
        initial_filter_editors = [
            create_filter_editor(i, (3, 3), len(initial_filters), initial_filters[i])
            for i in range(len(initial_filters))
        ]

    initial_count_text = f"Total: {len(initial_filters)} filtro(s)" if initial_filters else ""

    # Determinar tamaño de kernel por defecto
    default_kernel_size = "3x3"
    if initial_filters and len(initial_filters) > 0:
        first_filter_size = initial_filters[0].get("kernel_size", [3, 3])
        default_kernel_size = f"{first_filter_size[0]}x{first_filter_size[1]}"

    return html.Div([
        # Store para los filtros de esta capa - inicializar con filtros guardados
        dcc.Store(id={"type": "conv-filters-store", "layer": layer_index}, data=initial_filters),

        # Store para el tamaño de kernel de la capa (aplicable a todos los filtros)
        dcc.Store(id={"type": "kernel-size-store", "layer": layer_index}, data=default_kernel_size),

        # Div para notificaciones de guardado
        html.Div(id={"type": "filter-save-notification", "layer": layer_index}, style={
            "position": "fixed",
            "top": "20px",
            "right": "20px",
            "zIndex": "9999",
            "minWidth": "250px"
        }),

        # Descripción de la capa
        dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            html.Strong("Capa Convolucional: "),
            "Extrae características espaciales usando filtros deslizantes (kernels). ",
            "Cada filtro tiene 3 kernels (R, G, B) que se aplican sobre diferentes canales de la entrada."
        ], color="info", style={"fontSize": "13px"}),

        # Selector de tamaño de kernel (aplicable a TODOS los filtros)
        html.Div([
            html.Label("Tamaño de Kernel para esta capa:", style={"color": "white", "fontSize": "14px", "marginRight": "10px", "fontWeight": "bold"}),
            dcc.Dropdown(
                id={"type": "layer-kernel-size-selector", "layer": layer_index},
                options=[
                    {"label": "3×3 (rápido, menos parámetros)", "value": "3x3"},
                    {"label": "5×5 (balance)", "value": "5x5"},
                    {"label": "7×7 (campo receptivo grande)", "value": "7x7"}
                ],
                value=default_kernel_size,
                clearable=False,
                disabled=len(initial_filters) > 0,  # Disabled si ya hay filtros
                style={"width": "350px", "display": "inline-block"}
            ),
            html.Span(
                " ⚠️ No se puede cambiar después de agregar filtros" if len(initial_filters) > 0 else "",
                style={"color": "#ff6b6b", "fontSize": "12px", "marginLeft": "10px", "fontStyle": "italic"}
            )
        ], style={"marginBottom": "20px", "display": "flex", "alignItems": "center"}),

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
                initial_count_text,
                id={"type": "filter-count", "layer": layer_index},
                style={"color": "rgba(255,255,255,0.7)", "marginLeft": "15px", "fontSize": "14px"}
            )
        ], style={"display": "flex", "alignItems": "center"}),

        # Contenedor de filtros - inicializar con filtros guardados
        html.Div(
            id={"type": "filters-container", "layer": layer_index},
            children=initial_filter_editors,
            style={"marginTop": "15px"}
        )
    ])


# ============ CALLBACKS ============

def register_cnn_kernel_callbacks():
    """Registra callbacks para el editor de kernels de CNN."""

    # Callback: Actualizar store de tamaño de kernel cuando cambia el selector
    @callback(
        Output({"type": "kernel-size-store", "layer": MATCH}, "data"),
        Input({"type": "layer-kernel-size-selector", "layer": MATCH}, "value"),
        prevent_initial_call=True
    )
    def update_kernel_size_store(selected_size):
        """Actualiza el store con el tamaño de kernel seleccionado."""
        return selected_size if selected_size else "3x3"

    # Callback: Agregar nuevo filtro con kernels aleatorios
    @callback(
        [Output({"type": "conv-filters-store", "layer": MATCH}, "data", allow_duplicate=True),
         Output({"type": "filters-container", "layer": MATCH}, "children"),
         Output({"type": "filter-count", "layer": MATCH}, "children"),
         Output({"type": "layer-kernel-size-selector", "layer": MATCH}, "disabled")],
        Input({"type": "add-filter-btn", "layer": MATCH}, "n_clicks"),
        [State({"type": "conv-filters-store", "layer": MATCH}, "data"),
         State({"type": "kernel-size-store", "layer": MATCH}, "data"),
         State({"type": "stride-h", "filter": ALL}, "value"),
         State({"type": "stride-w", "filter": ALL}, "value"),
         State({"type": "stride-h", "filter": ALL}, "id"),
         State({"type": "padding", "filter": ALL}, "value"),
         State({"type": "activation", "filter": ALL}, "value")],
        prevent_initial_call=True
    )
    def add_new_filter(n_clicks, current_filters, kernel_size_str,
                       stride_h_values, stride_w_values, stride_h_ids,
                       padding_values, activation_values):
        """Agrega un nuevo filtro con kernels RGB generados aleatoriamente."""
        if not n_clicks:
            return no_update, no_update, no_update, no_update

        import copy
        import numpy as np

        current_filters = copy.deepcopy(current_filters) if current_filters else []

        # AUTO-GUARDAR parámetros editables (stride, padding, activation) de filtros existentes
        if stride_h_ids and stride_h_values and stride_w_values:
            for stride_h_id, stride_h_val, stride_w_val in zip(stride_h_ids, stride_h_values, stride_w_values):
                filter_idx = stride_h_id["filter"]
                if filter_idx < len(current_filters):
                    current_filters[filter_idx]["stride"] = [
                        int(stride_h_val) if stride_h_val is not None else 1,
                        int(stride_w_val) if stride_w_val is not None else 1
                    ]

        if padding_values:
            for i, padding_val in enumerate(padding_values):
                if i < len(current_filters) and padding_val is not None:
                    current_filters[i]["padding"] = padding_val

        if activation_values:
            for i, activation_val in enumerate(activation_values):
                if i < len(current_filters) and activation_val is not None:
                    current_filters[i]["activation"] = activation_val

        print("💾 Auto-guardado: Parámetros de filtros existentes guardados")

        # ✅ Obtener tamaño de kernel del selector (aplicable a todos los filtros)
        if not kernel_size_str:
            kernel_size_str = "3x3"

        rows, cols = map(int, kernel_size_str.split("x"))
        kernel_size = (rows, cols)

        new_filter_index = len(current_filters)

        # ✅ Generar kernels RGB aleatorios con el tamaño seleccionado
        kernel_R = np.random.randn(*kernel_size).astype(np.float32).tolist()
        kernel_G = np.random.randn(*kernel_size).astype(np.float32).tolist()
        kernel_B = np.random.randn(*kernel_size).astype(np.float32).tolist()

        new_filter = {
            "index": new_filter_index,
            "kernel_size": list(kernel_size),  # Guardar como lista para JSON
            "kernels": [kernel_R, kernel_G, kernel_B],  # RGB kernels aleatorios
            "stride": [1, 1],
            "padding": "same",
            "activation": "relu"
        }
        current_filters.append(new_filter)

        print(f"🎲 Nuevo filtro {new_filter_index} generado con kernels {kernel_size_str} aleatorios")

        # Crear UI para todos los filtros con sus datos guardados
        filter_editors = [
            create_filter_editor(i, tuple(current_filters[i]["kernel_size"]), len(current_filters), current_filters[i])
            for i in range(len(current_filters))
        ]

        count_text = f"Total: {len(current_filters)} filtro(s)"

        # Deshabilitar el selector después de agregar el primer filtro
        disable_selector = True

        return current_filters, filter_editors, count_text, disable_selector


    # Callback: Eliminar filtro (con auto-guardado previo)
    @callback(
        [Output({"type": "conv-filters-store", "layer": MATCH}, "data", allow_duplicate=True),
         Output({"type": "filters-container", "layer": MATCH}, "children", allow_duplicate=True),
         Output({"type": "filter-count", "layer": MATCH}, "children", allow_duplicate=True)],
        Input({"type": "delete-filter-btn", "index": ALL}, "n_clicks"),
        [State({"type": "conv-filters-store", "layer": MATCH}, "data"),
         State({"type": "kernel-cell", "filter": ALL, "kernel": ALL, "row": ALL, "col": ALL}, "value"),
         State({"type": "kernel-cell", "filter": ALL, "kernel": ALL, "row": ALL, "col": ALL}, "id"),
         State({"type": "stride-h", "filter": ALL}, "value"),
         State({"type": "stride-w", "filter": ALL}, "value"),
         State({"type": "stride-h", "filter": ALL}, "id"),
         State({"type": "padding", "filter": ALL}, "value"),
         State({"type": "activation", "filter": ALL}, "value"),
         State({"type": "kernel-size-dropdown", "filter": ALL}, "value"),
         State({"type": "kernel-size-dropdown", "filter": ALL}, "id")],
        prevent_initial_call=True
    )
    def delete_filter(n_clicks_list, current_filters, kernel_values, kernel_ids,
                      stride_h_values, stride_w_values, stride_h_ids,
                      padding_values, activation_values, kernel_size_values, kernel_size_ids):
        """Elimina un filtro. AUTO-GUARDA todos los filtros antes de eliminar."""
        if not any(n_clicks_list):
            return no_update, no_update, no_update

        triggered = ctx.triggered_id
        if not triggered:
            return no_update, no_update, no_update

        import copy
        current_filters = copy.deepcopy(current_filters) if current_filters else []

        # AUTO-GUARDAR: Capturar valores actuales antes de eliminar
        if kernel_ids and kernel_values:
            for cell_id, cell_value in zip(kernel_ids, kernel_values):
                filter_idx = cell_id["filter"]
                kernel_idx = cell_id["kernel"]
                row = cell_id["row"]
                col = cell_id["col"]

                if filter_idx < len(current_filters):
                    if "kernels" not in current_filters[filter_idx]:
                        current_filters[filter_idx]["kernels"] = [
                            [[0.0 for _ in range(3)] for _ in range(3)] for _ in range(3)
                        ]
                    if kernel_idx < len(current_filters[filter_idx]["kernels"]):
                        if row < len(current_filters[filter_idx]["kernels"][kernel_idx]):
                            if col < len(current_filters[filter_idx]["kernels"][kernel_idx][row]):
                                current_filters[filter_idx]["kernels"][kernel_idx][row][col] = float(cell_value) if cell_value is not None else 0.0

        if stride_h_ids and stride_h_values and stride_w_values:
            for stride_h_id, stride_h_val, stride_w_val in zip(stride_h_ids, stride_h_values, stride_w_values):
                filter_idx = stride_h_id["filter"]
                if filter_idx < len(current_filters):
                    current_filters[filter_idx]["stride"] = [
                        int(stride_h_val) if stride_h_val is not None else 1,
                        int(stride_w_val) if stride_w_val is not None else 1
                    ]

        if padding_values:
            for i, padding_val in enumerate(padding_values):
                if i < len(current_filters) and padding_val is not None:
                    current_filters[i]["padding"] = padding_val

        if activation_values:
            for i, activation_val in enumerate(activation_values):
                if i < len(current_filters) and activation_val is not None:
                    current_filters[i]["activation"] = activation_val

        if kernel_size_ids and kernel_size_values:
            for size_id, size_val in zip(kernel_size_ids, kernel_size_values):
                filter_idx = size_id["filter"]
                if filter_idx < len(current_filters) and size_val:
                    rows, cols = map(int, size_val.split("x"))
                    current_filters[filter_idx]["kernel_size"] = (rows, cols)

        print("💾 Auto-guardado: Filtros guardados antes de eliminar")

        # Eliminar el filtro
        filter_index = triggered["index"]
        if filter_index < len(current_filters):
            current_filters.pop(filter_index)

            # Reindexar
            for i, f in enumerate(current_filters):
                f["index"] = i

        # Recrear UI con datos guardados del store
        filter_editors = [
            create_filter_editor(i, (3, 3), len(current_filters), current_filters[i])
            for i in range(len(current_filters))
        ]

        count_text = f"Total: {len(current_filters)} filtro(s)" if current_filters else "No hay filtros"

        return current_filters, filter_editors, count_text


    # Callback: Cambiar tamaño de kernel - solo actualiza UI
    @callback(
        Output({"type": "kernels-container", "filter": MATCH}, "children"),
        Input({"type": "kernel-size-dropdown", "filter": MATCH}, "value"),
        State({"type": "kernel-size-dropdown", "filter": MATCH}, "id"),
        prevent_initial_call=True
    )
    def update_kernel_size_ui(size_value, dropdown_id):
        """Solo actualiza la UI con el nuevo tamaño. Los valores se pierden temporalmente pero se restaurarán."""
        if not size_value:
            return no_update

        filter_index = dropdown_id["filter"]
        rows, cols = map(int, size_value.split("x"))

        # Recrear matrices con nuevo tamaño (valores en 0 temporalmente)
        return [
            create_kernel_matrix_editor(filter_index, k, (rows, cols))
            for k in range(3)
        ]



# Registrar callbacks al importar
register_cnn_kernel_callbacks()
