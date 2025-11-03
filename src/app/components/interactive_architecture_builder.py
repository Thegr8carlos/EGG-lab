"""
Sistema interactivo de construcción de arquitecturas de redes neuronales.
Permite al usuario agregar capas dinámicamente y configurarlas paso a paso.
"""

import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, no_update, ctx
import dash_bootstrap_components as dbc
from typing import Dict, Any, List, Tuple
import json

# Importar editor de kernels de CNN
from app.components.cnn_kernel_editor import create_convolution_layer_config

# Colores para tipos de capas
LAYER_COLORS = {
    "input": "#4A90E2",
    "lstm": "#F5A623",
    "gru": "#BD10E0",
    "dense": "#50E3C2",
    "conv": "#7ED321",
    "pooling": "#D0021B",
    "flatten": "#9013FE",
    "dropout": "#FF6B6B",
    "batchnorm": "#95E1D3",
    "output": "#417505"
}

# Iconos Font Awesome para cada tipo de capa
LAYER_ICONS = {
    "input": "fa-sign-in-alt",
    "LSTMLayer": "fa-project-diagram",
    "GRULayer": "fa-circle-notch",
    "DenseLayer": "fa-layer-group",
    "ConvolutionLayer": "fa-th",
    "PoolingLayer": "fa-compress-arrows-alt",
    "Dropout": "fa-random",
    "BatchNorm": "fa-balance-scale",
    "Flatten": "fa-align-justify",
    "output": "fa-flag-checkered"
}

# Definición de tipos de capas disponibles por modelo
AVAILABLE_LAYERS = {
    "LSTM": ["LSTMLayer", "DenseLayer", "Dropout"],
    "GRU": ["GRULayer", "DenseLayer", "Dropout"],
    "CNN": ["ConvolutionLayer", "PoolingLayer", "Flatten", "DenseLayer", "Dropout"],
    "SVNN": ["DenseLayer", "Dropout", "BatchNorm"]
}

# Nombres amigables
LAYER_NAMES = {
    "LSTMLayer": "Capa LSTM",
    "GRULayer": "Capa GRU",
    "DenseLayer": "Capa Densa",
    "ConvolutionLayer": "Capa Convolucional",
    "PoolingLayer": "Capa de Pooling",
    "Dropout": "Dropout",
    "BatchNorm": "Batch Normalization",
    "Flatten": "Aplanar"
}

# Descripciones de qué hace cada capa
LAYER_DESCRIPTIONS = {
    "LSTMLayer": {
        "short": "Procesa secuencias temporales capturando dependencias a largo plazo.",
        "details": "Las LSTM (Long Short-Term Memory) son ideales para datos secuenciales como señales EEG. Pueden recordar información importante durante largos períodos y olvidar la irrelevante mediante sus puertas de entrada, salida y olvido."
    },
    "GRULayer": {
        "short": "Versión simplificada de LSTM, procesa secuencias de forma más eficiente.",
        "details": "Las GRU (Gated Recurrent Units) son más rápidas que LSTM con solo 2 puertas (reset y update). Funcionan bien para secuencias donde las dependencias no son extremadamente largas."
    },
    "DenseLayer": {
        "short": "Capa completamente conectada que aprende representaciones no lineales.",
        "details": "Cada neurona está conectada a todas las neuronas de la capa anterior. Es la capa más común para clasificación y aprendizaje de patrones complejos después de la extracción de características."
    },
    "ConvolutionLayer": {
        "short": "Extrae características espaciales usando filtros deslizantes (kernels).",
        "details": "Aplica múltiples filtros sobre la entrada para detectar patrones locales como bordes, texturas o formas. Cada filtro aprende a detectar un tipo específico de característica en diferentes posiciones de la imagen/señal."
    },
    "PoolingLayer": {
        "short": "Reduce dimensionalidad preservando características importantes.",
        "details": "Max Pooling toma el valor máximo en cada región, manteniendo las características más prominentes. Avg Pooling promedia los valores. Ambos reducen el tamaño espacial y el costo computacional."
    },
    "Dropout": {
        "short": "Regularización: desactiva neuronas aleatoriamente para evitar overfitting.",
        "details": "Durante el entrenamiento, apaga aleatoriamente un porcentaje de neuronas. Esto previene que la red dependa demasiado de neuronas específicas y mejora la generalización."
    },
    "BatchNorm": {
        "short": "Normaliza las activaciones para entrenamient más estable y rápido.",
        "details": "Normaliza las salidas de cada capa para tener media 0 y varianza 1. Acelera el entrenamiento, permite tasas de aprendizaje más altas y actúa como regularización."
    },
    "Flatten": {
        "short": "Convierte matrices multidimensionales en un vector 1D.",
        "details": "Transforma la salida de capas convolucionales/pooling (matrices 2D/3D) en un vector plano que puede alimentar capas densas. Esencial para la transición de CNN a clasificador denso."
    }
}


# ============ VALIDACIONES DE REGLAS DE NEGOCIO ============

def validate_layer_addition(new_layer_type: str, current_layers: List[Dict[str, Any]], model_type: str) -> Tuple[bool, str]:
    """
    Valida si se puede agregar una capa según las reglas de negocio de redes neuronales.

    Args:
        new_layer_type: Tipo de capa que se quiere agregar
        current_layers: Capas actuales en la arquitectura
        model_type: Tipo de modelo (LSTM, GRU, CNN, SVNN)

    Returns:
        (es_valido, mensaje_error)
    """
    if not current_layers:
        # Primera capa: debe ser del tipo principal del modelo
        if model_type == "LSTM" and new_layer_type != "LSTMLayer":
            return False, "La primera capa debe ser LSTM"
        if model_type == "GRU" and new_layer_type != "GRULayer":
            return False, "La primera capa debe ser GRU"
        if model_type == "CNN" and new_layer_type != "ConvolutionLayer":
            return False, "La primera capa debe ser Convolucional"
        # SVNN puede empezar con DenseLayer directamente
        return True, ""

    last_layer = current_layers[-1]["type"]

    # Regla: No se puede agregar Dropout o BatchNorm consecutivamente
    if new_layer_type == "Dropout" and last_layer == "Dropout":
        return False, "No puedes agregar Dropout después de otro Dropout"

    if new_layer_type == "BatchNorm" and last_layer == "BatchNorm":
        return False, "No puedes agregar BatchNorm después de otro BatchNorm"

    # Regla CNN: Pooling solo después de Convolución
    if model_type == "CNN":
        if new_layer_type == "PoolingLayer" and last_layer not in ["ConvolutionLayer", "PoolingLayer"]:
            return False, "Pooling debe ir después de una capa Convolucional"

        # Flatten debe ir después de capas convolucionales/pooling y antes de densas
        if new_layer_type == "Flatten":
            if last_layer not in ["ConvolutionLayer", "PoolingLayer"]:
                return False, "Flatten debe ir después de capas Convolucionales o Pooling"

        # Después de Flatten, solo se permiten capas densas o dropout
        if last_layer == "Flatten" and new_layer_type not in ["DenseLayer", "Dropout"]:
            return False, "Después de Flatten solo puedes agregar capas Densas o Dropout"

    # Regla LSTM/GRU: Después de capa recurrente, puedes agregar otra recurrente o densa
    if model_type in ["LSTM", "GRU"]:
        recurrent_type = "LSTMLayer" if model_type == "LSTM" else "GRULayer"

        # Si la última es recurrente
        if last_layer == recurrent_type:
            # Puedes agregar otra recurrente, densa o dropout
            if new_layer_type not in [recurrent_type, "DenseLayer", "Dropout"]:
                return False, f"Después de {LAYER_NAMES[recurrent_type]}, solo puedes agregar otra {LAYER_NAMES[recurrent_type]}, Densa o Dropout"

        # Si ya hay capas densas, solo puedes agregar más densas o dropout
        if last_layer == "DenseLayer":
            if new_layer_type == recurrent_type:
                return False, f"No puedes agregar {LAYER_NAMES[recurrent_type]} después de capas Densas"

    # Regla general: Mínimo una capa del tipo principal del modelo
    if model_type == "LSTM":
        has_lstm = any(layer["type"] == "LSTMLayer" for layer in current_layers)
        if not has_lstm and new_layer_type != "LSTMLayer":
            return False, "Debes tener al menos una capa LSTM en tu arquitectura"

    if model_type == "GRU":
        has_gru = any(layer["type"] == "GRULayer" for layer in current_layers)
        if not has_gru and new_layer_type != "GRULayer":
            return False, "Debes tener al menos una capa GRU en tu arquitectura"

    return True, ""


def validate_complete_architecture(layers: List[Dict[str, Any]], model_type: str) -> Tuple[bool, str]:
    """
    Valida que la arquitectura completa sea válida antes de entrenar.

    Args:
        layers: Lista completa de capas
        model_type: Tipo de modelo

    Returns:
        (es_valido, mensaje_error)
    """
    if not layers:
        return False, "La arquitectura está vacía. Agrega al menos una capa."

    # Debe tener al menos una capa del tipo principal
    if model_type == "LSTM":
        has_lstm = any(layer["type"] == "LSTMLayer" for layer in layers)
        if not has_lstm:
            return False, "La arquitectura debe contener al menos una capa LSTM"

    if model_type == "GRU":
        has_gru = any(layer["type"] == "GRULayer" for layer in layers)
        if not has_gru:
            return False, "La arquitectura debe contener al menos una capa GRU"

    if model_type == "CNN":
        has_conv = any(layer["type"] == "ConvolutionLayer" for layer in layers)
        if not has_conv:
            return False, "La arquitectura debe contener al menos una capa Convolucional"

        # CNN debe tener Flatten antes de capas densas
        has_dense = any(layer["type"] == "DenseLayer" for layer in layers)
        has_flatten = any(layer["type"] == "Flatten" for layer in layers)

        if has_dense and not has_flatten:
            return False, "Las CNNs deben tener una capa Flatten antes de las capas Densas"

    # No puede terminar con Dropout o BatchNorm
    last_layer = layers[-1]["type"]
    if last_layer in ["Dropout", "BatchNorm"]:
        return False, "La arquitectura no puede terminar con Dropout o BatchNorm. Agrega una capa final."

    return True, ""


def build_model_config_from_layers(layers: List[Dict], model_type: str) -> Dict[str, Any]:
    """
    Construye el diccionario de configuración completo del modelo desde las capas.

    Args:
        layers: Lista de capas con sus configuraciones
        model_type: Tipo de modelo (LSTM, GRU, CNN, SVNN)

    Returns:
        Diccionario con la configuración completa del modelo en formato Pydantic
    """
    config = {}

    if model_type in ["LSTM", "GRU"]:
        # Extraer capas LSTM/GRU para el encoder
        encoder_layers = []
        for layer in layers:
            if layer["type"] in ["LSTMLayer", "GRULayer"]:
                encoder_layers.append(layer["config"])

        config["encoder"] = {
            "layers": encoder_layers,
            "input_feature_dim": encoder_layers[0].get("input_size") if encoder_layers else 64
        }

        # Pooling temporal
        pooling_layers = [l for l in layers if l["type"] == "TemporalPooling"]
        if pooling_layers:
            config["pooling"] = pooling_layers[0]["config"]
        else:
            # Agregar pooling por defecto
            config["pooling"] = {"kind": "last"}

    elif model_type == "CNN":
        # Extraer capas convolucionales
        conv_layers = []
        for layer in layers:
            if layer["type"] == "ConvolutionLayer":
                conv_layers.append(layer["config"])

        if conv_layers:
            config["feature_extractor"] = {
                "layers": conv_layers
            }

        # Capas densas después de Flatten
        dense_layers = []
        for layer in layers:
            if layer["type"] == "DenseLayer":
                dense_layers.append(layer["config"])
        if dense_layers:
            config["classifier"] = {
                "layers": dense_layers
            }

    elif model_type == "SVNN":
        # Red neuronal simple
        dense_layers = []
        for layer in layers:
            if layer["type"] == "DenseLayer":
                dense_layers.append(layer["config"])

        if dense_layers:
            config["hidden_layers"] = dense_layers

    # Para LSTM/GRU: Necesitamos separar fc_layers de classification
    if model_type in ["LSTM", "GRU"]:
        # Encontrar todas las capas densas
        all_dense = [layer for layer in layers if layer["type"] == "DenseLayer"]

        if all_dense:
            # La última DenseLayer es la clasificación (debe tener softmax)
            classification_layer = all_dense[-1]["config"].copy()
            # Asegurar que tenga softmax
            classification_layer["activation"] = {"kind": "softmax"}
            config["classification"] = classification_layer

            # Las demás son fc_layers intermedias
            if len(all_dense) > 1:
                fc_layers = [layer["config"] for layer in all_dense[:-1]]
                config["fc_layers"] = fc_layers
        else:
            # Si no hay DenseLayer, crear una por defecto para clasificación
            config["classification"] = {
                "units": 2,  # Default: 2 clases
                "activation": {"kind": "softmax"}
            }

    # Parámetros de entrenamiento (pueden estar en cualquier capa o usar defaults)
    # Estos suelen ser campos top-level del modelo
    for layer in layers:
        layer_config = layer.get("config", {})
        if "epochs" in layer_config:
            config["epochs"] = layer_config["epochs"]
        if "batch_size" in layer_config:
            config["batch_size"] = layer_config["batch_size"]
        if "learning_rate" in layer_config:
            config["learning_rate"] = layer_config["learning_rate"]

    return config


def create_layer_node(layer_type: str, layer_index: int, is_current: bool = False, is_fixed: bool = False) -> html.Div:
    """
    Crea un nodo visual para una capa en la arquitectura.

    Args:
        layer_type: Tipo de capa (LSTMLayer, DenseLayer, etc.)
        layer_index: Índice de la capa
        is_current: Si es la capa actualmente seleccionada
        is_fixed: Si es una capa fija (input/output)
    """
    # Determinar color base
    color_key = layer_type.lower().replace("layer", "")
    node_color = LAYER_COLORS.get(color_key, "#999")

    # Obtener icono
    icon_class = LAYER_ICONS.get(layer_type, "fa-circle")

    border_style = "3px solid white" if is_current and not is_fixed else "none"
    cursor_style = "pointer" if not is_fixed else "default"
    opacity = "0.7" if is_fixed else "1"

    # Contenido del nodo
    node_content = html.I(className=f"fas {icon_class}", style={"fontSize": "24px"}) if is_fixed else str(layer_index + 1)

    return html.Div([
        html.Div(
            node_content,
            style={
                "backgroundColor": node_color,
                "width": "60px",
                "height": "60px",
                "borderRadius": "50%",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.3)",
                "border": border_style,
                "color": "white",
                "fontWeight": "bold",
                "fontSize": "20px",
                "cursor": cursor_style,
                "transition": "all 0.3s ease",
                "opacity": opacity
            },
            id={"type": "layer-node", "index": layer_index} if not is_fixed else {},
            n_clicks=0 if not is_fixed else None
        ),
        html.Div(
            LAYER_NAMES.get(layer_type, layer_type) if not is_fixed else ("Input Layer" if layer_type == "input" else "Output Layer"),
            style={
                "textAlign": "center",
                "marginTop": "8px",
                "fontSize": "11px",
                "color": "white" if not is_fixed else "rgba(255,255,255,0.6)",
                "fontWeight": "600"
            }
        )
    ], style={"padding": "10px", "minWidth": "80px"})


def create_architecture_visualization(layers: List[Dict[str, Any]], current_step: int) -> html.Div:
    """
    Crea la visualización completa de la arquitectura con todas las capas.
    Incluye nodos fijos de Input y Output.

    Args:
        layers: Lista de capas agregadas por el usuario
        current_step: Paso actual (índice de la capa siendo configurada)
    """
    nodes = []

    # Nodo de INPUT (fijo)
    nodes.append(create_layer_node("input", -1, False, is_fixed=True))

    # Agregar flecha
    nodes.append(html.Div([
        html.I(className="fas fa-arrow-right",
               style={"color": "rgba(255,255,255,0.5)", "fontSize": "20px"})
    ], style={"display": "flex", "alignItems": "center", "padding": "0 10px"}))

    if not layers:
        # Si no hay capas, mostrar placeholder
        nodes.append(html.Div([
            html.Div([
                html.I(className="fas fa-plus-circle fa-2x", style={"color": "rgba(255,255,255,0.3)"}),
                html.P("Agrega capas aquí",
                       style={"color": "rgba(255,255,255,0.5)", "marginTop": "10px", "fontSize": "12px"})
            ], style={"textAlign": "center", "padding": "20px"})
        ], style={
            "backgroundColor": "rgba(0, 0, 0, 0.2)",
            "borderRadius": "8px",
            "border": "2px dashed rgba(255, 255, 255, 0.2)",
            "minWidth": "120px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center"
        }))

        # Flecha hacia output
        nodes.append(html.Div([
            html.I(className="fas fa-arrow-right",
                   style={"color": "rgba(255,255,255,0.5)", "fontSize": "20px"})
        ], style={"display": "flex", "alignItems": "center", "padding": "0 10px"}))
    else:
        # Crear nodos de capas del usuario
        for i, layer in enumerate(layers):
            is_current = (i == current_step)
            nodes.append(create_layer_node(layer["type"], i, is_current))

            # Agregar flecha conectora
            nodes.append(html.Div([
                html.I(className="fas fa-arrow-right",
                       style={"color": "rgba(255,255,255,0.5)", "fontSize": "20px"})
            ], style={"display": "flex", "alignItems": "center", "padding": "0 10px"}))

    # Nodo de OUTPUT (fijo)
    nodes.append(create_layer_node("output", -1, False, is_fixed=True))

    return html.Div([
        html.H5("Arquitectura de Red",
                style={"color": "white", "marginBottom": "20px", "textAlign": "center"}),
        html.Div(
            nodes,
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "overflowX": "auto",
                "padding": "20px",
                "backgroundColor": "rgba(0, 0, 0, 0.2)",
                "borderRadius": "8px",
                "border": "1px solid rgba(255, 255, 255, 0.1)",
                "minHeight": "150px"
            }
        )
    ])


def create_add_layer_buttons(model_type: str) -> html.Div:
    """
    Crea botones para agregar diferentes tipos de capas según el modelo.
    Incluye iconos representativos para cada tipo.

    Args:
        model_type: Tipo de modelo (LSTM, GRU, CNN, SVNN)
    """
    available = AVAILABLE_LAYERS.get(model_type, [])

    buttons = []
    for layer_type in available:
        icon_class = LAYER_ICONS.get(layer_type, "fa-plus")
        buttons.append(
            dbc.Button([
                html.I(className=f"fas {icon_class} me-2"),
                f"{LAYER_NAMES.get(layer_type, layer_type)}"
            ],
            id={"type": "add-layer-btn", "layer_type": layer_type},
            color="success",
            outline=True,
            size="sm",
            className="me-2 mb-2",
            style={"fontSize": "13px", "fontWeight": "600"})
        )

    return html.Div([
        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "20px 0"}),
        html.H6("Agregar Capa:", style={"color": "white", "marginBottom": "10px"}),
        html.Div(buttons, style={"display": "flex", "flexWrap": "wrap"})
    ])


def get_layer_config_form(layer_type: str, layer_index: int, schema_defs: Dict[str, Any]) -> List:
    """
    Genera el formulario de configuración para un tipo de capa específico.

    Args:
        layer_type: Tipo de capa
        layer_index: Índice de la capa
        schema_defs: Definiciones del schema ($defs)
    """
    # Caso especial: ConvolutionLayer usa editor matricial
    if layer_type == "ConvolutionLayer":
        return [create_convolution_layer_config(layer_index)]

    layer_schema = schema_defs.get(layer_type, {})
    properties = layer_schema.get("properties", {})

    if not properties:
        return [html.P(f"No hay configuración disponible para {layer_type}",
                      style={"color": "rgba(255,255,255,0.5)"})]

    form_fields = []

    # Mostrar descripción de la capa
    description = LAYER_DESCRIPTIONS.get(layer_type, {})
    if description:
        form_fields.append(
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                html.Strong(f"{LAYER_NAMES.get(layer_type, layer_type)}: "),
                description.get("short", "")
            ], color="info", style={"fontSize": "13px"}, className="mb-3")
        )

    for field_name, field_info in properties.items():
        field_type = field_info.get("type", "string")
        field_label = field_info.get("title", field_name)
        field_desc = field_info.get("description", "")
        default_value = field_info.get("default")

        input_id = {
            "type": "layer-config-input",
            "layer_index": layer_index,
            "field": field_name
        }

        # Detectar enums
        enum_values = field_info.get("enum")
        if enum_values:
            form_fields.append(html.Div([
                dbc.Label(field_label, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dcc.Dropdown(
                    id=input_id,
                    options=[{"label": str(v), "value": v} for v in enum_values],
                    value=default_value,
                    placeholder=f"Selecciona valor",
                    style={"flex": "1", "color": "black", "fontSize": "14px"}
                )
            ], className="input-field-group"))
            continue

        # Campos numéricos
        if field_type in ["integer", "number"]:
            min_val = field_info.get("minimum", field_info.get("exclusiveMinimum"))
            max_val = field_info.get("maximum")

            form_fields.append(html.Div([
                dbc.Label(field_label, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dbc.Input(
                    id=input_id,
                    type="number",
                    min=min_val,
                    max=max_val,
                    value=default_value,
                    step=1 if field_type == "integer" else 0.001,
                    style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
                )
            ], className="input-field-group"))
            continue

        # Booleanos
        if field_type == "boolean":
            form_fields.append(html.Div([
                dbc.Label(field_label, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dbc.Checkbox(
                    id=input_id,
                    value=default_value if default_value is not None else False,
                    style={"marginTop": "10px"}
                )
            ], className="input-field-group"))
            continue

        # String por defecto
        form_fields.append(html.Div([
            dbc.Label(field_label, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
            dbc.Input(
                id=input_id,
                type="text",
                value=default_value,
                placeholder=f"Ingresa {field_label}",
                style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
            )
        ], className="input-field-group"))

    return form_fields


def create_interactive_config_card(model_name: str, schema: Dict[str, Any]) -> html.Div:
    """
    Crea la card interactiva completa para construcción de arquitectura.

    Args:
        model_name: Nombre del modelo
        schema: Schema completo del modelo
    """
    return html.Div([
        # Stores para estado
        dcc.Store(id="architecture-layers", data=[]),  # Lista de capas agregadas
        dcc.Store(id="current-step", data=0),  # Paso actual
        dcc.Store(id="model-type", data=model_name),  # Tipo de modelo
        dcc.Store(id="validation-trigger", data=None),  # Trigger para mensajes de validación

        # Toast para mensajes de validación
        html.Div(id="validation-message", style={
            "position": "fixed",
            "top": "20px",
            "right": "20px",
            "zIndex": "9999",
            "minWidth": "300px"
        }),

        dbc.Card([
            # Header con navegación
            dbc.CardHeader([
                html.Div([
                    dbc.Button(
                        "← Volver",
                        id="config-back-btn",
                        color="link",
                        size="sm",
                        style={"color": "white", "textDecoration": "none"}
                    ),
                    html.Div(
                        id="step-indicator",
                        style={
                            "color": "white",
                            "fontSize": "16px",
                            "fontWeight": "600",
                            "flex": "1",
                            "textAlign": "center"
                        }
                    ),
                    html.Div(style={"width": "80px"})  # Espaciador
                ], style={"display": "flex", "alignItems": "center", "width": "100%"})
            ], className="right-panel-card-header"),

            dbc.CardBody([
                # Visualización de arquitectura
                html.Div(id="architecture-visualization"),

                # Botones para agregar capas
                html.Div(id="add-layer-buttons"),

                # Formulario de configuración del paso actual
                html.Div(id="current-step-form", style={"marginTop": "30px"}),

                # Botón de eliminar capa (siempre presente pero oculto inicialmente)
                html.Div([
                    dbc.Button(
                        [html.I(className="fas fa-trash me-2"), "Eliminar Capa"],
                        id="delete-current-layer-btn",
                        color="danger",
                        outline=True,
                        size="sm",
                        style={"display": "none"}
                    )
                ], style={"marginTop": "10px", "marginBottom": "10px"}),

                # Navegación entre pasos (siempre presente)
                html.Div([
                    dbc.Button(
                        [html.I(className="fas fa-chevron-left me-2"), "Anterior"],
                        id="prev-step-btn",
                        color="secondary",
                        disabled=True,
                        className="me-2"
                    ),
                    dbc.Button(
                        ["Siguiente", html.I(className="fas fa-chevron-right ms-2")],
                        id="next-step-btn",
                        color="secondary",
                        disabled=True
                    ),
                    html.Span(
                        id="step-counter",
                        style={"color": "rgba(255,255,255,0.7)", "marginLeft": "20px", "fontSize": "14px"}
                    )
                ], style={"display": "flex", "alignItems": "center", "marginTop": "20px"}),

                # Botón final
                html.Div([
                    dbc.Button(
                        "Probar Configuración",
                        id={"type": "test-config-btn", "model": model_name},
                        color="primary",
                        className="w-100 mt-3",
                        style={"fontSize": "15px", "height": "42px", "fontWeight": "600"}
                    ),
                    # Alert para mostrar resultados de validación
                    html.Div(id={"type": "test-config-result", "model": model_name}, className="mt-3")
                ])
            ])
        ], className="right-panel-card")
    ], id="interactive-config-container")


# ============ CALLBACKS ============

def register_interactive_callbacks():
    """Registra todos los callbacks necesarios para el sistema interactivo."""

    # Store para mensajes de error/validación
    @callback(
        Output("validation-message", "children"),
        Input("validation-trigger", "data"),
        prevent_initial_call=True
    )
    def show_validation_message(validation_data):
        if not validation_data:
            return ""

        is_error = validation_data.get("is_error", False)
        message = validation_data.get("message", "")

        if not message:
            return ""

        color = "danger" if is_error else "success"

        alert = dbc.Alert(
            [html.I(className=f"fas fa-{'exclamation-triangle' if is_error else 'check-circle'} me-2"), message],
            color=color,
            dismissable=True,
            duration=4000
        )

        return alert


    # Callback: Agregar capa a la arquitectura con validación
    @callback(
        [Output("architecture-layers", "data"),
         Output("validation-trigger", "data")],
        Input({"type": "add-layer-btn", "layer_type": ALL}, "n_clicks"),
        [State("architecture-layers", "data"),
         State("model-type", "data")],
        prevent_initial_call=True
    )
    def add_layer(n_clicks_list, current_layers, model_type):
        if not any(n_clicks_list):
            return no_update, no_update

        triggered = ctx.triggered_id
        if not triggered:
            return no_update, no_update

        layer_type = triggered["layer_type"]
        current_layers = current_layers or []

        # Validar si se puede agregar la capa
        is_valid, error_message = validate_layer_addition(layer_type, current_layers, model_type)

        if not is_valid:
            # Mostrar error
            return no_update, {"is_error": True, "message": error_message}

        # Agregar la capa
        new_layer = {
            "type": layer_type,
            "config": {}
        }

        current_layers.append(new_layer)

        return current_layers, {"is_error": False, "message": f"✓ {LAYER_NAMES.get(layer_type, layer_type)} agregada"}


    # Callback: Actualizar visualización de arquitectura
    @callback(
        Output("architecture-visualization", "children"),
        [Input("architecture-layers", "data"),
         Input("current-step", "data")]
    )
    def update_visualization(layers, current_step):
        return create_architecture_visualization(layers or [], current_step or 0)


    # Callback: Actualizar indicador de paso
    @callback(
        Output("step-indicator", "children"),
        [Input("current-step", "data"),
         Input("architecture-layers", "data"),
         Input("model-type", "data")]
    )
    def update_step_indicator(current_step, layers, model_type):
        if not layers or len(layers) == 0:
            return f"CONFIGURACIÓN: {model_type.upper()}"

        total_steps = len(layers)
        step_num = (current_step or 0) + 1
        layer = layers[current_step or 0]
        layer_name = LAYER_NAMES.get(layer["type"], layer["type"])

        return f"Paso {step_num}/{total_steps}: {layer_name}"


    # Callback: Mostrar botones de agregar capa
    @callback(
        Output("add-layer-buttons", "children"),
        Input("model-type", "data")
    )
    def show_add_buttons(model_type):
        if not model_type:
            return html.Div()
        return create_add_layer_buttons(model_type)


    # Callback: Mostrar formulario del paso actual
    @callback(
        [Output("current-step-form", "children"),
         Output("delete-current-layer-btn", "style")],
        [Input("current-step", "data"),
         Input("architecture-layers", "data")],
        State("model-type", "data")
    )
    def show_current_step_form(current_step, layers, model_type):
        if not layers or len(layers) == 0:
            return (
                html.Div([
                    html.P(
                        "Comienza agregando capas a tu arquitectura usando los botones de abajo",
                        style={"color": "rgba(255,255,255,0.5)", "textAlign": "center", "padding": "40px"}
                    )
                ]),
                {"display": "none"}  # Ocultar botón de eliminar
            )

        step = current_step or 0
        if step >= len(layers):
            return html.Div(), {"display": "none"}

        layer = layers[step]

        # Obtener schema del modelo (necesitamos cargarlo)
        from backend.classes.ClasificationModel.ClassifierSchemaFactory import ClassifierSchemaFactory
        schemas = ClassifierSchemaFactory.get_all_classifier_schemas()
        model_schema = schemas.get(model_type, {})
        schema_defs = model_schema.get("$defs", {})

        form_fields = get_layer_config_form(layer["type"], step, schema_defs)

        return (
            html.Div([
                html.Div([
                    html.H5(
                        f"Configurar: {LAYER_NAMES.get(layer['type'], layer['type'])}",
                        style={"color": "white", "marginBottom": "20px", "flex": "1"}
                    )
                ], style={"marginBottom": "20px"}),
                html.Div(form_fields)
            ]),
            {"display": "inline-block"}  # Mostrar botón de eliminar
        )


    # Callback: Eliminar capa actual
    @callback(
        [Output("architecture-layers", "data", allow_duplicate=True),
         Output("current-step", "data", allow_duplicate=True)],
        Input("delete-current-layer-btn", "n_clicks"),
        [State("architecture-layers", "data"),
         State("current-step", "data")],
        prevent_initial_call=True
    )
    def delete_current_layer(n_clicks, layers, current_step):
        if not n_clicks or not layers:
            return no_update, no_update

        step = current_step or 0
        if step >= len(layers):
            return no_update, no_update

        # Eliminar la capa
        layers.pop(step)

        # Ajustar step si es necesario
        new_step = min(step, max(0, len(layers) - 1)) if layers else 0

        return layers, new_step


    # Callback: Actualizar estado de botones de navegación
    @callback(
        [Output("prev-step-btn", "disabled"),
         Output("next-step-btn", "disabled"),
         Output("step-counter", "children")],
        [Input("architecture-layers", "data"),
         Input("current-step", "data")]
    )
    def update_navigation_state(layers, current_step):
        if not layers or len(layers) == 0:
            return True, True, ""

        step = current_step or 0
        total = len(layers)

        prev_disabled = (step == 0)
        next_disabled = (step >= total - 1)
        counter_text = f"Capa {step + 1} de {total}"

        return prev_disabled, next_disabled, counter_text


    # Callback: Navegar entre pasos
    @callback(
        Output("current-step", "data"),
        [Input("prev-step-btn", "n_clicks"),
         Input("next-step-btn", "n_clicks"),
         Input({"type": "layer-node", "index": ALL}, "n_clicks")],
        State("current-step", "data"),
        prevent_initial_call=True
    )
    def navigate_steps(prev_clicks, next_clicks, node_clicks, current_step):
        triggered = ctx.triggered_id

        if not triggered:
            return no_update

        # Click en nodo directo
        if isinstance(triggered, dict) and triggered.get("type") == "layer-node":
            return triggered["index"]

        # Botones de navegación
        if triggered == "prev-step-btn":
            return max(0, (current_step or 0) - 1)
        elif triggered == "next-step-btn":
            return (current_step or 0) + 1

        return no_update


    # Callback: Probar configuración del modelo
    @callback(
        Output({"type": "test-config-result", "model": MATCH}, "children"),
        Input({"type": "test-config-btn", "model": MATCH}, "n_clicks"),
        [State({"type": "layer-config-input", "layer_index": ALL, "field": ALL}, "value"),
         State({"type": "layer-config-input", "layer_index": ALL, "field": ALL}, "id"),
         State("architecture-layers", "data"),
         State("model-type", "data")],
        prevent_initial_call=True
    )
    def test_model_configuration(n_clicks, input_values, input_ids, layers, model_type):
        """
        Paso 1: Validar el modelo con las clases Pydantic
        Paso 2: Instanciar el modelo como nuevo experimento en Clasificación P300
        Paso 3: TODO - Mini entrenamiento para verificar que compila
        """
        if not n_clicks:
            return no_update

        from backend.classes.ClasificationModel.ClassifierSchemaFactory import ClassifierSchemaFactory
        from backend.classes.Experiment import Experiment
        from pydantic import ValidationError

        try:
            # ===== PASO 1: VALIDACIÓN =====

            # Verificar que hay capas
            if not layers or len(layers) == 0:
                return dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Error: Debes agregar al menos una capa a la arquitectura"
                ], color="danger", dismissable=True)

            # Validar arquitectura completa
            is_valid, error_msg = validate_complete_architecture(layers, model_type)
            if not is_valid:
                return dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Arquitectura incompleta: {error_msg}"
                ], color="warning", dismissable=True)

            # Construir configuración desde los inputs
            # Los input_ids son dicts con {"type": "layer-config-input", "layer_index": i, "field": "field_name"}
            config_by_layer = {}
            for input_id, value in zip(input_ids, input_values):
                layer_idx = input_id["layer_index"]
                field_name = input_id["field"]

                if layer_idx not in config_by_layer:
                    config_by_layer[layer_idx] = {}

                # Procesar valores (arrays, números, etc.)
                if isinstance(value, str) and "," in value:
                    try:
                        # Intentar parsear como lista de números
                        value = [float(v.strip()) for v in value.split(",")]
                    except (ValueError, AttributeError):
                        pass

                config_by_layer[layer_idx][field_name] = value

            # Actualizar las capas con la configuración recopilada
            for idx, layer in enumerate(layers):
                if idx in config_by_layer:
                    layer["config"] = config_by_layer[idx]

            # Construir el diccionario completo del modelo según el tipo
            model_config = build_model_config_from_layers(layers, model_type)

            # Obtener la clase Pydantic correspondiente
            available_classifiers = ClassifierSchemaFactory.available_classifiers
            classifier_class = available_classifiers.get(model_type)

            if not classifier_class:
                return dbc.Alert([
                    html.I(className="fas fa-times-circle me-2"),
                    f"Error: Modelo '{model_type}' no encontrado"
                ], color="danger", dismissable=True)

            # Validar con Pydantic
            try:
                validated_instance = classifier_class(**model_config)
                print(f"✅ Configuración válida para {model_type}: {validated_instance}")
            except ValidationError as ve:
                error_details = "\n".join([f"• {err['loc'][0]}: {err['msg']}" for err in ve.errors()])
                return dbc.Alert([
                    html.I(className="fas fa-times-circle me-2"),
                    html.Div([
                        html.Strong("Errores de validación:"),
                        html.Pre(error_details, style={"fontSize": "12px", "marginTop": "10px"})
                    ])
                ], color="danger", dismissable=True)

            # ===== PASO 2: INSTANCIAR EN EXPERIMENTO =====

            try:
                # Agregar clasificador al experimento P300
                Experiment.add_P300_classifier(validated_instance)
                print(f"✅ {model_type} agregado como P300Classifier al experimento")

                # ===== PASO 3: TODO - MINI ENTRENAMIENTO =====
                # TODO: Implementar mini-entrenamiento para verificar compilación
                # classifier_class.train(validated_instance)

                return dbc.Alert([
                    html.I(className="fas fa-check-circle me-2"),
                    html.Div([
                        html.Strong(f"✓ {model_type} configurado exitosamente"),
                        html.Br(),
                        html.Small("El modelo ha sido validado y agregado al experimento P300")
                    ])
                ], color="success", dismissable=True, duration=6000)

            except Exception as exp_err:
                print(f"❌ Error al agregar al experimento: {exp_err}")
                return dbc.Alert([
                    html.I(className="fas fa-times-circle me-2"),
                    f"Error al guardar en experimento: {str(exp_err)}"
                ], color="danger", dismissable=True)

        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            import traceback
            traceback.print_exc()
            return dbc.Alert([
                html.I(className="fas fa-times-circle me-2"),
                f"Error inesperado: {str(e)}"
            ], color="danger", dismissable=True)


# Registrar callbacks al importar
register_interactive_callbacks()
