"""
Componentes de configuración interactiva para modelos de clasificación.
Incluye representación gráfica de arquitecturas de redes neuronales.
"""

import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
from typing import Dict, Any, List, Tuple
import json

# Importar el nuevo sistema interactivo
from app.components.interactive_architecture_builder import create_interactive_config_card

# Colores para la visualización de redes
COLORS = {
    "input": "#4A90E2",
    "lstm": "#F5A623",
    "gru": "#BD10E0",
    "dense": "#50E3C2",
    "conv": "#7ED321",
    "pooling": "#D0021B",
    "flatten": "#9013FE",
    "output": "#417505"
}

def create_model_selector_card(model_name: str, model_index: int) -> dbc.Card:
    """
    Crea una card simple para seleccionar un modelo.

    Args:
        model_name: Nombre del modelo (LSTM, GRU, etc.)
        model_index: Índice del modelo en la lista
    """
    return dbc.Card([
        dbc.CardHeader(model_name.upper(), className="right-panel-card-header"),
        dbc.CardBody([
            dbc.Button(
                "Configurar",
                id={"type": "model-selector-btn", "index": model_index, "model": model_name},
                color="primary",
                className="w-100",
                style={"fontSize": "15px", "height": "42px", "fontWeight": "600"}
            )
        ])
    ], className="mb-3 right-panel-card")


def create_network_visualization(config: Dict[str, Any], model_type: str) -> html.Div:
    """
    Crea una representación gráfica de la arquitectura de red neuronal.

    Args:
        config: Configuración del modelo desde el schema
        model_type: Tipo de modelo (LSTM, GRU, CNN, SVNN)
    """
    layers = []

    if model_type == "LSTM" or model_type == "GRU":
        # Capa de entrada
        input_dim = config.get("encoder", {}).get("properties", {}).get("input_feature_dim", {})
        layers.append({
            "type": "input",
            "label": "Input",
            "info": f"Dim: {input_dim.get('description', 'N/A')}"
        })

        # Capas LSTM/GRU
        encoder_layers = config.get("encoder", {}).get("properties", {}).get("layers", {})
        layer_type = "lstm" if model_type == "LSTM" else "gru"
        layers.append({
            "type": layer_type,
            "label": f"{model_type} Encoder",
            "info": "Secuencia de capas recurrentes"
        })

        # Pooling temporal
        pooling = config.get("pooling", {})
        if pooling:
            layers.append({
                "type": "pooling",
                "label": "Temporal Pooling",
                "info": "Reducción temporal"
            })

        # Capas densas
        fc_layers = config.get("fc_layers", {})
        if fc_layers:
            layers.append({
                "type": "dense",
                "label": "Dense Layers",
                "info": "Capas completamente conectadas"
            })

        # Capa de clasificación
        layers.append({
            "type": "output",
            "label": "Classification",
            "info": "Capa de salida (softmax)"
        })

    elif model_type == "CNN":
        # Entrada
        layers.append({
            "type": "input",
            "label": "Input",
            "info": "Entrada de imágenes"
        })

        # Capas convolucionales
        feature_extractor = config.get("feature_extractor", {})
        layers.append({
            "type": "conv",
            "label": "Convolutional Layers",
            "info": "Extracción de características"
        })

        # Pooling
        layers.append({
            "type": "pooling",
            "label": "Pooling",
            "info": "Reducción espacial"
        })

        # Flatten
        layers.append({
            "type": "flatten",
            "label": "Flatten",
            "info": "Aplanamiento"
        })

        # Capas densas
        layers.append({
            "type": "dense",
            "label": "Dense Layers",
            "info": "Clasificación"
        })

        # Salida
        layers.append({
            "type": "output",
            "label": "Output",
            "info": "Clasificación final"
        })

    elif model_type == "SVNN":
        # Red simple
        layers.append({
            "type": "input",
            "label": "Input",
            "info": "Entrada de features"
        })

        layers.append({
            "type": "dense",
            "label": "Hidden Layers",
            "info": "Capas ocultas"
        })

        layers.append({
            "type": "output",
            "label": "Output",
            "info": "Clasificación"
        })

    # Crear visualización
    layer_nodes = []
    for i, layer in enumerate(layers):
        node_color = COLORS.get(layer["type"], "#999")

        node = html.Div([
            html.Div([
                html.Div(className="node-icon", style={
                    "backgroundColor": node_color,
                    "width": "60px",
                    "height": "60px",
                    "borderRadius": "50%",
                    "margin": "0 auto",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"
                }),
                html.Div(layer["label"], className="node-label", style={
                    "textAlign": "center",
                    "fontWeight": "bold",
                    "marginTop": "8px",
                    "fontSize": "12px",
                    "color": "white"
                }),
                html.Div(layer["info"], className="node-info", style={
                    "textAlign": "center",
                    "fontSize": "10px",
                    "color": "rgba(255, 255, 255, 0.6)",
                    "marginTop": "4px"
                })
            ], style={"padding": "10px"})
        ], className="network-node", style={"flex": "0 0 auto"})

        layer_nodes.append(node)

        # Agregar conexión entre nodos
        if i < len(layers) - 1:
            connector = html.Div([
                html.Div(className="connector-arrow", style={
                    "width": "40px",
                    "height": "3px",
                    "backgroundColor": "#ccc",
                    "position": "relative",
                    "margin": "30px 10px"
                }),
                html.Div(style={
                    "width": "0",
                    "height": "0",
                    "borderLeft": "8px solid #ccc",
                    "borderTop": "6px solid transparent",
                    "borderBottom": "6px solid transparent",
                    "position": "absolute",
                    "right": "-8px",
                    "top": "50%",
                    "transform": "translateY(-50%)"
                })
            ], style={"position": "relative", "flex": "0 0 auto"})
            layer_nodes.append(connector)

    return html.Div([
        html.H5("Arquitectura de Red", className="text-center mb-3", style={"color": "white"}),
        html.Div(
            layer_nodes,
            className="network-visualization",
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "overflowX": "auto",
                "padding": "20px",
                "backgroundColor": "rgba(0, 0, 0, 0.2)",
                "borderRadius": "8px",
                "minHeight": "200px",
                "border": "1px solid rgba(255, 255, 255, 0.1)"
            }
        )
    ])


def create_field_input(field_name: str, field_info: Dict[str, Any], model_name: str, parent_path: str = "") -> html.Div:
    """
    Crea un input dinámico basado en el tipo de campo del schema.

    Args:
        field_name: Nombre del campo
        field_info: Información del campo desde el schema
        model_name: Nombre del modelo
        parent_path: Ruta del padre para campos anidados
    """
    field_type = field_info.get("type", "string")
    field_label = field_info.get("title", field_name)
    field_desc = field_info.get("description", "")
    default_value = field_info.get("default")

    full_path = f"{parent_path}.{field_name}" if parent_path else field_name
    input_id = {"type": "config-input", "model": model_name, "field": full_path}

    # Detectar enums
    enum_values = field_info.get("enum")
    if enum_values:
        return html.Div([
            dbc.Label(field_label, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
            dcc.Dropdown(
                id=input_id,
                options=[{"label": str(v), "value": v} for v in enum_values],
                value=default_value,
                placeholder=f"Selecciona valor",
                style={"flex": "1", "color": "black", "fontSize": "14px"}
            )
        ], className="input-field-group")

    # Campos numéricos
    if field_type in ["integer", "number"]:
        min_val = field_info.get("minimum", field_info.get("exclusiveMinimum"))
        max_val = field_info.get("maximum")

        return html.Div([
            dbc.Label(field_label, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
            dbc.Input(
                id=input_id,
                type="number",
                min=min_val,
                max=max_val,
                value=default_value,
                step=1 if field_type == "integer" else 0.001,
                style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
            )
        ], className="input-field-group")

    # Booleanos
    if field_type == "boolean":
        return html.Div([
            dbc.Label(field_label, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
            dbc.Checkbox(
                id=input_id,
                value=default_value if default_value is not None else False,
                style={"marginTop": "10px"}
            )
        ], className="input-field-group")

    # Arrays
    if field_type == "array":
        return html.Div([
            dbc.Label(field_label, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
            dbc.Input(
                id=input_id,
                type="text",
                placeholder="Ej: 1,2,3 (separado por comas)",
                style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
            )
        ], className="input-field-group")

    # String por defecto
    return html.Div([
        dbc.Label(field_label, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
        dbc.Input(
            id=input_id,
            type="text",
            value=default_value,
            placeholder=f"Ingresa {field_label}",
            style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
        )
    ], className="input-field-group")


def create_simple_config_form(properties: Dict[str, Any], model_name: str) -> List[html.Div]:
    """
    Crea formulario simple para propiedades básicas (no anidadas).
    """
    inputs = []
    for field_name, field_info in properties.items():
        # Saltar referencias a objetos anidados
        if "$ref" in field_info:
            continue

        inputs.append(create_field_input(field_name, field_info, model_name))

    return inputs


def create_nested_object_stepper(model_name: str, schema: Dict[str, Any]) -> html.Div:
    """
    Crea un formulario por pasos para manejar objetos anidados complejos.
    """
    defs = schema.get("$defs", {})
    properties = schema.get("properties", {})

    steps = []
    step_number = 0

    # Paso 1: Configuración básica
    simple_fields = []
    for field_name, field_info in properties.items():
        if "$ref" not in field_info and field_info.get("type") != "array":
            simple_fields.append(create_field_input(field_name, field_info, model_name))

    if simple_fields:
        steps.append({
            "number": step_number,
            "title": "Configuración Básica",
            "content": simple_fields
        })
        step_number += 1

    # Paso 2+: Objetos anidados
    for field_name, field_info in properties.items():
        if "$ref" in field_info:
            ref_path = field_info["$ref"].split("/")[-1]
            ref_def = defs.get(ref_path, {})
            ref_props = ref_def.get("properties", {})

            nested_inputs = []
            for nested_field, nested_info in ref_props.items():
                nested_inputs.append(
                    create_field_input(nested_field, nested_info, model_name, field_name)
                )

            if nested_inputs:
                steps.append({
                    "number": step_number,
                    "title": field_info.get("title", field_name),
                    "content": nested_inputs
                })
                step_number += 1

    # Crear stepper UI
    stepper_content = []
    for step in steps:
        stepper_content.append(
            html.Div([
                html.Div([
                    html.Div(
                        str(step["number"] + 1),
                        className="step-number",
                        style={
                            "width": "30px",
                            "height": "30px",
                            "borderRadius": "50%",
                            "backgroundColor": "#4A90E2",
                            "color": "white",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontWeight": "bold"
                        }
                    ),
                    html.H6(step["title"], className="ms-3 mb-0", style={"color": "white"})
                ], className="d-flex align-items-center mb-3"),
                html.Div(step["content"], className="step-content ms-4 ps-3 border-start")
            ], className="stepper-step mb-4")
        )

    return html.Div(stepper_content, className="stepper-container")


def create_config_card(model_name: str, schema: Dict[str, Any], classifier_type: str = "P300") -> html.Div:
    """
    Crea la card completa de configuración para un modelo.
    Usa el sistema interactivo para redes neuronales y formulario simple para modelos clásicos.

    Args:
        model_name: Nombre del modelo (LSTM, GRU, SVM, etc.)
        schema: Schema del modelo
        classifier_type: Tipo de clasificador - "P300" o "InnerSpeech" (default: "P300")
    """
    # Determinar si es una red neuronal compleja (usa sistema interactivo)
    is_neural_network = model_name in ["LSTM", "GRU", "CNN", "SVNN"]

    if is_neural_network:
        # Usar el nuevo sistema interactivo de construcción de arquitectura
        return create_interactive_config_card(model_name, schema, classifier_type)
    else:
        # Para modelos clásicos (SVM, RandomForest), usar formulario simple
        properties = schema.get("properties", {})

        form_content = html.Div(
            create_simple_config_form(properties, model_name),
            className="simple-form"
        )

        return html.Div([
            # Store para guardar el tipo de clasificador - persiste durante la sesión
            dcc.Store(id="classifier-type-store", data=classifier_type, storage_type='session'),

            dbc.Card([
                dbc.CardHeader([
                    html.Div(f"CONFIGURACIÓN: {model_name.upper()}", className="mb-0", style={"flex": "1"}),
                    dbc.Button(
                        "← Volver",
                        id="config-back-btn",
                        color="link",
                        size="sm",
                        style={"color": "white", "textDecoration": "none"}
                    )
                ], className="right-panel-card-header d-flex justify-content-between align-items-center"),
                dbc.CardBody([
                    # Formulario de configuración
                    form_content,

                    # Botón de prueba
                    html.Div([
                        dbc.Button(
                            "Probar Configuración",
                            id={"type": "classic-test-config-btn", "model": model_name},
                            color="primary",
                            className="mt-2",
                            style={"fontSize": "15px", "height": "42px", "fontWeight": "600", "width": "100%"}
                        ),
                        html.Div(id={"type": "classic-test-config-result", "model": model_name}, className="mt-3")
                    ])
                ])
            ], className="right-panel-card")
        ], id="config-card-container")


# TODO: Callback para manejar "Probar Configuración"
# @callback(
#     Output(...),
#     Input({"type": "test-config-btn", "model": MATCH}, "n_clicks"),
#     State({"type": "config-input", "model": MATCH, "field": ALL}, "value"),
#     prevent_initial_call=True
# )
# def test_configuration(n_clicks, values):
#     """
#     Callback para probar la configuración del modelo.
#     Aquí se validaría y ejecutaría el modelo con la configuración proporcionada.
#     """
#     pass


from dash import no_update, ctx
import dash_bootstrap_components as dbc
from dash import html

# ------------------------------------------------------------
# Utilidades internas para reconstruir config y castear valores
# ------------------------------------------------------------
def _coerce_value(v):
    # Mantén None/booleanos tal cual
    if v is None or isinstance(v, bool):
        return v

    # Si ya es numérico/lista, respeta
    if isinstance(v, (int, float, list, dict)):
        return v

    if isinstance(v, str):
        s = v.strip()

        # Bool por string
        if s.lower() in ("true", "false"):
            return s.lower() == "true"

        # Lista separada por comas -> intenta numérico
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            casted = []
            for p in parts:
                try:
                    # int si aplica, si no float, si no deja string
                    casted.append(int(p))
                except ValueError:
                    try:
                        casted.append(float(p))
                    except ValueError:
                        casted.append(p)
            return casted

        # Número único
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s

    # Cualquier otro tipo lo regresamos tal cual
    return v


def _set_deep(dct, dotted_path, value):
    """
    Inserta value en dct siguiendo una ruta con puntos, por ejemplo:
    dotted_path = "encoder.input_feature_dim" -> dct["encoder"]["input_feature_dim"] = value
    """
    parts = dotted_path.split(".")
    cur = dct
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


# ------------------------------------------------------------
# Callback para "modelos simples" (SVM, RandomForest, etc.)
# ------------------------------------------------------------
@callback(
    Output({"type": "classic-test-config-result", "model": MATCH}, "children"),
    Input({"type": "classic-test-config-btn", "model": MATCH}, "n_clicks"),
    [
        State({"type": "config-input", "model": MATCH, "field": ALL}, "value"),
        State({"type": "config-input", "model": MATCH, "field": ALL}, "id"),
        State("classifier-type-store", "data"),
    ],
    prevent_initial_call=True
)
def test_classic_model_configuration(n_clicks, input_values, input_ids, classifier_type):
    """
    Valida y registra modelos NO-NN (SVM, RandomForest, etc.) usando los inputs simples.
    Este callback se auto-omite si no hay `config-input` (lo que ocurre en las NN).

    Args:
        classifier_type: "P300" o "InnerSpeech" - determina qué método del experimento usar
    """
    if not n_clicks:
        return no_update

    # Si no hay inputs 'config-input', asumimos que no es una card clásica -> omitir
    if not input_ids or len(input_ids) == 0:
        return no_update

    # Determinar el nombre del modelo desde los IDs
    # (cualquier id trae {"model": "<Modelo>"})
    model_name = input_ids[0].get("model", "UNKNOWN")

    # Reconstruir configuración a partir de full_path en 'field'
    # Ej: field = "hyperparams.C"  -> {"hyperparams": {"C": <valor>}}
    config = {}
    for id_obj, raw_value in zip(input_ids, input_values):
        dotted_path = id_obj.get("field", "")
        value = _coerce_value(raw_value)
        if dotted_path:
            _set_deep(config, dotted_path, value)

    # Validar contra la clase Pydantic y registrar en el experimento
    try:
        from backend.classes.ClasificationModel.ClassifierSchemaFactory import ClassifierSchemaFactory
        from pydantic import ValidationError

        classifier_class = ClassifierSchemaFactory.available_classifiers.get(model_name)
        if not classifier_class:
            return dbc.Alert(
                [
                    html.I(className="fas fa-times-circle me-2"),
                    f"Error: Modelo '{model_name}' no está registrado en ClassifierSchemaFactory"
                ],
                color="danger",
                dismissable=True
            )

        try:
            validated_instance = classifier_class(**config)
        except ValidationError as ve:
            # Formatear errores de Pydantic
            err_lines = []
            for e in ve.errors():
                loc = ".".join(str(x) for x in e.get("loc", []))
                msg = e.get("msg", "Error")
                err_lines.append(f"• {loc}: {msg}")
            return dbc.Alert(
                [
                    html.I(className="fas fa-times-circle me-2"),
                    html.Div([
                        html.Strong("Errores de validación:"), html.Br(),
                        html.Pre("\n".join(err_lines), style={"fontSize": "12px", "marginTop": "8px"})
                    ])
                ],
                color="danger",
                dismissable=True
            )

        # Agregar al experimento según el tipo de clasificador
        try:
            from backend.classes.Experiment import Experiment

            # Usar el método correcto según el tipo
            if classifier_type == "InnerSpeech":
                Experiment.add_inner_speech_classifier(validated_instance)
                success_msg = "El modelo ha sido validado y agregado al experimento de Habla Interna"
            else:  # P300 por defecto
                Experiment.add_P300_classifier(validated_instance)
                success_msg = "El modelo ha sido validado y agregado al experimento P300"

        except Exception as exp_err:
            return dbc.Alert(
                [
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Configuración válida, pero no se pudo registrar en el experimento: {exp_err}"
                ],
                color="warning",
                dismissable=True
            )

        # Éxito
        return dbc.Alert(
            [
                html.I(className="fas fa-check-circle me-2"),
                html.Div([
                    html.Strong(f"✓ {model_name} configurado exitosamente"),
                    html.Br(),
                    html.Small(success_msg)
                ])
            ],
            color="success",
            dismissable=True,
            duration=6000
        )

    except Exception as e:
        return dbc.Alert(
            [
                html.I(className="fas fa-times-circle me-2"),
                f"Error inesperado: {str(e)}"
            ],
            color="danger",
            dismissable=True
        )
