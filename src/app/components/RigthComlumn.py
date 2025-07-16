import dash_bootstrap_components as dbc 
from dash import html, dcc
from backend.classes.filter import FilterSchemaFactory
from backend.classes.featureExtracture import TransformSchemaFactory
from backend.classes.clsificationModels import ClassifierSchemaFactory
NOMBRE_CAMPOS_ES = {
    # Común
    "sp": "Frecuencia de Muestreo",
    "epochs": "Épocas",
    "batch_size": "Tamaño de lote",

    # Filtros / Transformadas
    "numeroComponentes": "Número de componentes",
    "method": "Método",
    "random_state": "Semilla aleatoria",
    "max_iter": "Iteraciones máximas",
    "wavelet": "Wavelet",
    "level": "Nivel",
    "mode": "Modo",
    "threshold": "Umbral",
    "filter_type": "Tipo de filtro",
    "freq": "Frecuencia",
    "freqs": "Frecuencias",
    "order": "Orden del filtro",
    "phase": "Fase",
    "fir_window": "Ventana FIR",
    "quality": "Calidad",
    "window": "Ventana de análisis",
    "nfft": "Tamaño de FFT",
    "overlap": "Solapamiento",
    "type": "Tipo de DCT",
    "norm": "Normalización",
    "axis": "Eje",

    # Clasificadores
    "hidden_size": "Tamaño oculto",
    "num_layers": "Número de capas",
    "bidirectional": "Bidireccional",
    "dropout": "Dropout",
    "learning_rate": "Tasa de aprendizaje",
    "kernel": "Kernel",
    "C": "Coeficiente C",
    "gamma": "Gamma",
    "n_estimators": "Número de árboles",
    "max_depth": "Profundidad máxima",
    "criterion": "Criterio",
    "num_filters": "Número de filtros",
    "kernel_size": "Tamaño del kernel",
    "pool_size": "Tamaño de pooling"
}



def build_configuration_ui(schema: dict):
    components = []

    propiedades = schema.get("properties", {})
    tipo = schema.get("title", schema["type"])

    for field_name, field_info in propiedades.items():
        nombre_mostrado = NOMBRE_CAMPOS_ES.get(field_name, field_info.get("title", field_name))

        # --- Detectar si hay enum (selector) ---
        if "enum" in field_info:
            input_component = html.Div([
                dbc.Label(nombre_mostrado, html_for=field_name, style={"minWidth": "140px", "color": "white"}),
                dcc.Dropdown(
                    id=f"{schema['type']}-{field_name}",
                    options=[{"label": str(val), "value": val} for val in field_info["enum"]],
                    placeholder=f"Selecciona valor",
                    style={"flex": "1", "color": "black"}
                )
            ], className="input-field-group")

        # --- Si es combinación de tipos (anyOf), tratamos de deducir ---
        elif "anyOf" in field_info:
            posibles_tipos = {x.get("type") for x in field_info["anyOf"] if "type" in x}
            if "number" in posibles_tipos or "integer" in posibles_tipos:
                tipo_input = "number"
            else:
                tipo_input = "text"

            input_component = html.Div([
                dbc.Label(nombre_mostrado, html_for=field_name, style={"minWidth": "140px", "color": "white"}),
                dbc.Input(
                    type=tipo_input,
                    id=f"{schema['type']}-{field_name}",
                    placeholder=f"Ingresa {nombre_mostrado}",
                    style={"flex": "1"}
                )
            ], className="input-field-group")

        # --- Si es tipo array (de número por ejemplo) ---
        elif field_info.get("type") == "array":
            input_component = html.Div([
                dbc.Label(nombre_mostrado, html_for=field_name, style={"minWidth": "140px", "color": "white"}),
                dbc.Input(
                    type="text",  # se podría transformar a varios inputs si lo deseas
                    id=f"{schema['type']}-{field_name}",
                    placeholder=f"Ingresa lista separada por comas",
                    style={"flex": "1"}
                )
            ], className="input-field-group")

        # --- Caso normal: tipo string, number, etc. ---
        else:
            tipo_dash = {
                "number": "number",
                "integer": "number",
                "string": "text",
            }.get(field_info.get("type", "string"), "text")

            input_component = html.Div([
                dbc.Label(nombre_mostrado, html_for=field_name, style={"minWidth": "140px", "color": "white"}),
                dbc.Input(
                    type=tipo_dash,
                    id=f"{schema['type']}-{field_name}",
                    placeholder=f"Ingresa {nombre_mostrado}",
                    style={"flex": "1"}
                )
            ], className="input-field-group")

        components.append(input_component)

    # Botón de aplicar
    components.append(
        dbc.Button("Aplicar", color="primary", id=f"btn-aplicar-{schema['type']}", className="mt-2")
    )

    return dbc.Card([
        dbc.CardHeader(tipo.upper(), className="right-panel-card-header"),
        dbc.CardBody(components)
    ], className="mb-3 right-panel-card")



def get_rightColumn(window: str):
    """
    Construye la columna lateral según el tipo de ventana (filter, extractores, etc.)
    """
    title = ""
    if window == "filter":
        all_schemas = FilterSchemaFactory.get_all_filter_schemas()
        title = "Filtros"
    elif window == "featureExtracture":
        # Asumimos que hay otra fábrica similar para extractores
        all_schemas = TransformSchemaFactory.get_all_transform_schemas()  # FeatureExtractorFactory.get_all_schemas()
        title = "Extractores de características"
    elif window == "clasificationModels":
        # Asumimos que hay otra fábrica similar para extractores
        all_schemas = ClassifierSchemaFactory.get_all_classifier_schemas()  # FeatureExtractorFactory.get_all_schemas()
        title = "Modelos de clasificación"
    else:
        return dbc.Alert("Ventana no soportada", color="warning")

    cards = []
    for filter_type, schema in all_schemas.items():
        cards.append(build_configuration_ui(schema))
    divReturn = html.Div([
        html.H2(title, className="right-panel-title"),
        *cards
    ], className="right-panel-container")
    return divReturn

