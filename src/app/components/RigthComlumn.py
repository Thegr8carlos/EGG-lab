import dash_bootstrap_components as dbc 
from dash import html, dcc
from backend.classes.Filter.FilterSchemaFactory import FilterSchemaFactory, filterCallbackRegister
from backend.classes.FeatureExtracture.TransformSchemaFactory import TransformSchemaFactory, TransformCallbackRegister
from backend.classes.ClasificationModel.ClassifierSchemaFactory import ClassifierSchemaFactory, ClassifierCallbackRegister
from backend.helpers.mapaValidacion import generar_mapa_validacion_inputs


# First, we define a mapping from field names to more user-friendly Spanish names.
# If we update the models, we only need to update this dictionary for display names.
NOMBRE_CAMPOS_ES = {
    # General
    "sp": "Frecuencia de Muestreo",
    "epochs": "Épocas",
    "batch_size": "Tamaño de lote",

    # Filters and Feature Extractors
    "numeroComponentes": "Número de componentes",
    "method": "Método",
    "random_state": "Semilla aleatoria",
    "max_iter": "Iteraciones máximas",
    "wavelet": "Tipo de Wavelet",
    "level": "Nivel de descomposición",
    "mode": "Modo de borde",
    "threshold": "Umbral (denoising)",
    "frame_length": "Tamaño de ventana",
    "hop_samples": "Salto entre ventanas",
    "filter_type": "Tipo de filtro",
    "freq": "Frecuencia de corte",
    "freqs": "Frecuencias",
    "order": "Orden del filtro",
    "phase": "Tipo de fase",
    "fir_window": "Ventana FIR",
    "quality": "Calidad",
    "window": "Ventana de análisis",
    "nfft": "Tamaño de FFT",
    "overlap": "Solapamiento",
    "type": "Tipo de DCT",
    "norm": "Normalización",
    "axis": "Eje",

    # Classifiers
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

    ## Add more mappings as needed
    #...
}



def build_configuration_ui(schema: dict):
    # In this variable, we will store the Forms components
    components = []
    # Each schema attribute has properties and a type
    properties = schema.get("properties", {})
    type = schema.get("title", schema["type"])

    for field_name, field_info in properties.items():

        showName = NOMBRE_CAMPOS_ES.get(field_name, field_info.get("title", field_name))
        ## We process each field according to its type

        # Detectar enums (puede venir como "enum" directo o dentro de "anyOf" con "const")
        enum_values = None
        if "enum" in field_info:
            enum_values = field_info["enum"]
        elif "anyOf" in field_info:
            # Buscar si todos los anyOf son const (indica Literal de Pydantic)
            consts = [x.get("const") for x in field_info["anyOf"] if "const" in x]
            if consts and len(consts) == len([x for x in field_info["anyOf"] if "const" in x or x.get("type") == "null"]):
                # Filtrar None/null de las opciones
                enum_values = [c for c in consts if c is not None]

        # Case "enum" (directo o detectado desde anyOf)
        if enum_values:
            input_component = html.Div([
                dbc.Label(showName, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dcc.Dropdown(
                    id=f"{type}-{field_name}",
                    options=[{"label": str(val), "value": val} for val in enum_values],
                    placeholder=f"Selecciona valor",
                    style={"flex": "1", "color": "black", "fontSize": "14px"}
                )
            ], className="input-field-group")

        # Case "anyOf"
        elif "anyOf" in field_info:
            posibles_tipos = {x.get("type") for x in field_info["anyOf"] if "type" in x}

            inputType = "text"
            inputAtributes = {}
            placeholder_text = f"Ingresa {showName}"

            # Verificar si anyOf incluye tanto number como array
            has_number = any(x.get("type") in ["number", "integer"] for x in field_info["anyOf"])
            has_array = any(x.get("type") == "array" for x in field_info["anyOf"])

            if has_number and has_array:
                # Caso especial: puede ser número O array (ej: freq en BandPass)
                inputType = "text"
                placeholder_text = f"Ej: 30 (un valor) o 1,30 (dos valores separados por coma)"
            elif has_number:
                # Solo número
                inputType = "number"
                for tipo_Field in field_info["anyOf"]:
                    if tipo_Field.get("type") in ["number", "integer"]:
                        if "minimum" in tipo_Field:
                            inputAtributes["min"] = tipo_Field["minimum"]
                        if "maximum" in tipo_Field:
                            inputAtributes["max"] = tipo_Field["maximum"]
                        if "default" in field_info:
                            inputAtributes["value"] = field_info["default"]

            input_component = html.Div([
                dbc.Label(showName, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dbc.Input(
                    type=inputType,
                    id=f"{type}-{field_name}",
                    placeholder=placeholder_text,
                    style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"},
                    **inputAtributes
                )
            ], className="input-field-group")


        # Case "array"
        elif field_info.get("type") == "array":
            input_component = html.Div([
                dbc.Label(showName, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dbc.Input(
                    type="text",  # se podría transformar a varios inputs si lo deseas
                    id=f"{type}-{field_name}",
                    placeholder=f"Ingresa lista separada por comas",
                    style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
                )
            ], className="input-field-group")

        # Case "normal":
        else:
            tipo_dash = {
                "number": "number",
                "integer": "number",
                "string": "text",
            }.get(field_info.get("type", "string"), "text")

            input_component = html.Div([
                dbc.Label(showName, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dbc.Input(
                    type=tipo_dash,
                    id=f"{type}-{field_name}",
                    placeholder=f"Ingresa {showName}",
                    style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
                )
            ], className="input-field-group")

        components.append(input_component)

    # Botón de aplicar
    components.append(
        dbc.Button("Aplicar", color="primary", id=f"btn-aplicar-{type}", className="mt-2", style={"fontSize": "15px", "height": "42px", "fontWeight": "600"})
    )

    return dbc.Card([
        dbc.CardHeader(str(type).upper(), className="right-panel-card-header"),
        dbc.CardBody(components)
    ], className="mb-3 right-panel-card")



def get_rightColumn(window: str):
    """
    Construye la columna lateral según el type de ventana (filter, extractores, etc.)
    """
    title = ""
    if window == "filter":
        all_schemas = FilterSchemaFactory.get_all_filter_schemas()
        title = "Filtros"
    elif window == "featureExtracture":

        all_schemas = TransformSchemaFactory.get_all_transform_schemas() 
        title = "Extractores de características"
    elif window == "clasificationModelsP300":
        all_schemas = _tag_classifier_schemas("_p300")
        title = "Modelos de clasificación"

    elif window == "clasificationModelsInner":
        all_schemas = _tag_classifier_schemas("_inner")
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








#  This function is a local helper. 
def _tag_classifier_schemas(type_suffix: str):
    schemas = ClassifierSchemaFactory.get_all_classifier_schemas()
    for s in schemas.values():
        base = s.get("title", s["type"])
        
        s["title"] = f"{base}{type_suffix}"  # p.ej., "LSTM_p300" o "LSTM_inner"
    
    return schemas











##--------------------------------------Callbacks------------------------------------------------------
# In this section, we registrate the callbacks.
# in each case, from the factory, we redirect the callback response 
# to the specific function that will handle the form submission.
# Specifically, each child class has its own fuction to do changes in the signal.
# On the other hand,  Experiments management is done in the factories.

 ## Transforms
for grupo in generar_mapa_validacion_inputs(TransformSchemaFactory.get_all_transform_schemas()):
    for boton_id, inputs_map in grupo.items():
        TransformCallbackRegister(boton_id, inputs_map)

#Filters
for grupo in generar_mapa_validacion_inputs(FilterSchemaFactory.get_all_filter_schemas()):
    for boton_id, inputs_map in grupo.items():
        filterCallbackRegister(boton_id, inputs_map)

# Callbacks de clasificadores movidos a modelado_p300.py para configuración interactiva

#--------------------------------------------------------------------------------------------------------