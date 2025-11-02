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

        # ✅ Caso especial: campo "wavelet" - generar dropdown dinámicamente
        # Diferencia entre WaveletsBase (filtro, 16 opciones) y WaveletTransform (60+ opciones)
        if field_name == "wavelet" and field_info.get("type") == "string":
            try:
                import pywt

                # 🔍 Detectar si es WaveletsBase (filtro) o WaveletTransform (transformada)
                is_filter = "WaveletsBase" in type

                if is_filter:
                    # WaveletsBase: Solo wavelets válidos según el Literal del modelo
                    valid_wavelets = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db8',
                                    'sym2', 'sym3', 'sym4', 'sym5',
                                    'coif1', 'coif2', 'coif3', 'coif5', 'haar']
                    dropdown_options = [{"label": w, "value": w} for w in valid_wavelets]
                else:
                    # WaveletTransform: Catálogo completo de wavelets
                    wavelet_families = {
                        "Daubechies": [f"db{i}" for i in range(1, 39)],
                        "Symlets": [f"sym{i}" for i in range(2, 21)],
                        "Coiflets": [f"coif{i}" for i in range(1, 18)],
                        "Biorthogonal": [f"bior{i}.{j}" for i in range(1, 7) for j in [1, 3, 5, 7, 9] if f"bior{i}.{j}" in pywt.wavelist(kind='discrete')],
                        "Reverse biorthogonal": [f"rbio{i}.{j}" for i in range(1, 7) for j in [1, 3, 5, 7, 9] if f"rbio{i}.{j}" in pywt.wavelist(kind='discrete')],
                        "Discrete Meyer": ["dmey"],
                        "Haar": ["haar"],
                    }

                    dropdown_options = []
                    for family, wavelets in wavelet_families.items():
                        for w in wavelets:
                            dropdown_options.append({"label": w, "value": w})

                input_component = html.Div([
                    dbc.Label(showName, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                    dcc.Dropdown(
                        id=f"{type}-{field_name}",
                        options=dropdown_options,
                        value="db4",  # Default
                        placeholder=f"Selecciona wavelet",
                        style={"flex": "1", "color": "black", "fontSize": "14px"}
                    )
                ], className="input-field-group")
                components.append(input_component)
                continue
            except Exception as e:
                print(f"⚠️ No se pudo generar dropdown de wavelets: {e}")
                # Continuar con lógica normal si falla

        # ✅ Caso especial: campo "window" (para FFT/DCT/etc.) - generar dropdown
        # Detecta tanto type=string como anyOf con string (Optional[Literal[...]])
        is_window_field = (field_name == "window" and
                          (field_info.get("type") == "string" or
                           ("anyOf" in field_info and any(x.get("type") == "string" for x in field_info.get("anyOf", [])))))

        if is_window_field:
            # Intentar obtener opciones desde anyOf primero
            window_options = []
            if "anyOf" in field_info:
                for item in field_info["anyOf"]:
                    if "const" in item and item["const"] is not None:
                        window_options.append(item["const"])

            # Si no se encontraron opciones, usar las por defecto
            if not window_options:
                window_options = ["hann", "hamming", "blackman", "rectangular", "bartlett", "kaiser"]

            default_window = field_info.get("default", window_options[0] if window_options else "hann")

            input_component = html.Div([
                dbc.Label(showName, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dcc.Dropdown(
                    id=f"{type}-{field_name}",
                    options=[{"label": w, "value": w} for w in window_options],
                    value=default_window,
                    placeholder=f"Selecciona ventana",
                    style={"flex": "1", "color": "black", "fontSize": "14px"}
                )
            ], className="input-field-group")
            components.append(input_component)
            continue

        # ✅ Caso especial: campo "mode" - generar dropdown (Wavelets, DCT, etc.)
        # Puede ser type="string" o anyOf con string (cuando es Optional[str])
        is_mode_field = (field_name == "mode" and
                        (field_info.get("type") == "string" or
                         ("anyOf" in field_info and any(x.get("type") == "string" for x in field_info.get("anyOf", [])))))

        if is_mode_field:
            mode_options = ["symmetric", "periodization", "reflect", "antireflect", "zero", "constant"]
            default_mode = field_info.get("default", "symmetric")

            input_component = html.Div([
                dbc.Label(showName, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dcc.Dropdown(
                    id=f"{type}-{field_name}",
                    options=[{"label": m, "value": m} for m in mode_options],
                    value=default_mode,
                    placeholder=f"Selecciona modo",
                    style={"flex": "1", "color": "black", "fontSize": "14px"}
                )
            ], className="input-field-group")
            components.append(input_component)
            continue

        # ✅ Caso especial: campo "type" (para DCT) - generar dropdown
        # Detecta tanto type=integer como anyOf con integer (Optional[Literal[1,2,3,4]])
        is_type_field_dct = (field_name == "type" and "dct" in type.lower() and
                            (field_info.get("type") == "integer" or
                             ("anyOf" in field_info and any(x.get("type") == "integer" for x in field_info.get("anyOf", [])))))

        if is_type_field_dct:
            type_options = [1, 2, 3, 4]
            default_type = field_info.get("default", 2)

            input_component = html.Div([
                dbc.Label(showName, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dcc.Dropdown(
                    id=f"{type}-{field_name}",
                    options=[{"label": f"Tipo {t}", "value": t} for t in type_options],
                    value=default_type,
                    placeholder=f"Selecciona tipo DCT",
                    style={"flex": "1", "color": "black", "fontSize": "14px"}
                )
            ], className="input-field-group")
            components.append(input_component)
            continue

        # ✅ Caso especial: campo "norm" - dropdown con None (para DCT, FFT, etc.)
        if field_name == "norm":
            input_component = html.Div([
                dbc.Label(showName, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dcc.Dropdown(
                    id=f"{type}-{field_name}",
                    options=[
                        {"label": "ortho", "value": "ortho"},
                        {"label": "None", "value": "None"}
                    ],
                    value="None",
                    placeholder=f"Selecciona normalización",
                    style={"flex": "1", "color": "black", "fontSize": "14px"}
                )
            ], className="input-field-group")
            components.append(input_component)
            continue

        # Detectar enums (puede venir como "enum" directo o dentro de "anyOf" con "const")
        enum_values = None
        if "enum" in field_info:
            enum_values = field_info["enum"]
        elif "anyOf" in field_info:
            # Buscar si todos los anyOf son const (indica Literal de Pydantic)
            consts = [x.get("const") for x in field_info["anyOf"] if "const" in x]
            if consts and len(consts) == len([x for x in field_info["anyOf"] if "const" in x or x.get("type") == "null"]):
                # MANTENER los valores None/null - se procesarán después
                enum_values = consts

        # Case "enum" (directo o detectado desde anyOf)
        if enum_values:
            # Obtener valor por defecto si existe
            default_value = field_info.get("default")

            # Filtrar valores None/null de las opciones (Dash Dropdown no los acepta)
            # Si hay None, agregarlo como string "None" para que sea seleccionable
            dropdown_options = []
            has_none = False
            for val in enum_values:
                if val is None:
                    has_none = True
                else:
                    dropdown_options.append({"label": str(val), "value": val})

            # Si había None en las opciones, agregar como string "None"
            if has_none:
                dropdown_options.append({"label": "None", "value": "None"})

            input_component = html.Div([
                dbc.Label(showName, html_for=field_name, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dcc.Dropdown(
                    id=f"{type}-{field_name}",
                    options=dropdown_options,
                    value=default_value if default_value is not None else None,
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

# --- Clasificadores: dos for separados ---
for grupo in generar_mapa_validacion_inputs(_tag_classifier_schemas("_p300")):
    #print(_tag_classifier_schemas("_p300"))
    for boton_id, inputs_map in grupo.items():
        ClassifierCallbackRegister(boton_id, inputs_map)

for grupo in generar_mapa_validacion_inputs(_tag_classifier_schemas("_inner")):
    for boton_id, inputs_map in grupo.items():
        ClassifierCallbackRegister(boton_id, inputs_map)

#--------------------------------------------------------------------------------------------------------