"""
Sistema interactivo de construcción de arquitecturas de redes neuronales.
Permite al usuario agregar capas dinámicamente y configurarlas paso a paso.
"""

import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, no_update, ctx
import dash_bootstrap_components as dbc
from typing import Dict, Any, List, Tuple, Optional
import json
import time
from pathlib import Path
import numpy as np

# Importar editor de kernels de CNN
from app.components.cnn_kernel_editor import create_convolution_layer_config
from app.components.CloudTrainingComponent import create_cloud_training_section
from app.components.LocalTrainingComponent import create_local_training_section

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
            return False, "❌ La primera capa debe ser LSTM"
        if model_type == "GRU" and new_layer_type != "GRULayer":
            return False, "❌ La primera capa debe ser GRU"
        if model_type == "CNN" and new_layer_type != "ConvolutionLayer":
            return False, "❌ La primera capa debe ser Convolucional"
        # SVNN puede empezar con DenseLayer directamente
        return True, ""

    last_layer = current_layers[-1]["type"]

    # ========== REGLAS ESPECÍFICAS PARA CNN ==========
    if model_type == "CNN":
        # Verificar si ya hay capas densas
        has_dense = any(layer["type"] == "DenseLayer" for layer in current_layers)

        # Regla 1: NO se pueden agregar Conv/Pool después de DenseLayer
        if has_dense and new_layer_type in ["ConvolutionLayer", "PoolingLayer"]:
            return False, "❌ No puedes agregar capas Conv/Pool después de capas Dense. Orden: Conv/Pool → Dense"

        # Regla 2: En fase de feature extraction, no se puede agregar Dense sin antes tener Conv
        if not has_dense and new_layer_type == "DenseLayer":
            has_conv = any(layer["type"] == "ConvolutionLayer" for layer in current_layers)
            if not has_conv:
                return False, f"❌ En CNN primero debes agregar al menos una capa Convolucional antes de Dense"

        # Regla 3: No se puede agregar Pool como primera capa
        if len(current_layers) == 0 and new_layer_type == "PoolingLayer":
            return False, "❌ No puedes comenzar con PoolingLayer. Primero agrega ConvolutionLayer"

    # ========== REGLAS ESPECÍFICAS PARA LSTM/GRU ==========
    if model_type in ["LSTM", "GRU"]:
        main_layer_type = "LSTMLayer" if model_type == "LSTM" else "GRULayer"
        has_dense = any(layer["type"] == "DenseLayer" for layer in current_layers)
        has_pooling = any(layer["type"] == "TemporalPooling" for layer in current_layers)

        # Regla 1: NO se pueden agregar más LSTM/GRU después de DenseLayer
        if has_dense and new_layer_type == main_layer_type:
            return False, f"❌ No puedes agregar {main_layer_type} después de Dense. Orden: {main_layer_type} → TemporalPooling → Dense"

        # Regla 2: NO se pueden agregar más LSTM/GRU después de TemporalPooling
        if has_pooling and new_layer_type == main_layer_type:
            return False, f"❌ No puedes agregar {main_layer_type} después de TemporalPooling"

        # Regla 3: TemporalPooling debe estar después de LSTM/GRU
        if new_layer_type == "TemporalPooling":
            has_main_layer = any(layer["type"] == main_layer_type for layer in current_layers)
            if not has_main_layer:
                return False, f"❌ TemporalPooling debe estar después de al menos una capa {main_layer_type}"

    # ========== REGLAS ESPECÍFICAS PARA SVNN ==========
    if model_type == "SVNN":
        # SVNN acepta DenseLayer, Dropout y BatchNorm
        allowed_svnn_layers = ["DenseLayer", "Dropout", "BatchNorm"]
        if new_layer_type not in allowed_svnn_layers:
            return False, f"❌ SVNN solo acepta capas Dense, Dropout y BatchNorm. No puedes agregar {LAYER_NAMES.get(new_layer_type, new_layer_type)}"

        # SVNN debe empezar con DenseLayer
        if not current_layers and new_layer_type != "DenseLayer":
            return False, f"❌ SVNN debe comenzar con una capa Dense"

    # ========== REGLAS GENERALES ==========
    # Regla: No se puede agregar Dropout o BatchNorm consecutivamente
    if new_layer_type == "Dropout" and last_layer == "Dropout":
        return False, "❌ No puedes agregar Dropout después de otro Dropout"

    if new_layer_type == "BatchNorm" and last_layer == "BatchNorm":
        return False, "❌ No puedes agregar BatchNorm después de otro BatchNorm"

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
        return False, "❌ La arquitectura está vacía. Agrega al menos una capa."

    # ========== VALIDACIONES ESPECÍFICAS POR MODELO ==========

    if model_type == "CNN":
        # Debe tener al menos una capa convolucional
        has_conv = any(layer["type"] == "ConvolutionLayer" for layer in layers)
        if not has_conv:
            return False, "❌ CNN debe contener al menos una capa Convolucional"

        # Validar orden: feature_extraction → fc_layers
        # Feature extraction: solo Conv y Pool
        # FC layers: solo Dense
        in_feature_extraction = True
        for i, layer in enumerate(layers):
            layer_type = layer["type"]

            if layer_type == "DenseLayer":
                in_feature_extraction = False

            # Si ya salimos de feature extraction, no se permiten Conv/Pool
            if not in_feature_extraction and layer_type in ["ConvolutionLayer", "PoolingLayer"]:
                return False, f"❌ Orden incorrecto: No puedes tener Conv/Pool después de Dense. Capa {i+1}: {layer_type}"

        # Validar que ConvolutionLayer tenga filtros configurados
        for i, layer in enumerate(layers):
            if layer["type"] == "ConvolutionLayer":
                config = layer.get("config", {})
                filters = config.get("filters", [])
                if not filters or len(filters) == 0:
                    return False, f"❌ ConvolutionLayer #{i+1} no tiene filtros configurados. Agrega al menos un filtro."

    if model_type == "LSTM":
        # Debe tener al menos una capa LSTM
        has_lstm = any(layer["type"] == "LSTMLayer" for layer in layers)
        if not has_lstm:
            return False, "❌ LSTM debe contener al menos una capa LSTMLayer"

        # Validar orden: LSTM → [TemporalPooling] → Dense
        # NO se permite LSTM después de Dense o TemporalPooling
        has_pooling = False
        has_dense = False
        for i, layer in enumerate(layers):
            layer_type = layer["type"]

            if layer_type == "TemporalPooling":
                has_pooling = True
            elif layer_type == "DenseLayer":
                has_dense = True
            elif layer_type == "LSTMLayer":
                if has_pooling or has_dense:
                    return False, f"❌ Orden incorrecto: No puedes agregar LSTMLayer después de TemporalPooling o Dense. Capa {i+1}"

    if model_type == "GRU":
        # Debe tener al menos una capa GRU
        has_gru = any(layer["type"] == "GRULayer" for layer in layers)
        if not has_gru:
            return False, "❌ GRU debe contener al menos una capa GRULayer"

        # Validar orden: GRU → [TemporalPooling] → Dense
        has_pooling = False
        has_dense = False
        for i, layer in enumerate(layers):
            layer_type = layer["type"]

            if layer_type == "TemporalPooling":
                has_pooling = True
            elif layer_type == "DenseLayer":
                has_dense = True
            elif layer_type == "GRULayer":
                if has_pooling or has_dense:
                    return False, f"❌ Orden incorrecto: No puedes agregar GRULayer después de TemporalPooling o Dense. Capa {i+1}"

    if model_type == "SVNN":
        # SVNN debe tener al menos una DenseLayer y solo acepta Dense, Dropout, BatchNorm
        has_dense = any(layer["type"] == "DenseLayer" for layer in layers)
        if not has_dense:
            return False, "❌ SVNN debe contener al menos una capa Dense"

        # Validar que solo tenga capas permitidas
        allowed_svnn_layers = ["DenseLayer", "Dropout", "BatchNorm"]
        for i, layer in enumerate(layers):
            if layer["type"] not in allowed_svnn_layers:
                return False, f"❌ SVNN solo acepta capas Dense, Dropout y BatchNorm. Capa {i+1} es: {layer['type']}"

    # ========== VALIDACIONES GENERALES ==========

    # No puede terminar con Dropout o BatchNorm
    last_layer = layers[-1]["type"]
    if last_layer in ["Dropout", "BatchNorm"]:
        return False, "❌ La arquitectura no puede terminar con Dropout o BatchNorm. Agrega una capa final."

    return True, ""


def build_model_config_from_layers(
    layers: List[Dict],
    model_type: str,
    num_channels: int,
    num_classes: int,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    classifier_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Construye el diccionario de configuración completo del modelo desde las capas.

    Args:
        layers: Lista de capas con sus configuraciones
        model_type: Tipo de modelo (LSTM, GRU, CNN, SVNN)
        num_channels: Número de canales de entrada (requerido)
        num_classes: Número de clases de salida (requerido)
        epochs: Número de épocas (opcional, del usuario)
        batch_size: Tamaño de batch (opcional, del usuario)
        learning_rate: Tasa de aprendizaje (opcional, del usuario)
        classifier_type: Tipo de clasificador - "P300" o "InnerSpeech" (opcional)

    Returns:
        Diccionario con la configuración completa del modelo en formato Pydantic

    Raises:
        ValueError: Si num_channels o num_classes no son proporcionados
    """
    if num_channels is None or num_channels <= 0:
        raise ValueError("num_channels es requerido y debe ser mayor a 0")
    if num_classes is None or num_classes <= 0:
        raise ValueError("num_classes es requerido y debe ser mayor a 0")
    
    # ✅ VALIDAR HIPERPARÁMETROS (sin defaults, solo validación estricta)
    if epochs is None or epochs <= 0:
        raise ValueError("epochs es requerido y debe ser mayor a 0")
    if batch_size is None or batch_size <= 0:
        raise ValueError("batch_size es requerido y debe ser mayor a 0")
    if learning_rate is None or learning_rate <= 0 or learning_rate > 1.0:
        raise ValueError("learning_rate es requerido y debe estar entre 0 y 1")

    config = {}
    
    print(f"🎛️ Hiperparámetros validados: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    if model_type in ["LSTM", "GRU"]:
        # Importar clases Pydantic para instanciación
        if model_type == "LSTM":
            from backend.classes.ClasificationModel.LSTM import (
                LSTMLayer, SequenceEncoder, TemporalPooling, DenseLayer, ActivationFunction
            )
            layer_class = LSTMLayer
        else:  # GRU
            from backend.classes.ClasificationModel.GRU import (
                GRULayer, SequenceEncoder, TemporalPooling, DenseLayer, ActivationFunction
            )
            layer_class = GRULayer

        # Extraer y crear instancias de capas LSTM/GRU para el encoder
        encoder_layer_instances = []
        for layer in layers:
            if layer["type"] in ["LSTMLayer", "GRULayer"]:
                # ✅ INSTANCIAR LSTMLayer o GRULayer
                layer_instance = layer_class(**layer["config"])
                print(f"✅ {layer['type']} instanciada: {layer_instance}")
                encoder_layer_instances.append(layer_instance)

        # ✅ INSTANCIAR SequenceEncoder
        encoder_instance = SequenceEncoder(
            layers=encoder_layer_instances,
            input_feature_dim=num_channels
        )
        config["encoder"] = encoder_instance
        print(f"✅ SequenceEncoder instanciado: {len(encoder_layer_instances)} capas")

        # ✅ INSTANCIAR TemporalPooling
        pooling_layers = [l for l in layers if l["type"] == "TemporalPooling"]
        if pooling_layers:
            pooling_instance = TemporalPooling(**pooling_layers[0]["config"])
        else:
            pooling_instance = TemporalPooling(kind="last")
        config["pooling"] = pooling_instance
        print(f"✅ TemporalPooling instanciado: {pooling_instance}")

        # ✅ INSTANCIAR fc_layers
        # SOLO las capas Dense que el usuario agregó (sin primera automática)
        # Keras conecta automáticamente con el encoder output
        fc_layer_instances = []
        dense_layers_found = [l for l in layers if l["type"] == "DenseLayer"]

        print(f"🔍 DEBUG LSTM/GRU - Total DenseLayers del usuario: {len(dense_layers_found)}")

        # TODAS las DenseLayer del usuario van a fc_layers
        for i, layer in enumerate(dense_layers_found):
            layer_cfg = layer['config'].copy()

            print(f"🔍 DEBUG LSTM/GRU - fc_layer usuario {i+1}: units={layer_cfg.get('units')}")

            # ✅ Convertir activation a ActivationFunction
            if "activation" in layer_cfg:
                if isinstance(layer_cfg["activation"], dict):
                    layer_cfg["activation"] = ActivationFunction(**layer_cfg["activation"])
                elif isinstance(layer_cfg["activation"], str):
                    layer_cfg["activation"] = ActivationFunction(kind=layer_cfg["activation"].lower())

            dense_inst = DenseLayer(**layer_cfg)
            fc_layer_instances.append(dense_inst)

        if fc_layer_instances:
            config["fc_layers"] = fc_layer_instances
            print(f"✅ fc_layers: {len(fc_layer_instances)} capas del usuario (Keras conecta automáticamente)")

        # ✅ Agregar fc_activation_common (activación común para fc_layers)
        config["fc_activation_common"] = ActivationFunction(kind="relu")

        # ✅ INSTANCIAR classification (SIEMPRE automática, separada del usuario)
        classification_inst = DenseLayer(
            units=num_classes,
            activation=ActivationFunction(kind="softmax")
        )
        config["classification"] = classification_inst
        print(f"✅ classification instanciada (automática): {classification_inst}")

    elif model_type == "CNN":
        # Importar clases Pydantic para instanciación
        from backend.classes.ClasificationModel.CNN import (
            Kernel, ConvolutionLayer, PoolingLayer, DenseLayer, ActivationFunction, FlattenLayer
        )
        import numpy as np

        # Extraer capas convolucionales y pooling
        feature_extractor = []
        for layer in layers:
            if layer["type"] == "ConvolutionLayer":
                layer_config = layer["config"]

                # TRANSFORMAR e INSTANCIAR filtros guardados a objetos Kernel validados
                if "filters" in layer_config:
                    filters_saved = layer_config["filters"]
                    print(f"🔍 DEBUG CNN - Filtros guardados del UI: {filters_saved}")

                    # Convertir filtros del UI a objetos Kernel instanciados y validados
                    kernels_list = []
                    for i, filter_data in enumerate(filters_saved):
                        kernel_matrices = filter_data.get("kernels", [])

                        if len(kernel_matrices) == 3:
                            # ✅ INSTANCIAR objetos Kernel con validación Pydantic
                            kR = Kernel(weights=np.array(kernel_matrices[0], dtype=np.float32))
                            kG = Kernel(weights=np.array(kernel_matrices[1], dtype=np.float32))
                            kB = Kernel(weights=np.array(kernel_matrices[2], dtype=np.float32))

                            print(f"✅ Filtro {i+1} - Kernel R instanciado: shape={kR.shape}")
                            print(f"✅ Filtro {i+1} - Kernel G instanciado: shape={kG.shape}")
                            print(f"✅ Filtro {i+1} - Kernel B instanciado: shape={kB.shape}")

                            # Cada filtro es una tupla de 3 objetos Kernel
                            kernels_list.append((kR, kG, kB))

                    # Extraer parámetros del primer filtro
                    if filters_saved and len(filters_saved) > 0:
                        first_filter = filters_saved[0]
                        stride = tuple(first_filter.get("stride", [1, 1]))
                        padding = first_filter.get("padding", "same")
                        activation_data = first_filter.get("activation", {"kind": "relu"})

                        # ✅ INSTANCIAR ActivationFunction
                        if isinstance(activation_data, dict):
                            activation = ActivationFunction(kind=activation_data.get("kind", "relu"))
                        elif isinstance(activation_data, str):
                            activation = ActivationFunction(kind=activation_data.lower())
                        else:
                            activation = ActivationFunction(kind="relu")
                    else:
                        stride = (1, 1)
                        padding = "same"
                        activation = ActivationFunction(kind="relu")

                    # ✅ INSTANCIAR ConvolutionLayer completa con validación
                    conv_layer_instance = ConvolutionLayer(
                        kernels=kernels_list,
                        stride=stride,
                        padding=padding,
                        activation=activation
                    )

                    print(f"✅ ConvolutionLayer instanciada: {conv_layer_instance.num_filters()} filtros, kernel_shape={conv_layer_instance.kernel_shape()}")

                    feature_extractor.append(conv_layer_instance)

            elif layer["type"] == "PoolingLayer":
                # ✅ INSTANCIAR PoolingLayer
                pooling_instance = PoolingLayer(**layer["config"])
                print(f"✅ PoolingLayer instanciada: {pooling_instance}")
                feature_extractor.append(pooling_instance)

        print(f"🔍 DEBUG CNN - Total feature_extractor layers: {len(feature_extractor)}")
        print(f"🔍 DEBUG CNN - feature_extractor completo: {feature_extractor}")

        # feature_extractor es una LISTA DIRECTA (no un dict con "layers")
        if feature_extractor:
            config["feature_extractor"] = feature_extractor
        else:
            # Si no hay capas, al menos poner una lista vacía para que no falle la validación
            config["feature_extractor"] = []

        # ✅ Agregar FlattenLayer (siempre requerido para CNN)
        config["flatten"] = FlattenLayer()

        # fc_layers (TODAS las capas densas del usuario)
        fc_layers = []
        dense_layers_found = [l for l in layers if l["type"] == "DenseLayer"]

        print(f"🔍 DEBUG CNN - Total DenseLayers del usuario: {len(dense_layers_found)}")

        # TODAS las DenseLayer del usuario van a fc_layers
        for i, layer in enumerate(dense_layers_found):
            layer_cfg = layer['config'].copy()

            print(f"🔍 DEBUG CNN - fc_layer {i+1}: units={layer_cfg.get('units')}")

            # ✅ Convertir activation a ActivationFunction
            if "activation" in layer_cfg:
                if isinstance(layer_cfg["activation"], dict):
                    layer_cfg["activation"] = ActivationFunction(**layer_cfg["activation"])
                elif isinstance(layer_cfg["activation"], str):
                    layer_cfg["activation"] = ActivationFunction(kind=layer_cfg["activation"].lower())

            # ✅ INSTANCIAR DenseLayer
            dense_instance = DenseLayer(**layer_cfg)
            print(f"✅ fc_layer instanciada: {dense_instance}")
            fc_layers.append(dense_instance)

        if fc_layers:
            config["fc_layers"] = fc_layers
            print(f"🔍 DEBUG CNN - fc_layers: {len(fc_layers)} capas instanciadas")

        # ✅ Agregar fc_activation_common (activación común para fc_layers)
        config["fc_activation_common"] = ActivationFunction(kind="relu")

        # ✅ INSTANCIAR classification (SIEMPRE automática, separada del usuario)
        classification_instance = DenseLayer(
            units=num_classes,
            activation=ActivationFunction(kind="softmax")
        )
        config["classification"] = classification_instance
        print(f"✅ classification instanciada (automática): {classification_instance}")

        # ✅ Agregar hiperparámetros de CNN (frame_context e image_hw)
        config["frame_context"] = 4  # Default del test
        config["image_hw"] = (32, 64)  # Default del test
        print(f"✅ CNN hiperparámetros agregados: frame_context=4, image_hw=(32, 64)")

    elif model_type == "SVNN":
        # Importar clases Pydantic para instanciación
        from backend.classes.ClasificationModel.SVNN import DenseLayer, ActivationFunction, InputAdapter

        # Red neuronal simple siguiendo el schema de SVNN
        dense_layer_instances = []
        for layer in layers:
            if layer["type"] == "DenseLayer":
                layer_cfg = layer["config"].copy()

                # ✅ INSTANCIAR ActivationFunction si es dict o string
                if "activation" in layer_cfg:
                    if isinstance(layer_cfg["activation"], dict):
                        layer_cfg["activation"] = ActivationFunction(**layer_cfg["activation"])
                    elif isinstance(layer_cfg["activation"], str):
                        layer_cfg["activation"] = ActivationFunction(kind=layer_cfg["activation"].lower())

                # ✅ INSTANCIAR DenseLayer
                dense_inst = DenseLayer(**layer_cfg)
                print(f"✅ SVNN DenseLayer instanciada: {dense_inst}")
                dense_layer_instances.append(dense_inst)

        # ✅ Configurar SVNN siguiendo el schema exacto (reaplicado)
        # Usar hiperparámetros del usuario (validados previamente)
        config["learning_rate"] = learning_rate
        config["epochs"] = epochs
        config["batch_size"] = batch_size
        config["hidden_size"] = config.get("hidden_size", 64)  # Default del test
        config["classification_units"] = num_classes
        config["input_adapter"] = InputAdapter(reduce_3d="flatten", scale="standard")
        config["fc_activation_common"] = ActivationFunction(kind="relu")

        # ✅ Las capas densas van en "layers" (NO "hidden_layers")
        if dense_layer_instances:
            config["layers"] = dense_layer_instances
        else:
            # Si no hay capas, dejar lista vacía (el validador puede aplicar defaults internos)
            config["layers"] = []

        print(f"✅ SVNN configurado: {len(config.get('layers', []))} layers, classification_units={num_classes}")

    # ✅ APLICAR HIPERPARÁMETROS DEL USUARIO (validados, sin fallbacks)
    config["epochs"] = epochs
    config["batch_size"] = batch_size
    config["learning_rate"] = learning_rate

    print(f"🔍🔍🔍 DEBUG FINAL - Config completo del modelo {model_type}:")
    print(f"🔍🔍🔍 epochs={config.get('epochs')}, batch_size={config.get('batch_size')}, lr={config.get('learning_rate')}")

    return config


def render_neurons_vertical(num_units: int) -> html.Div:
    """
    Renderiza neuronas como una columna vertical apilada.

    Siempre muestra 10 neuronas completas del mismo tamaño.
    Si hay más de 10, muestra las 10 + símbolo ⋮

    Args:
        num_units: Número de neuronas/unidades en la capa
    """
    # Tamaño fijo de neurona y máximo a mostrar
    neuron_size = 14  # Tamaño fijo para todas
    max_display = 10   # Siempre mostrar máximo 10 neuronas completas

    # Limitar cuántas neuronas mostramos visualmente
    display_count = min(num_units, max_display)
    show_ellipsis = num_units > max_display

    # Crear círculos de neuronas apilados verticalmente
    neurons = []
    for i in range(display_count):
        neurons.append(
            html.Div(
                style={
                    "width": f"{neuron_size}px",
                    "height": f"{neuron_size}px",
                    "borderRadius": "50%",
                    "backgroundColor": "rgba(255, 255, 255, 0.95)",
                    "marginBottom": "3px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.4)",
                    "border": "1px solid rgba(255, 255, 255, 0.3)"
                }
            )
        )

    # Agregar "..." si hay más neuronas
    if show_ellipsis:
        neurons.append(
            html.Div(
                "⋮",
                style={
                    "color": "white",
                    "fontSize": "18px",
                    "fontWeight": "bold",
                    "textAlign": "center",
                    "marginTop": "-2px",
                    "lineHeight": "1"
                }
            )
        )

    return html.Div(
        neurons,
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center"
        }
    )


def create_layer_node(
    layer_type: str,
    layer_index: int,
    is_current: bool = False,
    is_fixed: bool = False,
    layer_config: Optional[Dict[str, Any]] = None,
    fixed_units: Optional[int] = None,
    is_computed: bool = False,
    model_type_context: Optional[str] = None
) -> html.Div:
    """
    Crea un nodo visual para una capa en la arquitectura.

    Args:
        layer_type: Tipo de capa (LSTMLayer, DenseLayer, etc.)
        layer_index: Índice de la capa
        is_current: Si es la capa actualmente seleccionada
        is_fixed: Si es una capa fija (input/output)
        layer_config: Configuración de la capa (para mostrar neuronas en Dense)
        fixed_units: Número de unidades para capas fijas (input=channels, output=classes)
        is_computed: Si es una capa calculada automáticamente
        model_type_context: Tipo de modelo (para mostrar info contextual en Input)
    """
    # Determinar color base
    color_key = layer_type.lower().replace("layer", "")
    node_color = LAYER_COLORS.get(color_key, "#999")

    # Obtener icono
    icon_class = LAYER_ICONS.get(layer_type, "fa-circle")

    border_style = "3px solid white" if is_current and not is_fixed else "none"
    cursor_style = "pointer" if not is_fixed else "default"
    opacity = "0.7" if is_fixed else "1"

    # Capas fijas (Input/Output) con fixed_units
    if is_fixed and fixed_units is not None and fixed_units > 0:
        layer_name = "Input Layer" if layer_type == "input" else "Output Layer"

        # Para Input, personalizar según tipo de modelo
        if layer_type == "input":
            if model_type_context == "CNN":
                data_type = "Datos Transformados"
                data_desc = "(frames, time, channels)"
            elif model_type_context in ["LSTM", "GRU"]:
                data_type = "Datos Transformados"
                data_desc = "(frames, time, channels)"
            elif model_type_context == "SVNN":
                data_type = "Vector de características"
                data_desc = "(features)"
            else:
                data_type = "Input Data"
                data_desc = ""

            return html.Div([
                html.Div(
                    layer_name,
                    style={
                        "textAlign": "center",
                        "fontSize": "11px",
                        "color": "rgba(255,255,255,0.7)",
                        "fontWeight": "600",
                        "marginBottom": "3px"
                    }
                ),
                html.Div(
                    data_type,
                    style={
                        "textAlign": "center",
                        "fontSize": "9px",
                        "color": "rgba(255,255,255,0.5)",
                        "fontStyle": "italic",
                        "marginBottom": "5px"
                    }
                ),
                html.Div(
                    data_desc,
                    style={
                        "textAlign": "center",
                        "fontSize": "8px",
                        "color": "rgba(255,255,255,0.4)",
                        "marginBottom": "8px"
                    }
                ),
                render_neurons_vertical(fixed_units),
                html.Div(
                    f"{fixed_units} channels",
                    style={
                        "textAlign": "center",
                        "fontSize": "9px",
                        "color": "rgba(255,255,255,0.5)",
                        "fontWeight": "500",
                        "marginTop": "8px"
                    }
                )
            ],
            style={
                "padding": "10px",
                "minWidth": "40px",
                "opacity": "0.8"
            })

        # Para Output, mantener diseño original
        units_label = f"{fixed_units} classes"
        return html.Div([
            html.Div(
                layer_name,
                style={
                    "textAlign": "center",
                    "fontSize": "11px",
                    "color": "rgba(255,255,255,0.7)",
                    "fontWeight": "600",
                    "marginBottom": "8px"
                }
            ),
            render_neurons_vertical(fixed_units),
            html.Div(
                units_label,
                style={
                    "textAlign": "center",
                    "fontSize": "9px",
                    "color": "rgba(255,255,255,0.5)",
                    "fontWeight": "500",
                    "marginTop": "8px"
                }
            )
        ],
        style={
            "padding": "10px",
            "minWidth": "40px",
            "opacity": "0.8"
        })

    # Manejar capa informativa de conversión transformada→imagen (CNN)
    if layer_type == "TransformToImage" and is_computed and layer_config:
        input_shape = layer_config.get("input_shape", "?")
        output_shape = layer_config.get("output_shape", "?")
        description = layer_config.get("description", "Transform to Image")

        return html.Div([
            html.Div([
                html.I(className="fas fa-exchange-alt", style={"fontSize": "16px", "marginRight": "5px"}),
                description
            ], style={
                "textAlign": "center",
                "fontSize": "11px",
                "color": "rgba(255,255,255,0.9)",
                "fontWeight": "600",
                "marginBottom": "10px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            }),
            html.Div([
                html.Div([
                    html.Div("Transformada 3D", style={"fontSize": "9px", "color": "rgba(255,255,255,0.6)", "marginBottom": "3px"}),
                    html.Div(input_shape, style={"fontSize": "9px", "fontWeight": "600", "color": "#4A90E2", "fontFamily": "monospace"})
                ], style={"textAlign": "center", "padding": "8px", "backgroundColor": "rgba(74, 144, 226, 0.1)", "borderRadius": "4px", "flex": "1"}),
                html.Div("→", style={"fontSize": "14px", "margin": "0 10px", "color": "rgba(255,255,255,0.5)"}),
                html.Div([
                    html.Div("Imagen RGB", style={"fontSize": "9px", "color": "rgba(255,255,255,0.6)", "marginBottom": "3px"}),
                    html.Div(output_shape, style={"fontSize": "9px", "fontWeight": "600", "color": "#F5A623", "fontFamily": "monospace"})
                ], style={"textAlign": "center", "padding": "8px", "backgroundColor": "rgba(245, 166, 35, 0.1)", "borderRadius": "4px", "flex": "1"})
            ], style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "marginBottom": "8px"
            }),
            html.Div([
                html.Div("• Cada frame se convierte en una imagen RGB", style={"fontSize": "8px", "marginBottom": "2px"}),
                html.Div("• (frames, time, channels) → imagen 2D coloreada", style={"fontSize": "8px", "marginBottom": "2px"}),
                html.Div("• Procesamiento per-frame independiente", style={"fontSize": "8px"})
            ], style={
                "textAlign": "left",
                "color": "rgba(255,255,255,0.5)",
                "fontStyle": "italic",
                "marginTop": "5px",
                "paddingLeft": "15px"
            })
        ], style={
            "padding": "12px",
            "minWidth": "250px",
            "backgroundColor": "rgba(245, 166, 35, 0.08)",
            "borderRadius": "8px",
            "border": "1px dashed rgba(245, 166, 35, 0.4)",
            "opacity": "0.9"
        })

    # Manejar ConvolutionLayer con dimensionalidad calculada
    if layer_type == "ConvolutionLayer" and layer_config:
        input_shape = layer_config.get("_computed_input_shape")
        output_shape = layer_config.get("_computed_output_shape")
        num_filters = layer_config.get("_num_filters")
        kernel_size = layer_config.get("_kernel_size")
        stride = layer_config.get("_stride")
        padding = layer_config.get("_padding")

        if input_shape and output_shape:
            # Formato: (H, W, C)
            in_str = f"{input_shape[0]}×{input_shape[1]}×{input_shape[2]}"
            out_str = f"{output_shape[0]}×{output_shape[1]}×{output_shape[2]}"
            kernel_str = f"{kernel_size[0]}×{kernel_size[1]}" if kernel_size else "?"
            stride_str = f"{stride[0]}×{stride[1]}" if stride else "1×1"

            return html.Div([
                html.Div([
                    html.I(className="fas fa-th", style={"fontSize": "14px", "marginRight": "5px"}),
                    "Convolution"
                ], style={
                    "textAlign": "center",
                    "fontSize": "11px",
                    "color": "rgba(255,255,255,0.9)",
                    "fontWeight": "600",
                    "marginBottom": "10px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center"
                }),
                html.Div([
                    html.Div([
                        html.Div("Input", style={"fontSize": "8px", "color": "rgba(255,255,255,0.5)", "marginBottom": "3px"}),
                        html.Div(in_str, style={"fontSize": "10px", "fontWeight": "600", "color": "#4A90E2", "fontFamily": "monospace"})
                    ], style={"textAlign": "center", "padding": "6px", "backgroundColor": "rgba(74, 144, 226, 0.1)", "borderRadius": "4px", "flex": "1"}),
                    html.Div("→", style={"fontSize": "12px", "margin": "0 8px", "color": "rgba(255,255,255,0.5)"}),
                    html.Div([
                        html.Div("Output", style={"fontSize": "8px", "color": "rgba(255,255,255,0.5)", "marginBottom": "3px"}),
                        html.Div(out_str, style={"fontSize": "10px", "fontWeight": "600", "color": "#50E3C2", "fontFamily": "monospace"})
                    ], style={"textAlign": "center", "padding": "6px", "backgroundColor": "rgba(80, 227, 194, 0.1)", "borderRadius": "4px", "flex": "1"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
                html.Div([
                    html.Div(f"Filtros: {num_filters}", style={"fontSize": "8px", "marginBottom": "2px"}),
                    html.Div(f"Kernel: {kernel_str}, Stride: {stride_str}", style={"fontSize": "8px", "marginBottom": "2px"}),
                    html.Div(f"Padding: {padding}", style={"fontSize": "8px"})
                ], style={
                    "textAlign": "center",
                    "color": "rgba(255,255,255,0.6)",
                    "marginTop": "5px"
                })
            ], style={
                "padding": "12px",
                "minWidth": "180px",
                "backgroundColor": "rgba(80, 227, 194, 0.08)",
                "borderRadius": "8px",
                "border": "1px solid rgba(80, 227, 194, 0.3)" if not is_computed else "1px dashed rgba(80, 227, 194, 0.3)",
                "cursor": cursor_style if not is_computed else "default",
                "transition": "all 0.3s ease"
            },
            id={"type": "layer-node", "index": layer_index} if not is_computed else {},
            n_clicks=0 if not is_computed else None)

    # Manejar PoolingLayer con dimensionalidad calculada
    if layer_type == "PoolingLayer" and layer_config:
        input_shape = layer_config.get("_computed_input_shape")
        output_shape = layer_config.get("_computed_output_shape")
        pool_size = layer_config.get("_pool_size")
        stride = layer_config.get("_stride")
        kind = layer_config.get("_kind", "max")

        if input_shape and output_shape:
            in_str = f"{input_shape[0]}×{input_shape[1]}×{input_shape[2]}"
            out_str = f"{output_shape[0]}×{output_shape[1]}×{output_shape[2]}"
            pool_str = f"{pool_size[0]}×{pool_size[1]}" if pool_size else "?"
            stride_str = f"{stride[0]}×{stride[1]}" if stride else "?"

            return html.Div([
                html.Div([
                    html.I(className="fas fa-compress", style={"fontSize": "14px", "marginRight": "5px"}),
                    f"{kind.upper()} Pooling"
                ], style={
                    "textAlign": "center",
                    "fontSize": "11px",
                    "color": "rgba(255,255,255,0.9)",
                    "fontWeight": "600",
                    "marginBottom": "10px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center"
                }),
                html.Div([
                    html.Div([
                        html.Div("Input", style={"fontSize": "8px", "color": "rgba(255,255,255,0.5)", "marginBottom": "3px"}),
                        html.Div(in_str, style={"fontSize": "10px", "fontWeight": "600", "color": "#4A90E2", "fontFamily": "monospace"})
                    ], style={"textAlign": "center", "padding": "6px", "backgroundColor": "rgba(74, 144, 226, 0.1)", "borderRadius": "4px", "flex": "1"}),
                    html.Div("→", style={"fontSize": "12px", "margin": "0 8px", "color": "rgba(255,255,255,0.5)"}),
                    html.Div([
                        html.Div("Output", style={"fontSize": "8px", "color": "rgba(255,255,255,0.5)", "marginBottom": "3px"}),
                        html.Div(out_str, style={"fontSize": "10px", "fontWeight": "600", "color": "#9013FE", "fontFamily": "monospace"})
                    ], style={"textAlign": "center", "padding": "6px", "backgroundColor": "rgba(144, 19, 254, 0.1)", "borderRadius": "4px", "flex": "1"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
                html.Div([
                    html.Div(f"Pool: {pool_str}", style={"fontSize": "8px", "marginBottom": "2px"}),
                    html.Div(f"Stride: {stride_str}", style={"fontSize": "8px"})
                ], style={
                    "textAlign": "center",
                    "color": "rgba(255,255,255,0.6)",
                    "marginTop": "5px"
                })
            ], style={
                "padding": "12px",
                "minWidth": "160px",
                "backgroundColor": "rgba(144, 19, 254, 0.08)",
                "borderRadius": "8px",
                "border": "1px solid rgba(144, 19, 254, 0.3)" if not is_computed else "1px dashed rgba(144, 19, 254, 0.3)",
                "cursor": cursor_style if not is_computed else "default",
                "transition": "all 0.3s ease"
            },
            id={"type": "layer-node", "index": layer_index} if not is_computed else {},
            n_clicks=0 if not is_computed else None)

    # Manejar capa Flatten calculada (mostrar como Dense con output_size)
    if layer_type == "Flatten" and is_computed and layer_config and "output_size" in layer_config:
        output_size = layer_config.get("output_size")
        input_shape = layer_config.get("_computed_input_shape")

        if input_shape:
            in_str = f"{input_shape[0]}×{input_shape[1]}×{input_shape[2]}"
        else:
            in_str = "?"

        return html.Div([
            html.Div([
                html.I(className="fas fa-layer-group", style={"fontSize": "14px", "marginRight": "5px"}),
                "Flatten"
            ], style={
                "textAlign": "center",
                "fontSize": "11px",
                "color": "rgba(255,255,255,0.7)",
                "fontWeight": "600",
                "marginBottom": "10px",
                "fontStyle": "italic",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            }),
            html.Div([
                html.Div([
                    html.Div("3D Input", style={"fontSize": "8px", "color": "rgba(255,255,255,0.5)", "marginBottom": "3px"}),
                    html.Div(in_str, style={"fontSize": "10px", "fontWeight": "600", "color": "#4A90E2", "fontFamily": "monospace"})
                ], style={"textAlign": "center", "padding": "6px", "backgroundColor": "rgba(74, 144, 226, 0.1)", "borderRadius": "4px", "flex": "1"}),
                html.Div("→", style={"fontSize": "12px", "margin": "0 8px", "color": "rgba(255,255,255,0.5)"}),
                html.Div([
                    html.Div("1D Output", style={"fontSize": "8px", "color": "rgba(255,255,255,0.5)", "marginBottom": "3px"}),
                    html.Div(f"{output_size:,}", style={"fontSize": "10px", "fontWeight": "600", "color": "#F5A623", "fontFamily": "monospace"})
                ], style={"textAlign": "center", "padding": "6px", "backgroundColor": "rgba(245, 166, 35, 0.1)", "borderRadius": "4px", "flex": "1"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            render_neurons_vertical(min(output_size, 10)),  # Limitar visualización
            html.Div(
                f"{output_size:,} neurons",
                style={
                    "textAlign": "center",
                    "fontSize": "9px",
                    "color": "rgba(255,255,255,0.5)",
                    "fontWeight": "500",
                    "marginTop": "8px"
                }
            )
        ],
        style={
            "padding": "12px",
            "minWidth": "160px",
            "opacity": "0.85",
            "backgroundColor": "rgba(245, 166, 35, 0.08)",
            "borderRadius": "8px",
            "border": "1px dashed rgba(245, 166, 35, 0.3)"
        })

    # Para DenseLayer con configuración, mostrar como columna vertical de neuronas
    if layer_type == "DenseLayer" and layer_config and "units" in layer_config and not is_fixed:
        num_units = layer_config.get("units")
        description = layer_config.get("description", "")

        # ✅ Validar que num_units no sea None
        if num_units is not None and isinstance(num_units, (int, float)) and num_units > 0:
            # Renderizar columna vertical de neuronas
            layer_name = description if is_computed and description else LAYER_NAMES.get(layer_type, layer_type)

            return html.Div([
                html.Div(
                    layer_name,
                    style={
                        "textAlign": "center",
                        "fontSize": "11px",
                        "color": "white" if not is_computed else "rgba(255,255,255,0.7)",
                        "fontWeight": "600",
                        "marginBottom": "8px",
                        "fontStyle": "italic" if is_computed else "normal"
                    }
                ),
                render_neurons_vertical(int(num_units)),
                html.Div(
                    f"{int(num_units)} units",
                    style={
                        "textAlign": "center",
                        "fontSize": "9px",
                        "color": "rgba(255,255,255,0.6)" if not is_computed else "rgba(255,255,255,0.5)",
                        "fontWeight": "500",
                        "marginTop": "8px"
                    }
                )
            ],
            style={
                "padding": "10px",
                "minWidth": "40px",
                "cursor": cursor_style if not is_computed else "default",
                "border": border_style if not is_computed else "1px dashed rgba(80, 227, 194, 0.3)",
                "borderRadius": "8px",
                "transition": "all 0.3s ease",
                "backgroundColor": "rgba(80, 227, 194, 0.05)" if is_computed else "transparent"
            },
            id={"type": "layer-node", "index": layer_index} if not is_computed else {},
            n_clicks=0 if not is_computed else None)

    # Manejar capa TemporalPooling (calculada o del usuario)
    if layer_type == "TemporalPooling" and layer_config:
        pooling_kind = layer_config.get("kind", "last")
        output_size = layer_config.get("output_size")

        return html.Div([
            html.Div([
                html.I(className="fas fa-compress-arrows-alt", style={"fontSize": "16px", "marginRight": "5px"}),
                "Temporal Pooling"
            ], style={
                "textAlign": "center",
                "fontSize": "11px",
                "color": "rgba(255,255,255,0.9)" if not is_computed else "rgba(255,255,255,0.7)",
                "fontWeight": "600",
                "marginBottom": "8px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            }),
            html.Div(
                f"Method: {pooling_kind.upper()}",
                style={
                    "textAlign": "center",
                    "fontSize": "10px",
                    "color": "rgba(255,255,255,0.6)",
                    "marginBottom": "5px",
                    "fontFamily": "monospace"
                }
            ),
            html.Div(
                f"Output: {output_size} features" if output_size else "Reduces temporal dimension",
                style={
                    "textAlign": "center",
                    "fontSize": "9px",
                    "color": "rgba(255,255,255,0.5)",
                    "fontStyle": "italic"
                }
            )
        ], style={
            "padding": "10px",
            "minWidth": "120px",
            "backgroundColor": "rgba(189, 16, 224, 0.1)" if not is_computed else "rgba(189, 16, 224, 0.05)",
            "borderRadius": "8px",
            "border": "1px solid rgba(189, 16, 224, 0.3)" if not is_computed else "1px dashed rgba(189, 16, 224, 0.3)",
            "opacity": "0.9" if not is_computed else "0.85"
        })

    # Manejar primera capa LSTM/GRU con input_feature_dim
    if layer_type in ["LSTMLayer", "GRULayer"] and layer_config and "input_feature_dim" in layer_config:
        input_dim = layer_config.get("input_feature_dim")
        hidden_size = layer_config.get("hidden_size", "?")

        if input_dim and isinstance(input_dim, int):
            layer_name = LAYER_NAMES.get(layer_type, layer_type)

            return html.Div([
                html.Div(
                    layer_name,
                    style={
                        "textAlign": "center",
                        "fontSize": "11px",
                        "color": "white",
                        "fontWeight": "600",
                        "marginBottom": "5px"
                    }
                ),
                html.Div(
                    f"input: {input_dim}",
                    style={
                        "textAlign": "center",
                        "fontSize": "9px",
                        "color": "rgba(255,255,255,0.5)",
                        "fontStyle": "italic",
                        "marginBottom": "5px"
                    }
                ),
                html.Div(
                    html.I(className=f"fas {icon_class}", style={"fontSize": "20px"}),
                    style={
                        "backgroundColor": node_color,
                        "width": "50px",
                        "height": "50px",
                        "borderRadius": "50%",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "boxShadow": "0 4px 8px rgba(0,0,0,0.3)",
                        "border": border_style,
                        "color": "white",
                        "margin": "0 auto"
                    }
                ),
                html.Div(
                    f"hidden: {hidden_size}",
                    style={
                        "textAlign": "center",
                        "fontSize": "9px",
                        "color": "rgba(255,255,255,0.6)",
                        "marginTop": "5px"
                    }
                )
            ],
            style={
                "padding": "10px",
                "minWidth": "80px",
                "cursor": cursor_style,
                "border": border_style,
                "borderRadius": "8px",
                "transition": "all 0.3s ease"
            },
            id={"type": "layer-node", "index": layer_index},
            n_clicks=0)

    # Para otras capas o DenseLayer sin configurar, usar el diseño de nodo circular
    node_size = 60
    node_content = html.I(className=f"fas {icon_class}", style={"fontSize": "24px"}) if is_fixed else str(layer_index + 1)

    return html.Div([
        html.Div(
            node_content,
            style={
                "backgroundColor": node_color,
                "width": f"{node_size}px",
                "height": f"{node_size}px",
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


def calculate_computed_layers(
    layers: List[Dict[str, Any]],
    model_type: str,
    num_channels: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Calcula capas "virtuales" que se insertan automáticamente según el modelo.

    - CNN: Calcula el tamaño del flatten después de feature extraction
    - LSTM/GRU: Marca la primera capa con input_feature_dim

    Returns:
        Lista de capas con capas calculadas insertadas
    """
    if not layers:
        return []

    computed_layers = []

    if model_type == "CNN":
        import math

        # Agregar capa informativa de conversión de transformada a imagen al inicio
        if layers and num_channels:
            # Capa VISUAL para mostrar conversión de transformada a imagen (no va al JSON)
            computed_layers.append({
                "type": "TransformToImage",
                "config": {
                    "input_shape": f"(frames, time, {num_channels})",
                    "output_shape": "64×128×3 RGB",
                    "description": "Transformada 3D → Imágenes RGB per-frame"
                },
                "computed": True,
                "informative": True,
                "visual_only": True  # ← Solo para referencia académica
            })

        # Inicializar dimensiones de entrada (imagen estándar RGB)
        H, W, C = 64, 128, 3
        flatten_inserted = False

        for i, layer in enumerate(layers):
            layer_type = layer["type"]

            # Procesar capas Conv/Pool y calcular dimensionalidad
            if layer_type == "ConvolutionLayer" and not flatten_inserted:
                config = layer.get("config", {})
                filters = config.get("filters", [])

                if filters and len(filters) > 0:
                    # Obtener parámetros del primer filtro (todos deben ser iguales)
                    first_filter = filters[0]
                    kernel_size = tuple(first_filter.get("kernel_size", [3, 3]))
                    stride = tuple(first_filter.get("stride", [1, 1]))
                    padding = first_filter.get("padding", "same")
                    num_filters = len(filters)

                    # Calcular output shape
                    input_shape = (H, W, C)

                    if padding == "same":
                        # Con padding='same', el tamaño espacial se mantiene (si stride=1)
                        H_out = math.ceil(H / stride[0])
                        W_out = math.ceil(W / stride[1])
                    else:  # padding='valid'
                        # Sin padding, se reduce según el kernel
                        H_out = math.floor((H - kernel_size[0]) / stride[0]) + 1
                        W_out = math.floor((W - kernel_size[1]) / stride[1]) + 1

                    C_out = num_filters  # Cada filtro genera un canal de salida
                    output_shape = (H_out, W_out, C_out)

                    # Actualizar dimensiones actuales
                    H, W, C = H_out, W_out, C_out

                    # Agregar capa con dimensionalidad calculada
                    layer_with_dims = layer.copy()
                    if "config" not in layer_with_dims:
                        layer_with_dims["config"] = {}
                    layer_with_dims["config"]["_computed_input_shape"] = input_shape
                    layer_with_dims["config"]["_computed_output_shape"] = output_shape
                    layer_with_dims["config"]["_num_filters"] = num_filters
                    layer_with_dims["config"]["_kernel_size"] = kernel_size
                    layer_with_dims["config"]["_stride"] = stride
                    layer_with_dims["config"]["_padding"] = padding

                    computed_layers.append(layer_with_dims)
                    print(f"📐 Conv #{i+1}: {input_shape} → {output_shape} (filters={num_filters}, kernel={kernel_size}, stride={stride}, padding={padding})")
                else:
                    # Sin filtros, agregar sin cálculo
                    computed_layers.append(layer)

            elif layer_type == "PoolingLayer" and not flatten_inserted:
                config = layer.get("config", {})
                pool_size = tuple(config.get("pool_size", (2, 2)))
                stride = tuple(config.get("stride", pool_size))  # Default stride = pool_size
                padding = config.get("padding", "valid")
                kind = config.get("kind", "max")

                # Calcular output shape
                input_shape = (H, W, C)

                if padding == "same":
                    H_out = math.ceil(H / stride[0])
                    W_out = math.ceil(W / stride[1])
                else:  # padding='valid'
                    H_out = math.floor((H - pool_size[0]) / stride[0]) + 1
                    W_out = math.floor((W - pool_size[1]) / stride[1]) + 1

                C_out = C  # Pooling no cambia el número de canales
                output_shape = (H_out, W_out, C_out)

                # Actualizar dimensiones actuales
                H, W, C = H_out, W_out, C_out

                # Agregar capa con dimensionalidad calculada
                layer_with_dims = layer.copy()
                if "config" not in layer_with_dims:
                    layer_with_dims["config"] = {}
                layer_with_dims["config"]["_computed_input_shape"] = input_shape
                layer_with_dims["config"]["_computed_output_shape"] = output_shape
                layer_with_dims["config"]["_pool_size"] = pool_size
                layer_with_dims["config"]["_stride"] = stride
                layer_with_dims["config"]["_kind"] = kind

                computed_layers.append(layer_with_dims)
                print(f"📐 Pool #{i+1}: {input_shape} → {output_shape} (pool={pool_size}, stride={stride}, kind={kind})")

            # Al encontrar la primera Dense, calcular y agregar Flatten
            elif layer_type == "DenseLayer" and not flatten_inserted:
                # Calcular flatten_size con las dimensiones actuales
                flatten_size = H * W * C
                final_shape_before_flatten = (H, W, C)

                # Insertar capa Flatten VISUAL
                computed_layers.append({
                    "type": "Flatten",
                    "config": {
                        "output_size": flatten_size,
                        "_computed_input_shape": final_shape_before_flatten,
                        "_computed_output_shape": (flatten_size,)
                    },
                    "computed": True,
                    "visual_only": True
                })
                print(f"📐 Flatten: {final_shape_before_flatten} → ({flatten_size},)")
                flatten_inserted = True

                computed_layers.append(layer)
            else:
                computed_layers.append(layer)

        return computed_layers

    elif model_type in ["LSTM", "GRU"]:
        # Marcar la primera capa LSTM/GRU con input_feature_dim
        # Y rastrear la ÚLTIMA para calcular el encoder_output_size
        main_layer_type = "LSTMLayer" if model_type == "LSTM" else "GRULayer"
        first_marked = False
        has_pooling = False
        encoder_output_size = None
        first_dense_inserted = False
        last_encoder_layer = None

        for layer in layers:
            layer_type = layer["type"]

            # Marcar primera capa LSTM/GRU con input_feature_dim
            if layer_type == main_layer_type and not first_marked and num_channels:
                layer_copy = layer.copy()
                if "config" not in layer_copy:
                    layer_copy["config"] = {}
                layer_copy["config"]["input_feature_dim"] = num_channels
                layer_copy["is_first"] = True
                computed_layers.append(layer_copy)
                first_marked = True
                last_encoder_layer = layer_copy  # Guardar como última
            # Rastrear TODAS las capas LSTM/GRU (la última será la que determine la salida)
            elif layer_type == main_layer_type:
                computed_layers.append(layer)
                last_encoder_layer = layer  # Actualizar última

            # TemporalPooling
            elif layer_type == "TemporalPooling":
                has_pooling = True
                computed_layers.append(layer)

            # Primera DenseLayer después de encoder/pooling
            elif layer_type == "DenseLayer" and not first_dense_inserted:
                # Calcular tamaño de salida del encoder usando la ÚLTIMA capa LSTM/GRU
                if last_encoder_layer:
                    config = last_encoder_layer.get("config", {})
                    hidden_size = config.get("hidden_size", 64)
                    bidirectional = config.get("bidirectional", False)
                    encoder_output_size = hidden_size * 2 if bidirectional else hidden_size

                if encoder_output_size:
                    # Si no hay pooling explícito, agregar uno por defecto
                    if not has_pooling:
                        computed_layers.append({
                            "type": "TemporalPooling",
                            "config": {
                                "kind": "last",
                                "output_size": encoder_output_size
                            },
                            "computed": True
                        })
                        has_pooling = True

                    # Agregar capa Dense VISUAL (solo para referencia académica, no va al JSON)
                    # Keras conecta automáticamente: encoder_output → primera Dense del usuario
                    computed_layers.append({
                        "type": "DenseLayer",
                        "config": {
                            "units": encoder_output_size,
                            "description": "Primera Dense (salida del encoder) - Visual"
                        },
                        "computed": True,
                        "visual_only": True  # ← Flag para identificar que es solo visual
                    })
                    first_dense_inserted = True

                # Agregar la Dense del usuario
                computed_layers.append(layer)

            else:
                computed_layers.append(layer)

        return computed_layers

    # Para otros modelos, devolver sin cambios
    return layers


def create_architecture_visualization(
    layers: List[Dict[str, Any]],
    current_step: int,
    num_channels: Optional[int] = None,
    num_classes: Optional[int] = None,
    model_type: Optional[str] = None
) -> html.Div:
    """
    Crea la visualización completa de la arquitectura con todas las capas.
    Incluye nodos fijos de Input y Output, más capas calculadas automáticamente.

    Args:
        layers: Lista de capas agregadas por el usuario
        current_step: Paso actual (índice de la capa siendo configurada)
        num_channels: Número de canales de entrada (para capa Input)
        num_classes: Número de clases de salida (para capa Output)
        model_type: Tipo de modelo (CNN, LSTM, GRU, SVNN)
    """
    nodes = []

    # Nodo de INPUT (fijo) - con información según tipo de modelo
    nodes.append(create_layer_node(
        "input", -1, False,
        is_fixed=True,
        fixed_units=num_channels,
        model_type_context=model_type
    ))

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
        # Calcular capas con inserciones automáticas
        computed_layers = calculate_computed_layers(layers, model_type or "SVNN", num_channels)

        # Crear nodos de capas (incluyendo calculadas)
        user_layer_index = 0
        for i, layer in enumerate(computed_layers):
            is_computed = layer.get("computed", False)
            layer_type = layer["type"]
            layer_config = layer.get("config", {})

            # Determinar si es la capa actual (solo para capas del usuario)
            if not is_computed:
                is_current = (user_layer_index == current_step)
                user_layer_index += 1
            else:
                is_current = False

            # Estilo especial para capas calculadas
            if is_computed:
                nodes.append(create_layer_node(
                    layer_type, i, is_current,
                    layer_config=layer_config,
                    is_computed=True
                ))
            else:
                nodes.append(create_layer_node(
                    layer_type, i, is_current,
                    layer_config=layer_config
                ))

            # Agregar flecha conectora
            nodes.append(html.Div([
                html.I(className="fas fa-arrow-right",
                       style={"color": "rgba(255,255,255,0.5)", "fontSize": "20px"})
            ], style={"display": "flex", "alignItems": "center", "padding": "0 10px"}))

    # Nodo de OUTPUT (fijo) - con num_classes si está disponible
    nodes.append(create_layer_node("output", -1, False, is_fixed=True, fixed_units=num_classes))

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


def get_layer_config_form(layer_type: str, layer_index: int, schema_defs: Dict[str, Any], saved_config: Dict[str, Any] = None) -> List:
    """
    Genera el formulario de configuración para un tipo de capa específico.

    Args:
        layer_type: Tipo de capa
        layer_index: Índice de la capa
        schema_defs: Definiciones del schema ($defs)
        saved_config: Configuración guardada previamente (para restaurar valores)
    """
    saved_config = saved_config or {}

    # Caso especial: ConvolutionLayer usa editor matricial
    if layer_type == "ConvolutionLayer":
        # Extraer filtros guardados si existen
        saved_filters = saved_config.get("filters", [])
        return [create_convolution_layer_config(layer_index, saved_filters)]

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

        # ✅ Usar valor guardado si existe, sino usar default del schema
        current_value = saved_config.get(field_name, default_value)

        input_id = {
            "type": "layer-config-input",
            "layer_index": layer_index,
            "field": field_name
        }

        # ✅ Detectar campos de ActivationFunction (campo anidado)
        if field_name == "activation":
            # Opciones de activación disponibles
            activation_options = ["relu", "tanh", "sigmoid", "gelu", "softmax", "linear"]

            # Extraer valor actual (puede ser dict, string, o None)
            if isinstance(current_value, dict):
                current_activation = current_value.get("kind", "relu")
            elif isinstance(current_value, str):
                current_activation = current_value
            else:
                current_activation = "relu"

            form_fields.append(html.Div([
                dbc.Label(field_label, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dcc.Dropdown(
                    id=input_id,
                    options=[{"label": act.upper(), "value": act} for act in activation_options],
                    value=current_activation,
                    placeholder="Selecciona activación",
                    clearable=False,
                    style={"flex": "1", "fontSize": "14px"}
                )
            ], className="input-field-group"))
            continue

        # Detectar enums
        enum_values = field_info.get("enum")
        if enum_values:
            form_fields.append(html.Div([
                dbc.Label(field_label, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dcc.Dropdown(
                    id=input_id,
                    options=[{"label": str(v), "value": v} for v in enum_values],
                    value=current_value,  # ✅ Usar valor guardado
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
                    value=current_value,  # ✅ Usar valor guardado
                    step=1 if field_type == "integer" else 0.001,
                    style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
                )
            ], className="input-field-group"))
            continue

        # Booleanos
        if field_type == "boolean":
            # Para booleanos, asegurar que sea True/False
            bool_value = current_value if current_value is not None else (default_value if default_value is not None else False)
            form_fields.append(html.Div([
                dbc.Label(field_label, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
                dbc.Checkbox(
                    id=input_id,
                    value=bool_value,  # ✅ Usar valor guardado
                    style={"marginTop": "10px"}
                )
            ], className="input-field-group"))
            continue

        # String por defecto
        # Para arrays, convertir de vuelta a string con comas
        if isinstance(current_value, list):
            current_value = ", ".join(map(str, current_value))

        form_fields.append(html.Div([
            dbc.Label(field_label, style={"minWidth": "140px", "color": "white", "fontSize": "13px"}),
            dbc.Input(
                id=input_id,
                type="text",
                value=current_value,  # ✅ Usar valor guardado
                placeholder=f"Ingresa {field_label}",
                style={"flex": "1", "fontSize": "15px", "height": "42px", "padding": "8px 12px"}
            )
        ], className="input-field-group"))

    return form_fields


def create_interactive_config_card(model_name: str, schema: Dict[str, Any], classifier_type: str = "P300") -> html.Div:
    """
    Crea la card interactiva completa para construcción de arquitectura.

    Args:
        model_name: Nombre del modelo
        schema: Schema completo del modelo
        classifier_type: Tipo de clasificador - "P300" o "InnerSpeech" (default: "P300")
    """
    return html.Div([
        # Stores para estado - usando storage_type='memory' para evitar problemas de sincronización
        dcc.Store(id="architecture-layers", data=[], storage_type='memory'),  # Lista de capas agregadas
        dcc.Store(id="current-step", data=0, storage_type='memory'),  # Paso actual
        dcc.Store(id="model-type", data=model_name, storage_type='memory'),  # Tipo de modelo
        dcc.Store(id="classifier-type-store", data=classifier_type, storage_type='memory'),  # Tipo de clasificador (P300 o InnerSpeech)
        dcc.Store(id="validation-trigger", data=None, storage_type='memory'),  # Trigger para mensajes de validación

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
                    dbc.Button([
                        html.I(className="fas fa-eraser me-1"),
                        "Limpiar"
                    ],
                        id="clear-architecture-btn",
                        color="secondary",
                        size="sm",
                        outline=True,
                        style={"fontSize": "12px"}
                    )
                ], style={"display": "flex", "alignItems": "center", "width": "100%", "gap": "10px"})
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

                # === SECCIÓN DE HIPERPARÁMETROS DE ENTRENAMIENTO ===
                html.Div([
                    html.Hr(style={"margin": "30px 0 20px", "borderTop": "1px solid rgba(255,255,255,0.15)"}),
                    html.H6("Hiperparámetros de Entrenamiento", style={
                        "color": "white",
                        "fontWeight": "600",
                        "marginBottom": "15px",
                        "textAlign": "center"
                    }),
                    
                    # Formulario de hiperparámetros con estilos consistentes
                    html.Div([
                        # Epochs
                        html.Div([
                            dbc.Label("Épocas", style={
                                "minWidth": "120px",
                                "color": "white",
                                "fontSize": "13px",
                                "fontWeight": "500"
                            }),
                            dbc.Input(
                                id={"type": "hyperparam-epochs", "model": model_name},
                                type="number",
                                min=1,
                                max=1000,
                                value=10,
                                step=1,
                                style={
                                    "flex": "1",
                                    "fontSize": "14px",
                                    "height": "38px",
                                    "padding": "8px 12px"
                                }
                            )
                        ], className="input-field-group", style={"marginBottom": "12px"}),
                        
                        # Batch Size
                        html.Div([
                            dbc.Label("Tamaño de Batch", style={
                                "minWidth": "120px",
                                "color": "white",
                                "fontSize": "13px",
                                "fontWeight": "500"
                            }),
                            dbc.Input(
                                id={"type": "hyperparam-batch-size", "model": model_name},
                                type="number",
                                min=1,
                                max=512,
                                value=32,
                                step=1,
                                style={
                                    "flex": "1",
                                    "fontSize": "14px",
                                    "height": "38px",
                                    "padding": "8px 12px"
                                }
                            )
                        ], className="input-field-group", style={"marginBottom": "12px"}),
                        
                        # Learning Rate
                        html.Div([
                            dbc.Label("Learning Rate", style={
                                "minWidth": "120px",
                                "color": "white",
                                "fontSize": "13px",
                                "fontWeight": "500"
                            }),
                            dbc.Input(
                                id={"type": "hyperparam-lr", "model": model_name},
                                type="number",
                                min=0.00001,
                                max=1.0,
                                value=0.001,
                                step=0.0001,
                                style={
                                    "flex": "1",
                                    "fontSize": "14px",
                                    "height": "38px",
                                    "padding": "8px 12px"
                                }
                            )
                        ], className="input-field-group", style={"marginBottom": "8px"}),
                        
                        # Información adicional
                        dbc.Alert([
                            html.I(className="fas fa-info-circle me-2"),
                            html.Strong("Importante: "),
                            "Estos parámetros son obligatorios y se usarán para todos los entrenamientos (Probar Configuración y Entrenamiento Local)."
                        ], color="info", style={"fontSize": "12px", "marginTop": "15px", "padding": "10px"})
                    ], style={
                        "backgroundColor": "rgba(0,0,0,0.2)",
                        "borderRadius": "8px",
                        "padding": "15px",
                        "border": "1px solid rgba(255,255,255,0.1)"
                    })
                ]),

                # Botón final
                html.Div([
                    dbc.Button(
                        "Probar Configuración",
                        id={"type": "test-config-btn", "model": model_name},
                        color="primary",
                        className="w-100 mt-3",
                        style={"fontSize": "15px", "height": "42px", "fontWeight": "600"}
                    ),
                    # Alert para mostrar resultados de validación con animación de carga
                    dcc.Loading(
                        id={"type": "test-config-loading", "model": model_name},
                        type="circle",
                        fullscreen=False,
                        children=[
                            html.Div(id={"type": "test-config-result", "model": model_name}, className="mt-3")
                        ],
                        color="#0d6efd",
                        style={"marginTop": "20px"}
                    )
                ]),

                # Divisor visual antes de sección de entrenamiento
                html.Hr(style={"margin": "30px 0", "borderTop": "1px solid rgba(255,255,255,0.15)"}),

                # Sección de entrenamiento local
                create_local_training_section(model_name),

                # Divisor visual antes de sección cloud
                html.Hr(style={"margin": "30px 0", "borderTop": "1px solid rgba(255,255,255,0.15)"}),

                # Sección de entrenamiento en la nube (simulación)
                create_cloud_training_section(model_name),
            ])
        ], className="right-panel-card")
    ], id="interactive-config-container")


# ============ CALLBACKS ============

def register_interactive_callbacks():
    """Registra todos los callbacks necesarios para el sistema interactivo."""

    # Utilidad de logging central (reaplicada)
    def _arch_log(*parts: Any) -> None:
        try:
            msg = " ".join(str(p) for p in parts)
            print(f"[ARCH] {msg}")
        except Exception:
            pass

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
        # Crear una copia de la lista para evitar mutaciones
        current_layers = list(current_layers) if current_layers else []

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

        print(f"🔵 DEBUG: Agregando capa {layer_type}. Total de capas: {len(current_layers)}")

        return current_layers, {"is_error": False, "message": f"✓ {LAYER_NAMES.get(layer_type, layer_type)} agregada"}


    # Callback: Actualizar visualización de arquitectura
    @callback(
        Output("architecture-visualization", "children"),
        [Input("architecture-layers", "data"),
         Input("current-step", "data"),
         Input("selected-dataset", "data"),
         Input("model-type", "data"),
         State("classifier-type-store", "data")]
    )
    def update_visualization(layers, current_step, selected_dataset, model_type, classifier_type):
        # Obtener num_channels y num_classes del dataset
        num_channels = None
        num_classes = None

        if selected_dataset:
            try:
                from shared.fileUtils import get_dataset_metadata
                metadata = get_dataset_metadata(selected_dataset)
                channel_names = metadata.get("channel_names") or []
                classes = metadata.get("classes") or []

                # ===== FIX: Calcular num_classes según el tipo de clasificador =====
                if classifier_type == "InnerSpeech":
                    # Inner Speech: Multiclase sin rest/none/unlabeled
                    excluded_classes = ["rest", "none", "unlabeled"]
                    classes = [cls for cls in classes if cls.lower() not in excluded_classes]
                    num_classes = len(classes)
                elif classifier_type == "P300":
                    # P300: Clasificación binaria (0=NonTarget/rest, 1=Target)
                    num_classes = 2
                else:
                    # Fallback: usar todas las clases del dataset
                    num_classes = len(classes)

                num_channels = len(channel_names)
            except Exception:
                pass  # Si falla, usar valores por defecto

        return create_architecture_visualization(
            layers or [],
            current_step or 0,
            num_channels=num_channels,
            num_classes=num_classes,
            model_type=model_type
        )


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
        current_step = current_step or 0

        # Validar que current_step esté en rango
        if current_step >= total_steps:
            current_step = total_steps - 1
        elif current_step < 0:
            current_step = 0

        step_num = current_step + 1
        layer = layers[current_step]
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

        # ✅ Pasar la configuración guardada para restaurar valores
        saved_config = layer.get("config", {})
        form_fields = get_layer_config_form(layer["type"], step, schema_defs, saved_config)

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

        # Crear una copia de la lista
        layers = list(layers)

        step = current_step or 0
        if step >= len(layers):
            return no_update, no_update

        # Eliminar la capa
        layers.pop(step)

        # Ajustar step si es necesario
        new_step = min(step, max(0, len(layers) - 1)) if layers else 0

        print(f"🔴 DEBUG: Eliminada capa en índice {step}. Total de capas: {len(layers)}")

        return layers, new_step


    # Callback: Limpiar toda la arquitectura
    @callback(
        [Output("architecture-layers", "data", allow_duplicate=True),
         Output("current-step", "data", allow_duplicate=True),
         Output("validation-trigger", "data", allow_duplicate=True)],
        Input("clear-architecture-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def clear_architecture(n_clicks):
        """Limpia todas las capas y reinicia la configuración."""
        if not n_clicks:
            return no_update, no_update, no_update

        return [], 0, {"is_error": False, "message": "✓ Arquitectura limpiada"}


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

        # Validar que step esté en rango
        if step >= total:
            step = total - 1
        elif step < 0:
            step = 0

        prev_disabled = (step == 0)
        next_disabled = (step >= total - 1)
        counter_text = f"Capa {step + 1} de {total}"

        return prev_disabled, next_disabled, counter_text


    # Callback: Guardar valores actuales antes de navegar
    @callback(
        Output("architecture-layers", "data", allow_duplicate=True),
        [Input("prev-step-btn", "n_clicks"),
         Input("next-step-btn", "n_clicks"),
         Input({"type": "layer-node", "index": ALL}, "n_clicks")],
        [State({"type": "layer-config-input", "layer_index": ALL, "field": ALL}, "value"),
         State({"type": "layer-config-input", "layer_index": ALL, "field": ALL}, "id"),
         State({"type": "conv-filters-store", "layer": ALL}, "data"),
         State({"type": "conv-filters-store", "layer": ALL}, "id"),
         State({"type": "kernel-cell", "filter": ALL, "kernel": ALL, "row": ALL, "col": ALL}, "value"),
         State({"type": "kernel-cell", "filter": ALL, "kernel": ALL, "row": ALL, "col": ALL}, "id"),
         State({"type": "stride-h", "filter": ALL}, "value"),
         State({"type": "stride-w", "filter": ALL}, "value"),
         State({"type": "stride-h", "filter": ALL}, "id"),
         State({"type": "padding", "filter": ALL}, "value"),
         State({"type": "activation", "filter": ALL}, "value"),
         State({"type": "kernel-size-dropdown", "filter": ALL}, "value"),
         State({"type": "kernel-size-dropdown", "filter": ALL}, "id"),
         State("architecture-layers", "data"),
         State("current-step", "data")],
        prevent_initial_call=True
    )
    def save_current_values_before_action(prev_clicks, next_clicks, node_clicks,
                                           input_values, input_ids, cnn_filters_data, cnn_filters_ids,
                                           kernel_values, kernel_ids,
                                           stride_h_values, stride_w_values, stride_h_ids,
                                           padding_values, activation_values,
                                           kernel_size_values, kernel_size_ids,
                                           layers, current_step):
        """Guarda los valores actuales de los inputs antes de cualquier acción de navegación."""
        if not layers:
            return no_update

        # Crear una copia profunda de la lista de capas
        import copy
        layers = copy.deepcopy(layers)

        # Agrupar valores por layer_index
        config_by_layer = {}
        if input_ids and input_values:
            for input_id, value in zip(input_ids, input_values):
                layer_idx = input_id["layer_index"]
                field_name = input_id["field"]

                if layer_idx not in config_by_layer:
                    config_by_layer[layer_idx] = {}

                # Procesar valores (arrays, números, etc.)
                if isinstance(value, str) and "," in value:
                    try:
                        value = [float(v.strip()) for v in value.split(",")]
                    except (ValueError, AttributeError):
                        pass

                config_by_layer[layer_idx][field_name] = value

        # Guardar filtros de CNN por layer_index
        cnn_config_by_layer = {}
        if cnn_filters_ids and cnn_filters_data:
            for filter_id, filter_data in zip(cnn_filters_ids, cnn_filters_data):
                layer_idx = filter_id["layer"]
                if filter_data:  # Solo si hay filtros definidos
                    import copy
                    updated_filters = copy.deepcopy(filter_data)

                    # AUTO-GUARDAR: Capturar valores actuales de las matrices ANTES de guardar
                    if kernel_ids and kernel_values:
                        for cell_id, cell_value in zip(kernel_ids, kernel_values):
                            filter_idx = cell_id["filter"]
                            kernel_idx = cell_id["kernel"]
                            row = cell_id["row"]
                            col = cell_id["col"]

                            if filter_idx < len(updated_filters):
                                if "kernels" not in updated_filters[filter_idx]:
                                    updated_filters[filter_idx]["kernels"] = [
                                        [[0.0 for _ in range(3)] for _ in range(3)] for _ in range(3)
                                    ]
                                if kernel_idx < len(updated_filters[filter_idx]["kernels"]):
                                    if row < len(updated_filters[filter_idx]["kernels"][kernel_idx]):
                                        if col < len(updated_filters[filter_idx]["kernels"][kernel_idx][row]):
                                            updated_filters[filter_idx]["kernels"][kernel_idx][row][col] = float(cell_value) if cell_value is not None else 0.0

                    # Guardar stride, padding, activation, kernel_size
                    if stride_h_ids and stride_h_values and stride_w_values:
                        for stride_h_id, stride_h_val, stride_w_val in zip(stride_h_ids, stride_h_values, stride_w_values):
                            f_idx = stride_h_id["filter"]
                            if f_idx < len(updated_filters):
                                updated_filters[f_idx]["stride"] = [
                                    int(stride_h_val) if stride_h_val is not None else 1,
                                    int(stride_w_val) if stride_w_val is not None else 1
                                ]

                    if padding_values:
                        for i, padding_val in enumerate(padding_values):
                            if i < len(updated_filters) and padding_val is not None:
                                updated_filters[i]["padding"] = padding_val

                    if activation_values:
                        for i, activation_val in enumerate(activation_values):
                            if i < len(updated_filters) and activation_val is not None:
                                updated_filters[i]["activation"] = activation_val

                    if kernel_size_ids and kernel_size_values:
                        for size_id, size_val in zip(kernel_size_ids, kernel_size_values):
                            f_idx = size_id["filter"]
                            if f_idx < len(updated_filters) and size_val:
                                rows, cols = map(int, size_val.split("x"))
                                updated_filters[f_idx]["kernel_size"] = (rows, cols)

                    cnn_config_by_layer[layer_idx] = updated_filters
                    print(f"💾 Auto-guardado: Filtros CNN de capa {layer_idx} guardados al navegar")

        # Actualizar las capas con la configuración
        for idx, layer in enumerate(layers):
            if idx in config_by_layer:
                # Merge con config existente
                if "config" not in layer:
                    layer["config"] = {}
                layer["config"].update(config_by_layer[idx])

            # Guardar filtros de CNN si esta capa es ConvolutionLayer
            if layer["type"] == "ConvolutionLayer" and idx in cnn_config_by_layer:
                if "config" not in layer:
                    layer["config"] = {}
                layer["config"]["filters"] = cnn_config_by_layer[idx]

        return layers


    # Callback: Navegar entre pasos
    @callback(
        Output("current-step", "data"),
        [Input("prev-step-btn", "n_clicks"),
         Input("next-step-btn", "n_clicks"),
         Input({"type": "layer-node", "index": ALL}, "n_clicks")],
        [State("current-step", "data"),
         State("architecture-layers", "data")],
        prevent_initial_call=True
    )
    def navigate_steps(prev_clicks, next_clicks, node_clicks, current_step, layers):
        triggered = ctx.triggered_id

        if not triggered:
            return no_update

        # Obtener número total de capas
        max_step = len(layers) - 1 if layers else 0

        # Click en nodo directo
        if isinstance(triggered, dict) and triggered.get("type") == "layer-node":
            target_step = triggered["index"]
            # Asegurar que esté en rango
            return min(max(0, target_step), max_step)

        # Botones de navegación
        if triggered == "prev-step-btn":
            return max(0, (current_step or 0) - 1)
        elif triggered == "next-step-btn":
            new_step = (current_step or 0) + 1
            # No permitir ir más allá del último paso
            return min(new_step, max_step)

        return no_update


    # Callback: Probar configuración del modelo (ahora también habilita entrenamiento en la nube)
    @callback(
        Output({"type": "test-config-result", "model": MATCH}, "children"),
        Output({"type": "btn-cloud-training", "model": MATCH}, "disabled", allow_duplicate=True),
        Output({"type": "cloud-training-hint", "model": MATCH}, "children", allow_duplicate=True),
        Output({"type": "model-validation-status", "model": MATCH}, "data", allow_duplicate=True),
        Input({"type": "test-config-btn", "model": MATCH}, "n_clicks"),
        [State({"type": "layer-config-input", "layer_index": ALL, "field": ALL}, "value"),
         State({"type": "layer-config-input", "layer_index": ALL, "field": ALL}, "id"),
         State({"type": "conv-filters-store", "layer": ALL}, "data"),
         State({"type": "conv-filters-store", "layer": ALL}, "id"),
         State({"type": "kernel-cell", "filter": ALL, "kernel": ALL, "row": ALL, "col": ALL}, "value"),
         State({"type": "kernel-cell", "filter": ALL, "kernel": ALL, "row": ALL, "col": ALL}, "id"),
         State({"type": "stride-h", "filter": ALL}, "value"),
         State({"type": "stride-w", "filter": ALL}, "value"),
         State({"type": "stride-h", "filter": ALL}, "id"),
         State({"type": "padding", "filter": ALL}, "value"),
         State({"type": "activation", "filter": ALL}, "value"),
         State({"type": "kernel-size-dropdown", "filter": ALL}, "value"),
         State({"type": "kernel-size-dropdown", "filter": ALL}, "id"),
         State("architecture-layers", "data"),
         State("model-type", "data"),
         State("classifier-type-store", "data"),
         State("selected-dataset", "data"),
         State({"type": "hyperparam-epochs", "model": MATCH}, "value"),
         State({"type": "hyperparam-batch-size", "model": MATCH}, "value"),
         State({"type": "hyperparam-lr", "model": MATCH}, "value")],
        prevent_initial_call=True
    )
    def test_model_configuration(n_clicks, input_values, input_ids, cnn_filters_data, cnn_filters_ids,
                                  kernel_values, kernel_ids, stride_h_values, stride_w_values, stride_ids,
                                  padding_values, activation_values, kernel_size_values, kernel_size_ids,
                                  layers, model_type, classifier_type, selected_dataset,
                                  epochs, batch_size, learning_rate):
        """
        Validación y prueba de configuración del modelo interactivo.
        
        Paso 1: Obtener canales y clases del dataset
        Paso 2: Validar el modelo con las clases Pydantic
        Paso 3: Instanciar el modelo en el experimento (P300 o InnerSpeech según classifier_type)
        Paso 4: Verificación de compilación con mini-entrenamiento

        Args:
            classifier_type: "P300" o "InnerSpeech" - determina qué método del experimento usar
            selected_dataset: Path del dataset seleccionado para obtener metadata
            epochs: Número de épocas proporcionado por el usuario
            batch_size: Tamaño de batch proporcionado por el usuario  
            learning_rate: Tasa de aprendizaje proporcionada por el usuario
        """
        if not n_clicks:
            return no_update, no_update, no_update, no_update

        from backend.classes.ClasificationModel.ClassifierSchemaFactory import ClassifierSchemaFactory
        from backend.classes.Experiment import Experiment
        from pydantic import ValidationError
        from shared.fileUtils import get_dataset_metadata
        import copy

        # ✅ VALIDAR HIPERPARÁMETROS DEL USUARIO (sin fallbacks)
        if epochs is None or epochs <= 0:
            return (dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Error: Debes especificar un número válido de épocas (mayor a 0)"
            ], color="warning", dismissable=True), True, "Configura hiperparámetros válidos", False)
        
        if batch_size is None or batch_size <= 0:
            return (dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Error: Debes especificar un tamaño de batch válido (mayor a 0)"
            ], color="warning", dismissable=True), True, "Configura hiperparámetros válidos", False)
        
        if learning_rate is None or learning_rate <= 0 or learning_rate > 1.0:
            return (dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Error: Debes especificar un learning rate válido (entre 0 y 1)"
            ], color="warning", dismissable=True), True, "Configura hiperparámetros válidos", False)

        try:
            _arch_log("Iniciando validación de configuración →", model_type, "classifier_type=", classifier_type)
            _arch_log(f"Hiperparámetros del usuario: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
            if layers:
                for idx, layer in enumerate(layers):
                    ltype = layer.get("type")
                    lcfg = layer.get("config", {})
                    summary = []
                    if ltype in ("DenseLayer", "LSTMLayer", "GRULayer"):
                        units = lcfg.get("units") or lcfg.get("hidden_size")
                        if units:
                            summary.append(f"units={units}")
                    if ltype == "ConvolutionLayer":
                        filters = lcfg.get("filters", [])
                        summary.append(f"filters={len(filters)}")
                    act = lcfg.get("activation")
                    if isinstance(act, dict):
                        act = act.get("kind")
                    if act:
                        summary.append(f"act={act}")
                    _arch_log(f"Layer {idx+1}: {ltype}", "|", ", ".join(summary) if summary else "(sin detalle)")
            # Crear copia de layers para evitar mutaciones
            layers = copy.deepcopy(layers) if layers else []

            # ===== PASO 0: OBTENER CANALES Y CLASES DEL DATASET =====

            # Verificar que hay un dataset seleccionado
            if not selected_dataset:
                return (dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Error: No hay un dataset seleccionado. Por favor selecciona un dataset primero."
                ], color="warning", dismissable=True), True, "Selecciona un dataset para habilitar el entrenamiento en la nube", False)

            # Obtener metadata del dataset
            try:
                metadata = get_dataset_metadata(selected_dataset)
                channel_names = metadata.get("channel_names") or metadata.get("channel_name_union") or []
                classes = metadata.get("classes") or []

                # ===== FIX: Calcular num_classes según el tipo de clasificador =====
                if classifier_type == "InnerSpeech":
                    # Inner Speech: Multiclase sin rest/none/unlabeled
                    excluded_classes = ["rest", "none", "unlabeled"]
                    original_classes = classes.copy()
                    classes = [cls for cls in classes if cls.lower() not in excluded_classes]
                    if len(classes) < len(original_classes):
                        print(f"🗑️  [Inner Speech] Clases excluidas: {set(original_classes) - set(classes)}")
                        print(f"✅ [Inner Speech] Clases para entrenamiento: {classes}")
                    num_classes = len(classes)
                elif classifier_type == "P300":
                    # P300: Clasificación binaria (0=NonTarget/rest, 1=Target)
                    num_classes = 2
                    print(f"✅ [P300] Clasificación binaria: 2 clases (NonTarget/Target)")
                else:
                    # Fallback: usar todas las clases del dataset
                    num_classes = len(classes)

                num_channels = len(channel_names)

                if num_channels <= 0:
                    return (dbc.Alert([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        f"Error: El dataset no tiene canales válidos. Metadata: {metadata}"
                    ], color="danger", dismissable=True), True, "Dataset sin canales válidos", False)

                if num_classes <= 0:
                    return (dbc.Alert([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        f"Error: El dataset no tiene clases válidas. Metadata: {metadata}"
                    ], color="danger", dismissable=True), True, "Dataset sin clases válidas", False)

                print(f"📊 Dataset metadata: {num_channels} canales, {num_classes} clases ({classifier_type})")

            except Exception as meta_err:
                return (dbc.Alert([
                    html.I(className="fas fa-times-circle me-2"),
                    f"Error al leer metadata del dataset: {str(meta_err)}"
                ], color="danger", dismissable=True), True, "Error leyendo metadata", False)

            # ===== PASO 1: RECOPILAR CONFIGURACIÓN =====

            # Verificar que hay capas
            if not layers or len(layers) == 0:
                return (dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Error: Debes agregar al menos una capa a la arquitectura"
                ], color="danger", dismissable=True), True, "Agrega capas para habilitar el entrenamiento en la nube", False)

            # Construir configuración desde los inputs
            # Los input_ids son dicts con {"type": "layer-config-input", "layer_index": i, "field": "field_name"}
            config_by_layer = {}
            if input_ids and input_values:
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

            # Guardar filtros de CNN por layer_index CON AUTO-GUARDADO
            cnn_config_by_layer = {}

            # DEBUG: Ver qué stores CNN están disponibles
            print(f"🔍 DEBUG (test): cnn_filters_ids = {cnn_filters_ids}")
            print(f"🔍 DEBUG (test): Número de stores CNN = {len(cnn_filters_data) if cnn_filters_data else 0}")
            for i, data in enumerate(cnn_filters_data or []):
                print(f"🔍 DEBUG (test): Store {i} tiene {len(data) if data else 0} filtros")

            if cnn_filters_ids and cnn_filters_data:
                for filter_id, filter_data in zip(cnn_filters_ids, cnn_filters_data):
                    layer_idx = filter_id["layer"]
                    if filter_data:  # Solo si hay filtros definidos
                        import copy
                        updated_filters = copy.deepcopy(filter_data)

                        # AUTO-GUARDAR: Capturar valores actuales de las matrices de kernels
                        if kernel_ids and kernel_values:
                            for cell_id, cell_value in zip(kernel_ids, kernel_values):
                                filter_idx = cell_id["filter"]
                                kernel_idx = cell_id["kernel"]
                                row = cell_id["row"]
                                col = cell_id["col"]

                                if filter_idx < len(updated_filters):
                                    if "kernels" not in updated_filters[filter_idx]:
                                        updated_filters[filter_idx]["kernels"] = [[], [], []]

                                    while len(updated_filters[filter_idx]["kernels"]) <= kernel_idx:
                                        updated_filters[filter_idx]["kernels"].append([])

                                    kernel_matrix = updated_filters[filter_idx]["kernels"][kernel_idx]
                                    while len(kernel_matrix) <= row:
                                        kernel_matrix.append([])

                                    while len(kernel_matrix[row]) <= col:
                                        kernel_matrix[row].append(0.0)

                                    kernel_matrix[row][col] = float(cell_value) if cell_value is not None else 0.0

                        # AUTO-GUARDAR: stride
                        if stride_ids and stride_h_values and stride_w_values:
                            for stride_id, h_val, w_val in zip(stride_ids, stride_h_values, stride_w_values):
                                filter_idx = stride_id["filter"]
                                if filter_idx < len(updated_filters) and h_val is not None and w_val is not None:
                                    updated_filters[filter_idx]["stride"] = [int(h_val), int(w_val)]

                        # AUTO-GUARDAR: padding
                        if padding_values:
                            for i, padding_val in enumerate(padding_values):
                                if i < len(updated_filters) and padding_val is not None:
                                    updated_filters[i]["padding"] = padding_val

                        # AUTO-GUARDAR: activation
                        if activation_values:
                            for i, activation_val in enumerate(activation_values):
                                if i < len(updated_filters) and activation_val is not None:
                                    updated_filters[i]["activation"] = activation_val

                        # AUTO-GUARDAR: kernel_size
                        if kernel_size_ids and kernel_size_values:
                            for size_id, size_val in zip(kernel_size_ids, kernel_size_values):
                                f_idx = size_id["filter"]
                                if f_idx < len(updated_filters) and size_val:
                                    rows, cols = map(int, size_val.split("x"))
                                    updated_filters[f_idx]["kernel_size"] = (rows, cols)

                        cnn_config_by_layer[layer_idx] = updated_filters
                        print(f"💾 Auto-guardado (test): Filtros CNN de capa {layer_idx} guardados")

            # Actualizar las capas con la configuración recopilada
            for idx, layer in enumerate(layers):
                if idx in config_by_layer:
                    layer["config"] = config_by_layer[idx]

                # Guardar filtros de CNN si esta capa es ConvolutionLayer
                if layer["type"] == "ConvolutionLayer" and idx in cnn_config_by_layer:
                    if "config" not in layer:
                        layer["config"] = {}
                    layer["config"]["filters"] = cnn_config_by_layer[idx]

            # ===== PASO 2: VALIDACIÓN (después de asignar filtros) =====

            # Verificación adicional: Reportar qué capas Conv no tienen filtros
            conv_layers_without_filters = []
            for idx, layer in enumerate(layers):
                if layer["type"] == "ConvolutionLayer":
                    filters = layer.get("config", {}).get("filters", [])
                    if not filters or len(filters) == 0:
                        conv_layers_without_filters.append(idx + 1)  # +1 para mostrar número humano

            if conv_layers_without_filters:
                layer_numbers = ", ".join([f"#{n}" for n in conv_layers_without_filters])
                return (dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    html.Div([
                        html.Strong(f"⚠️ Capas Convolucionales sin filtros: {layer_numbers}"),
                        html.Br(),
                        html.Small("Navega a cada capa Conv y agrega al menos un filtro (botón '+ Agregar Filtro')")
                    ])
                ], color="warning", dismissable=True), True, "Completa filtros CNN para habilitar entrenamiento", False)

            is_valid, error_msg = validate_complete_architecture(layers, model_type)
            if not is_valid:
                return (dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Arquitectura incompleta: {error_msg}"
                ], color="warning", dismissable=True), True, "Arquitectura incompleta", False)

            # Construir el diccionario completo del modelo según el tipo
            # Pasar num_channels y num_classes del dataset + hiperparámetros del usuario
            try:
                model_config = build_model_config_from_layers(
                    layers, model_type, num_channels, num_classes,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    classifier_type=classifier_type
                )
                _arch_log("Hparams del usuario:", f"epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
            except ValueError as ve:
                return (dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Error en configuración: {str(ve)}"
                ], color="warning", dismissable=True), True, "Configura parámetros válidos", False)

            # Obtener la clase Pydantic correspondiente
            available_classifiers = ClassifierSchemaFactory.available_classifiers
            classifier_class = available_classifiers.get(model_type)

            if not classifier_class:
                return (dbc.Alert([
                    html.I(className="fas fa-times-circle me-2"),
                    f"Error: Modelo '{model_type}' no encontrado"
                ], color="danger", dismissable=True), True, "Modelo no encontrado", False)

            # Validar con Pydantic
            try:
                validated_instance = classifier_class(**model_config)
                _arch_log("Instancia validada correctamente para", model_type)
            except ValidationError as ve:
                error_details = "\n".join([f"• {err['loc'][0]}: {err['msg']}" for err in ve.errors()])
                return (dbc.Alert([
                    html.I(className="fas fa-times-circle me-2"),
                    html.Div([
                        html.Strong("Errores de validación:"),
                        html.Pre(error_details, style={"fontSize": "12px", "marginTop": "10px"})
                    ])
                ], color="danger", dismissable=True), True, "Errores de validación", False)

            # ===== PASO 2: VERIFICACIÓN DE COMPILACIÓN =====
            # Primero verificar que el modelo compila ANTES de guardarlo en el experimento
            compilation_ok = False
            compilation_error = None
            experiment_type_msg = "Habla Interna" if classifier_type == "InnerSpeech" else "P300"

            try:
                # Validar que existe dataset
                if not selected_dataset:
                    compilation_error = "No se ha seleccionado un dataset"
                    print(f"⚠️ [INTERNO] No hay dataset seleccionado")
                else:
                    from pathlib import Path
                    from backend.classes.dataset import Dataset
                    
                    # Construir path del dataset con la estructura correcta
                    # Usar el método get_events_by_class para encontrar los eventos automáticamente
                    dataset_path = f"Aux/{selected_dataset}"

                    # Verificar que existen eventos usando get_events_by_class
                    events_result = Dataset.get_events_by_class(dataset_path, class_name=None)
                    
                    if events_result["status"] != 200 or not events_result["event_files"]:
                        compilation_error = f"No se encontraron eventos en el dataset: {events_result['message']}"
                        print(f"⚠️ [INTERNO] {compilation_error}")
                    else:
                        events_dir = events_result["events_dir"]
                        n_events = events_result["n_events"]
                        
                        print(f"{'='*70}")
                        print(f"🔧 VERIFICANDO CONFIGURACIÓN DEL MODELO")
                        print(f"{'='*70}")
                        print(f"📦 Modelo: {model_type}")
                        print(f"🎯 Experimento: {experiment_type_msg}")
                        print(f"📂 Dataset: {dataset_path}")
                        print(f"📁 Eventos: {events_dir} ({n_events} eventos)")
                        print(f"\n⏳ Preparando datos de prueba...")

                        # Determinar model_type a partir de classifier_type
                        model_type_for_pipeline = "p300" if classifier_type == "P300" else "inner"
                        
                        # Generar mini dataset - usar método apropiado según tipo de modelo
                        # Según ClassifierSchemaFactory:
                        # - Modelos clásicos (formulario simple): SVM, RandomForest
                        # - Redes neuronales (sistema interactivo): LSTM, GRU, CNN, SVNN
                        
                        mini_dataset = Experiment.generate_pipeline_dataset(
                            dataset_path=dataset_path,
                            n_train=10,
                            n_test=5,
                            selected_classes=None,
                            force_recalculate=False,
                            verbose=False,
                            model_type=model_type_for_pipeline  # ✅ Pasar model_type
                        )
                            
                       

                        print(f"✅ Datos preparados: {mini_dataset['n_train']} train + {mini_dataset['n_test']} test")

                        if mini_dataset["n_train"] < 3:
                            compilation_error = "El pipeline de preprocesamiento no generó suficientes datos válidos"
                            print(f"⚠️ [INTERNO] Dataset insuficiente: {mini_dataset['n_train']} ejemplos")
                        else:
                            # Para modelos secuenciales (LSTM/GRU), ajustar input_feature_dim a la dimensionalidad real
                            if model_type in ["LSTM", "GRU"] and hasattr(validated_instance, 'encoder'):
                                import numpy as np
                                try:
                                    # Verificar si los datos son rutas de archivos o arrays directos
                                    if isinstance(mini_dataset['train_data'][0], str):
                                        # Rutas de archivos (generate_pipeline_dataset)
                                        sample_data_path = mini_dataset['train_data'][0]
                                        sample_data = np.load(sample_data_path)
                                    else:
                                        # Arrays directos (generate_model_dataset)
                                        sample_data = mini_dataset['train_data'][0]

                                    # Inferir dimensionalidad según shape:
                                    # - (timesteps, features): usar features
                                    # - (timesteps, features, channels): aplanar a features * channels
                                    if sample_data.ndim == 3:
                                        # Shape: (timesteps, features, channels)
                                        # LSTM necesita (timesteps, features_aplanadas)
                                        actual_feature_dim = sample_data.shape[1] * sample_data.shape[2]
                                    elif sample_data.ndim == 2:
                                        # Shape: (timesteps, features) - ya está aplanado
                                        actual_feature_dim = sample_data.shape[1]
                                    else:
                                        actual_feature_dim = validated_instance.encoder.input_feature_dim

                                    # Actualizar input_feature_dim si es diferente
                                    if actual_feature_dim != validated_instance.encoder.input_feature_dim:
                                        print(f"[{model_type}] Ajustando input_feature_dim: {validated_instance.encoder.input_feature_dim} -> {actual_feature_dim} (shape: {sample_data.shape})")
                                        validated_instance.encoder.input_feature_dim = actual_feature_dim
                                except Exception as dim_err:
                                    print(f"⚠️ No se pudo inferir dimensionalidad: {dim_err}")

                            # Verificar si el modelo tiene método train
                            if hasattr(classifier_class, 'train'):
                                print(f"🔧 [INTERNO] Ejecutando mini-entrenamiento...")

                                # Los modelos usan parámetros: xTrain, yTrain, xTest, yTest
                                # (epochs, batch_size, etc. ya están en validated_instance)
                                try:
                                    metrics = classifier_class.train(
                                        validated_instance,
                                        xTrain=mini_dataset["train_data"],
                                        yTrain=mini_dataset["train_labels"],
                                        xTest=mini_dataset["test_data"],
                                        yTest=mini_dataset["test_labels"]
                                    )
                                    compilation_ok = True
                                    _arch_log("Compilación y mini-entrenamiento OK")
                                except Exception as train_err:
                                    compilation_error = "Error al compilar el modelo con los datos procesados"
                                    print(f"❌ [INTERNO] Error en compilación: {train_err}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                # Si no tiene método train, asumimos OK si llegó aquí
                                compilation_ok = True
                                print(f"✅ [INTERNO] Modelo sin método train(), validación OK")

            except Exception as test_err:
                compilation_error = "Error en el pipeline de preprocesamiento. Revisa la configuración de filtros y transformadas"
                print(f"⚠️ [INTERNO] Error en verificación: {test_err}")
                import traceback
                traceback.print_exc()

            # Si hubo error de compilación, NO guardar en experimento
            if compilation_error:
                return (dbc.Alert([
                    html.I(className="fas fa-times-circle me-2"),
                    html.Div([
                        html.Strong("✗ Error de compilación"),
                        html.Br(),
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle me-1", style={"fontSize": "12px"}),
                            html.Small(compilation_error, style={"fontWeight": "500"})
                        ], className="mt-2"),
                        html.Div([
                            html.I(className="fas fa-info-circle me-1", style={"fontSize": "12px"}),
                            html.Small("El modelo NO fue guardado. Verifica:")
                        ], className="mt-2", style={"opacity": "0.9"}),
                        html.Ul([
                            html.Li("Configuración de filtros y transformadas", style={"fontSize": "13px"}),
                            html.Li("Que el dataset tenga eventos procesados", style={"fontSize": "13px"}),
                            html.Li("Compatibilidad del pipeline con el modelo", style={"fontSize": "13px"})
                        ], className="mb-0 mt-1", style={"opacity": "0.8"})
                    ])
                ], color="danger", dismissable=True, duration=12000), True, "Errores de compilación", False)

            # ===== PASO 3: INSTANCIAR EN EXPERIMENTO (solo si compilación OK) =====
            try:
                # Agregar clasificador al experimento según el tipo
                if classifier_type == "InnerSpeech":
                    Experiment.add_inner_speech_classifier(validated_instance)
                    _arch_log(f"{model_type} agregado al experimento como InnerSpeechClassifier")
                else:  # P300 por defecto
                    Experiment.add_P300_classifier(validated_instance)
                    _arch_log(f"{model_type} agregado al experimento como P300Classifier")

                # ===== PASO 4: ENVIAR EXPERIMENTO A LA NUBE (SYNC) =====
                from shared.experimentUploader import upload_experiment_to_cloud_sync, download_trained_model
                import threading

                print(f"\n{'='*70}")
                print(f"[TestConfig] 📤 Enviando experimento a la nube después de probar configuración...")
                print(f"{'='*70}")

                cloud_upload_msg = None
                try:
                    cloud_result = upload_experiment_to_cloud_sync(
                        classifier_type=classifier_type,
                        model_name=model_type,
                        dataset_name=selected_dataset
                    )

                    if cloud_result["success"]:
                        job_id = cloud_result["jobid"]
                        print(f"[TestConfig] ✅ Experimento enviado a la nube exitosamente")
                        print(f"[TestConfig] 🆔 Job ID: {job_id}")
                        cloud_upload_msg = f"Enviado a la nube - Job ID: {job_id}"

                        # ===== PASO 4.5: DESCARGAR MODELO EN BACKGROUND =====
                        # Iniciar descarga en un thread separado para no bloquear la UI
                        def download_in_background():
                            import time
                            # Esperar 5 segundos antes de intentar descargar (dar tiempo al servidor)
                            print(f"[TestConfig] ⏳ Esperando 5s antes de descargar modelo...")
                            time.sleep(5)

                            download_result = download_trained_model(
                                job_id=job_id,
                                classifier_type=classifier_type
                            )

                            if download_result["success"]:
                                print(f"[TestConfig] 🎉 Modelo descargado y guardado en: {download_result['model_path']}")
                            else:
                                print(f"[TestConfig] ⚠️ No se pudo descargar el modelo: {download_result.get('error')}")

                        # Iniciar thread
                        download_thread = threading.Thread(target=download_in_background, daemon=True)
                        download_thread.start()
                        # ===== FIN PASO 4.5 =====

                    else:
                        print(f"[TestConfig] ⚠️ No se pudo enviar a la nube")
                        print(f"[TestConfig] Error: {cloud_result.get('error', 'Desconocido')}")
                        cloud_upload_msg = "No se pudo enviar a la nube (revisa la conexión)"
                except Exception as e:
                    print(f"[TestConfig] ⚠️ Error al enviar a la nube: {e}")
                    cloud_upload_msg = "Error al enviar a la nube"

                print(f"{'='*70}\n")
                # ===== FIN PASO 4 =====

                # Éxito total con detalles de compilación
                success_content = [
                    html.I(className="fas fa-check-circle me-2"),
                    html.Div([
                        html.Strong(f"✓ {model_type} configurado y compilado exitosamente"),
                        html.Br(),
                        html.Div([
                            html.I(className="fas fa-cogs me-1", style={"fontSize": "12px"}),
                            html.Small(f"Modelo compilado con datos reales del pipeline")
                        ], className="mt-2"),
                        html.Div([
                            html.I(className="fas fa-database me-1", style={"fontSize": "12px"}),
                            html.Small(f"{num_channels} canales • {num_classes} clases • Experimento: {experiment_type_msg}")
                        ], className="mt-1", style={"opacity": "0.8"}),
                        html.Div([
                            html.I(className="fas fa-cloud-upload-alt me-1", style={"fontSize": "12px"}),
                            html.Small(cloud_upload_msg, style={"opacity": "0.9"})
                        ], className="mt-1"),
                        html.Div([
                            html.I(className="fas fa-check me-1", style={"fontSize": "12px", "color": "#28a745"}),
                            html.Small("Listo para entrenamiento completo", style={"color": "#28a745", "fontWeight": "500"})
                        ], className="mt-2")
                    ])
                ]

                return (dbc.Alert(success_content, color="success", dismissable=True, duration=8000), False, "Listo: puedes entrenar el modelo en la nube", True)

            except Exception as exp_err:
                print(f"❌ Error al agregar al experimento: {exp_err}")
                return (dbc.Alert([
                    html.I(className="fas fa-times-circle me-2"),
                    f"Error al guardar en experimento: {str(exp_err)}"
                ], color="danger", dismissable=True), True, "Error guardando experimento", False)

        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            import traceback
            traceback.print_exc()
            return (dbc.Alert([
                html.I(className="fas fa-times-circle me-2"),
                f"Error inesperado: {str(e)}"
            ], color="danger", dismissable=True), True, "Error inesperado", False)


# Registrar callbacks al importar
register_interactive_callbacks()
