"""
Módulo para enviar configuraciones de experimentos al servidor de entrenamiento en la nube.
Mapea la configuración interna al esquema esperado por la API /upload-expr.
"""

import requests
import os
from typing import Dict, List, Optional, Any
import threading
import json
from backend.classes.Experiment import Experiment
from shared.fileUtils import get_dataset_metadata

# WebSocket para seguimiento de progreso
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("[UPLOAD] ⚠️ websocket-client no está instalado. No se podrá hacer seguimiento en tiempo real.")


def extract_experiment_config(
    classifier_type: str,
    model_name: str,
    dataset_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extrae la configuración completa del experimento actual.

    Args:
        classifier_type: "P300" o "InnerSpeech"
        model_name: Nombre del modelo seleccionado (ej: "SVM", "CNN")

    Returns:
        Dict con {filters, transform, model_config, dataset_name, metadata}

    Raises:
        ValueError: Si no hay experimento cargado, dataset, o modelo configurado
    """
    # Cargar experimento actual
    try:
        experiment = Experiment._load_latest_experiment()
    except Exception as e:
        raise ValueError(f"No se pudo cargar el experimento: {e}")

    # Obtener nombre del dataset (de parámetro o del experimento)
    if dataset_name:
        # Usar dataset del parámetro (viene del store "selected-dataset")
        pass
    elif experiment.dataset and experiment.dataset.get("name"):
        # Usar dataset del experimento
        dataset_name = experiment.dataset.get("name")
    else:
        raise ValueError("No hay dataset cargado. Selecciona un dataset en 'Cargar Datos'")

    # Obtener metadata del dataset
    try:
        metadata = get_dataset_metadata(dataset_name)
    except Exception as e:
        print(f"[UPLOAD] ⚠️ No se pudo obtener metadata del dataset: {e}")
        metadata = {}

    # Obtener configuración del clasificador correspondiente
    if classifier_type == "P300":
        classifier_config = experiment.P300Classifier or {}
    elif classifier_type == "InnerSpeech":
        classifier_config = experiment.innerSpeachClassifier or {}
    else:
        raise ValueError(f"Tipo de clasificador inválido: {classifier_type}")

    # Verificar que el modelo existe en la configuración
    if model_name not in classifier_config:
        raise ValueError(f"Modelo '{model_name}' no está configurado en el experimento")

    model_config = classifier_config[model_name]

    # DEBUG: Imprimir la estructura del modelo
    import json
    print(f"\n[DEBUG] 🔍 Estructura del modelo '{model_name}':")
    print(json.dumps(model_config, indent=2, default=str, ensure_ascii=False))
    print()

    # Verificar que hay transformada
    if "transform" not in model_config or not model_config["transform"]:
        raise ValueError(f"El modelo '{model_name}' no tiene transformada aplicada")

    # Extraer config del modelo (todo excepto "transform")
    model_only_config = {k: v for k, v in model_config.items() if k != "transform"}

    return {
        "filters": experiment.filters or [],
        "transform": model_config.get("transform"),
        "model_config": model_only_config,  # Config sin la transformada
        "dataset_name": dataset_name,
        "metadata": metadata,
        "classifier_type": classifier_type
    }


def map_filters_to_api(filters: List[dict], sampling_rate: float) -> List[dict]:
    """
    Convierte filtros del formato interno al formato de API.

    Args:
        filters: Lista de diccionarios con configuraciones de filtros
        sampling_rate: Frecuencia de muestreo en Hz (ej: 1000)

    Returns:
        Lista de filtros en formato API
    """
    api_filters = []

    # DEBUG: Imprimir filtros recibidos
    if filters:
        import json
        print(f"\n[DEBUG] 🔍 Filtros recibidos ({len(filters)}):")
        for i, f in enumerate(filters):
            print(f"  Filtro {i}: {json.dumps(f, indent=4, default=str, ensure_ascii=False)}")
        print()

    for filter_config in filters:
        filter_type = filter_config.get("filter_type", "").lower()

        if filter_type in ["bandpass", "lowpass", "highpass"]:
            # BandPass filter
            freq = filter_config.get("freq")

            # Asegurar que freq sea una lista [low, high]
            if isinstance(freq, (int, float)):
                freq = [0, freq] if filter_type == "lowpass" else [freq, sampling_rate / 2]
            elif isinstance(freq, (list, tuple)):
                freq = list(freq)
            else:
                print(f"[UPLOAD] ⚠️ Frecuencia inválida en filtro {filter_type}: {freq}")
                continue

            api_filter = {
                "filter_type": filter_type,
                "sp": 1 / sampling_rate if sampling_rate > 0 else 0.001,
                "freq": freq,
                "method": filter_config.get("method", "fir"),
                "order": filter_config.get("order", 4),
                "phase": filter_config.get("phase", "zero"),
                "fir_window": filter_config.get("fir_window", "hamming")
            }
            api_filters.append(api_filter)

        elif filter_type == "notch":
            # Notch filter
            api_filter = {
                "filter_type": "notch",
                "sp": 1 / sampling_rate if sampling_rate > 0 else 0.001,
                "freqs": filter_config.get("freqs", 50),  # Default 50 Hz (línea eléctrica)
                "quality": filter_config.get("quality", 30),
                "method": filter_config.get("method", "fir")
            }
            api_filters.append(api_filter)

        elif filter_type == "ica":
            # ICA filter
            api_filter = {
                "filter_type": "ICA",
                "sp": 1 / sampling_rate if sampling_rate > 0 else 0.001,
                "numeroComponentes": filter_config.get("numeroComponentes", 1),
                "method": filter_config.get("method", "fastica"),
                "random_state": filter_config.get("random_state", 0),
                "max_iter": filter_config.get("max_iter", 200)
            }
            api_filters.append(api_filter)

        else:
            print(f"[UPLOAD] ⚠️ Tipo de filtro desconocido: {filter_type}")

    return api_filters


def map_transform_to_api(
    transform: dict,
    model_type: str,
    all_classes: list,
    sampling_rate: float
) -> dict:
    """
    Convierte transformada del formato interno al formato de API.

    Args:
        transform: Configuración de la transformada
        model_type: "p300" o "inner" (para Inner Speech)
        all_classes: Lista de clases del dataset
        sampling_rate: Frecuencia de muestreo en Hz

    Returns:
        Dict con transformada en formato API
    """
    # DEBUG: Imprimir transformada recibida
    import json
    print(f"\n[DEBUG] 🔍 Transformada recibida:")
    print(json.dumps(transform, indent=2, default=str, ensure_ascii=False))
    print()

    # La transformada viene como {"WindowingTransform": {...config...}}
    # Necesitamos extraer el tipo y la configuración
    if not transform or not isinstance(transform, dict):
        print(f"[UPLOAD] ⚠️ Transformada vacía o inválida")
        return {
            "transform_type": "windowing",
            "sp": 1 / sampling_rate if sampling_rate > 0 else 0.001,
            "model_type": model_type,
            "all_classes": all_classes if all_classes else [],
            "window_size": 256
        }

    # Obtener el nombre de la clase de transformada (primera clave)
    transform_class_name = list(transform.keys())[0]
    transform_config = transform[transform_class_name]

    print(f"[DEBUG] 🔍 Clase de transformada: {transform_class_name}")
    print(f"[DEBUG] 🔍 Config interna: {json.dumps(transform_config, indent=2, default=str, ensure_ascii=False)}")

    # Mapear nombre de clase a transform_type
    # WindowingTransform -> windowing
    # FFTTransform -> FFT
    # DCTTransform -> DCT
    # WaveletTransform -> wavelets
    if "Windowing" in transform_class_name:
        transform_type = "WINDOWING"
    elif "FFT" in transform_class_name:
        transform_type = "FFT"
    elif "DCT" in transform_class_name:
        transform_type = "DCT"
    elif "Wavelet" in transform_class_name:
        transform_type = "WAVELETS"
    else:
        print(f"[UPLOAD] ⚠️ Clase de transformada desconocida: {transform_class_name}")
        transform_type = "WINDOWING"

    print(f"[DEBUG] 🔍 transform_type mapeado: {transform_type}")

    sp = 1 / sampling_rate if sampling_rate > 0 else 0.001

    # Base común para todas las transformadas
    api_transform = {
        "sp": sp,
        "model_type": model_type,
        # ✅ Filtrar "rest" si es modelo Inner
        "all_classes": [
            c for c in (all_classes if all_classes else [])
            if not (model_type == "inner" and str(c).lower() == "rest")
        ]
    }

    if transform_type == "FFT":
        api_transform.update({
            "transform_type": "FFT",
            "window": transform_config.get("window", "hann"),
            "nfft": transform_config.get("nfft", 1),
            "overlap": transform_config.get("overlap", 0.5),
            "frame_length": transform_config.get("frame_length", 256),
            "hop_samples": transform_config.get("hop_samples", 1)
        })

    elif transform_type == "DCT":
        api_transform.update({
            "transform_type": "DCT",
            "type": transform_config.get("type", 2),
            "norm": transform_config.get("norm", "ortho"),
            "axis": transform_config.get("axis", -1),
            "frame_length": transform_config.get("frame_length", 256),
            "overlap": transform_config.get("overlap", 0.5),
            "window": transform_config.get("window", "rectangular")
        })

    elif transform_type == "WAVELETS":
        api_transform.update({
            "transform_type": "wavelets",
            "wavelet": transform_config.get("wavelet", "db4"),
            "level": transform_config.get("level", 1),
            "mode": transform_config.get("mode", "symmetric"),
            "threshold": transform_config.get("threshold", 0),
            "frame_length": transform_config.get("frame_length", 256),
            "hop_samples": transform_config.get("hop_samples", 1),
            "overlap": transform_config.get("overlap", 0)
        })

    elif transform_type == "WINDOWING":
        api_transform.update({
            "transform_type": "windowing",
            "window_size": transform_config.get("window_size", 64)
        })

    else:
        print(f"[UPLOAD] ⚠️ Tipo de transformada desconocido: {transform_type}")
        # Retornar una transformada básica por defecto
        api_transform.update({
            "transform_type": "windowing",
            "window_size": 256
        })

    return api_transform


def map_model_to_api(model_config: dict, model_name: str) -> dict:
    """
    Convierte configuración de modelo al formato de API.

    Args:
        model_config: Configuración del modelo
        model_name: Nombre del modelo (ej: "SVM", "CNN", "LSTM")

    Returns:
        Dict con modelo en formato API
    """
    model_type_lower = model_name.lower()

    if model_type_lower == "svm":
        return {
            "model_type": "svm",
            "kernel": model_config.get("kernel", "linear"),
            "C": model_config.get("C", 1.0),
            "gamma": model_config.get("gamma", "scale"),
            "degree": model_config.get("degree", 3),
            "coef0": model_config.get("coef0", 0),
            "shrinking": model_config.get("shrinking", True),
            "tol": model_config.get("tol", 0.001),
            "max_iter": model_config.get("max_iter", -1),
            "probability": model_config.get("probability", False),
            "class_weight": model_config.get("class_weight", "balanced")
        }

    elif model_type_lower == "randomforest":
        return {
            "model_type": "random_forest",
            "max_depth": model_config.get("max_depth", 3),
            "n_estimators": model_config.get("n_estimators", 100),
            "random_state": model_config.get("random_state", 42)
        }

    elif model_type_lower == "cnn":
        return {
            "model_type": "cnn",
            "epochs": model_config.get("epochs", 50),
            "batch_size": model_config.get("batch_size", 32),
            "feature_extractor": model_config.get("feature_extractor", []),
            "flatten": model_config.get("flatten", {}),
            "fc_layers": model_config.get("fc_layers", []),
            "fc_activation_common": model_config.get("fc_activation_common", {"kind": "relu"}),
            "classification": model_config.get("classification", {}),
            "frame_context": model_config.get("frame_context", 8),
            "image_hw": model_config.get("image_hw", [64, 128]),
            "input_adapter": model_config.get("input_adapter", {})
        }

    elif model_type_lower in ["lstm", "gru"]:
        return {
            "model_type": model_type_lower,
            "epochs": model_config.get("epochs", 50),
            "batch_size": model_config.get("batch_size", 32),
            "encoder": model_config.get("encoder", {}),
            "pooling": model_config.get("pooling", {}),
            "fc_layers": model_config.get("fc_layers", []),
            "fc_activation_common": model_config.get("fc_activation_common", {"kind": "relu"}),
            "classification": model_config.get("classification", {}),
            "input_adapter": model_config.get("input_adapter", {})
        }

    elif model_type_lower == "svnn":
        return {
            "model_type": "svnn",
            "epochs": model_config.get("epochs", 50),
            "batch_size": model_config.get("batch_size", 16),
            "hidden_size": model_config.get("hidden_size", 64),
            "learning_rate": model_config.get("learning_rate", 0.001),
            "classification_units": model_config.get("classification_units", 2),
            "fc_activation_common": model_config.get("fc_activation_common", {"kind": "relu"}),
            "layers": model_config.get("layers", []),
            "input_adapter": model_config.get("input_adapter", {})
        }

    else:
        print(f"[UPLOAD] ⚠️ Tipo de modelo desconocido: {model_name}")
        # Retornar configuración básica de SVM por defecto
        return {
            "model_type": "svm",
            "kernel": "linear",
            "C": 1.0,
            "max_iter": 200,
            "probability": False,
            "tol": 0.001
        }


def build_experiment_payload(
    classifier_type: str,
    model_name: str,
    dataset_name: Optional[str] = None
) -> dict:
    """
    Construye el payload completo listo para enviar al endpoint /upload-expr.

    Args:
        classifier_type: "P300" o "InnerSpeech"
        model_name: Nombre del modelo seleccionado

    Returns:
        Dict con payload completo para la API

    Raises:
        ValueError: Si falta configuración necesaria
    """
    print(f"\n[UPLOAD] 📦 Construyendo payload para {classifier_type} - {model_name}")

    # Extraer configuración del experimento
    config = extract_experiment_config(classifier_type, model_name, dataset_name)

    # Obtener metadata
    metadata = config["metadata"]
    dataset_name = config["dataset_name"]

    # Obtener sampling rate (frecuencia de muestreo)
    sampling_rate = metadata.get("sampling_frequency_hz") or metadata.get("sfreq")
    if isinstance(sampling_rate, (list, tuple)):
        sampling_rate = sampling_rate[0] if sampling_rate else 1000
    sampling_rate = float(sampling_rate) if sampling_rate else 1000.0

    print(f"[UPLOAD] 📊 Dataset: {dataset_name}")
    print(f"[UPLOAD] 📊 Sampling rate: {sampling_rate} Hz")

    # Obtener clases del dataset
    all_classes = metadata.get("classes", [])
    print(f"[UPLOAD] 📊 Clases: {all_classes}")

    # Determinar model_type para transformadas (p300 vs inner)
    transform_model_type = "p300" if classifier_type == "P300" else "inner"

    # ===== VALIDACIÓN: Inner Speech NO debe incluir "rest" =====
    if transform_model_type == "inner" and "rest" in [str(c).lower() for c in all_classes]:
        print(f"[UPLOAD] ⚠️ ADVERTENCIA: Inner Speech NO debe incluir 'rest'. Filtrando...")
        all_classes = [c for c in all_classes if str(c).lower() != "rest"]
        print(f"[UPLOAD] 📊 Clases filtradas: {all_classes}")

    # Mapear filtros
    api_filters = map_filters_to_api(config["filters"], sampling_rate)
    print(f"[UPLOAD] 🔧 Filtros mapeados: {len(api_filters)}")

    # Mapear transformada
    api_transform = map_transform_to_api(
        config["transform"],
        transform_model_type,
        all_classes,
        sampling_rate
    )
    print(f"[UPLOAD] 🔄 Transformada: {api_transform.get('transform_type')}")

    # Mapear modelo
    api_model = map_model_to_api(config["model_config"], model_name)
    print(f"[UPLOAD] 🤖 Modelo: {api_model.get('model_type')}")

    # Construir payload final
    payload = {
        "data": "arabic_inner_speech",  # ✅ Dataset fijo para la nube
        "filters": api_filters,
        "transforms": [api_transform],  # Lista con una transformada
        "model": api_model
    }

    print(f"[UPLOAD] ✅ Payload construido exitosamente")

    # Imprimir el payload completo para debugging
    import json
    print(f"\n[UPLOAD] 📋 PAYLOAD COMPLETO:")
    print("="*60)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("="*60)

    return payload


def upload_experiment_to_cloud_sync(
    classifier_type: str,
    model_name: str,
    dataset_name: Optional[str] = None,
    server_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Envía el experimento al servidor (versión síncrona).

    Args:
        classifier_type: "P300" o "InnerSpeech"
        model_name: Nombre del modelo
        server_url: URL del servidor (opcional, usa variable de entorno si no se especifica)

    Returns:
        Dict con {"success": bool, "jobid": str, "message": str, "error": str}
    """
    if not server_url:
        server_url = os.getenv("EXTERNAL_SERVER_URL", "https://xx8mx485.usw3.devtunnels.ms:8000/")

    endpoint = f"{server_url.rstrip('/')}/upload-expr"

    print(f"\n{'='*60}")
    print(f"🚀 ENVIANDO EXPERIMENTO A LA NUBE")
    print(f"{'='*60}")
    print(f"[UPLOAD] 🔗 Endpoint: {endpoint}")
    print(f"[UPLOAD] 📝 Clasificador: {classifier_type}")
    print(f"[UPLOAD] 🤖 Modelo: {model_name}")

    try:
        # Construir payload
        payload = build_experiment_payload(classifier_type, model_name, dataset_name)

        # Enviar request
        print(f"\n[UPLOAD] 📤 Enviando request...")
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            # Éxito
            response_data = response.json() if response.text else {}

            # Parsear job_id correctamente del formato {"response":200,"job_id":"..."}
            jobid = None
            if isinstance(response_data, dict):
                jobid = response_data.get("job_id") or response_data.get("jobid")

            if not jobid:
                jobid = response.text

            print(f"\n[UPLOAD] ✅ ÉXITO!")
            print(f"[UPLOAD] 🆔 Job ID: {jobid}")
            print(f"{'='*60}\n")

            return {
                "success": True,
                "jobid": jobid,
                "response_data": response_data,  # Incluir data completa
                "message": f"Experimento enviado exitosamente. Job ID: {jobid}",
                "error": None
            }
        else:
            # Error del servidor
            error_text = response.text if response.text else f"HTTP {response.status_code}"
            print(f"\n[UPLOAD] ❌ ERROR DEL SERVIDOR")
            print(f"[UPLOAD] Status: {response.status_code}")
            print(f"[UPLOAD] Respuesta: {error_text}")
            print(f"{'='*60}\n")

            return {
                "success": False,
                "jobid": None,
                "message": "",
                "error": f"Error del servidor ({response.status_code}): {error_text[:200]}"
            }

    except ValueError as e:
        # Error de validación (falta configuración)
        print(f"\n[UPLOAD] ❌ ERROR DE VALIDACIÓN")
        print(f"[UPLOAD] {str(e)}")
        print(f"{'='*60}\n")

        return {
            "success": False,
            "jobid": None,
            "message": "",
            "error": f"Validación fallida: {str(e)}"
        }

    except requests.exceptions.Timeout:
        print(f"\n[UPLOAD] ⏱️ TIMEOUT")
        print(f"[UPLOAD] El servidor no respondió en 30 segundos")
        print(f"{'='*60}\n")

        return {
            "success": False,
            "jobid": None,
            "message": "",
            "error": "Timeout: El servidor no respondió en 30 segundos"
        }

    except requests.exceptions.RequestException as e:
        print(f"\n[UPLOAD] ❌ ERROR DE CONEXIÓN")
        print(f"[UPLOAD] {str(e)}")
        print(f"{'='*60}\n")

        return {
            "success": False,
            "jobid": None,
            "message": "",
            "error": f"Error de conexión: {str(e)}"
        }

    except Exception as e:
        print(f"\n[UPLOAD] ❌ ERROR INESPERADO")
        print(f"[UPLOAD] {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")

        return {
            "success": False,
            "jobid": None,
            "message": "",
            "error": f"Error inesperado: {str(e)}"
        }


def download_trained_model(
    job_id: str,
    server_url: Optional[str] = None,
    classifier_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Descarga un modelo entrenado desde la nube usando el job_id.

    Args:
        job_id: ID del job retornado por upload-expr
        server_url: URL del servidor (opcional)
        classifier_type: Tipo de clasificador ("P300" o "InnerSpeech") para saber dónde guardar

    Returns:
        Dict con {"success": bool, "model_path": str, "metrics": dict, "error": str}
    """
    import base64
    import pickle
    from pathlib import Path

    if not server_url:
        server_url = os.getenv("EXTERNAL_SERVER_URL", "https://xx8mx485.usw3.devtunnels.ms:8000/")

    endpoint = f"{server_url.rstrip('/')}/job-data"

    print(f"\n{'='*60}")
    print(f"📥 DESCARGANDO MODELO ENTRENADO EN LA NUBE")
    print(f"{'='*60}")
    print(f"[DOWNLOAD] 🔗 Endpoint: {endpoint}")
    print(f"[DOWNLOAD] 🆔 Job ID: {job_id}")

    try:
        # Hacer GET request con job_id como parámetro
        print(f"\n[DOWNLOAD] 📤 Solicitando datos del job...")
        response = requests.get(
            endpoint,
            params={"job_id": job_id},
            timeout=60  # Más tiempo porque el modelo puede ser grande
        )

        if response.status_code != 200:
            error_msg = f"Error del servidor ({response.status_code}): {response.text[:200]}"
            print(f"\n[DOWNLOAD] ❌ ERROR")
            print(f"[DOWNLOAD] {error_msg}")
            print(f"{'='*60}\n")

            return {
                "success": False,
                "model_path": None,
                "metrics": None,
                "error": error_msg
            }

        # Parsear respuesta
        response_data = response.json()
        print(f"[DOWNLOAD] ✅ Datos recibidos")

        # Extraer datos
        job_info = response_data.get("response", {})
        metrics = job_info.get("metrics", {})
        model_pkl_base64 = job_info.get("model_pkl_base64")

        if not model_pkl_base64:
            error_msg = "No se encontró el modelo en la respuesta"
            print(f"\n[DOWNLOAD] ❌ ERROR: {error_msg}")
            print(f"{'='*60}\n")

            return {
                "success": False,
                "model_path": None,
                "metrics": metrics,
                "error": error_msg
            }

        # Decodificar modelo
        print(f"[DOWNLOAD] 🔓 Decodificando modelo base64...")
        model_pkl_bytes = base64.b64decode(model_pkl_base64)

        # Determinar tipo de modelo
        model_type = "inner" if classifier_type == "InnerSpeech" else "p300"

        # Crear directorio para el job_id
        models_dir = Path("backend/models") / job_id / model_type
        models_dir.mkdir(parents=True, exist_ok=True)

        # Guardar modelo .pkl
        model_filename = f"model_cloud_{job_id}.pkl"
        model_path = models_dir / model_filename

        print(f"[DOWNLOAD] 💾 Guardando modelo...")
        with open(model_path, 'wb') as f:
            f.write(model_pkl_bytes)

        # Guardar métricas
        metrics_path = models_dir / "cloud_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Guardar info completa del job
        job_info_path = models_dir / "job_info.json"
        with open(job_info_path, 'w') as f:
            json.dump(response_data, f, indent=2)

        print(f"\n[DOWNLOAD] ✅ MODELO DESCARGADO EXITOSAMENTE!")
        print(f"[DOWNLOAD] 📁 Ruta: {model_path}")
        print(f"[DOWNLOAD] 📊 Métricas guardadas en: {metrics_path}")
        print(f"[DOWNLOAD] 📋 Accuracy: {metrics.get('accuracy', 'N/A')}")
        print(f"[DOWNLOAD] 📋 F1-Score: {metrics.get('f1_score', 'N/A')}")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "model_path": str(model_path),
            "metrics": metrics,
            "job_id": job_id,
            "error": None
        }

    except requests.exceptions.Timeout:
        error_msg = "Timeout: El servidor no respondió en 60 segundos"
        print(f"\n[DOWNLOAD] ⏱️ {error_msg}")
        print(f"{'='*60}\n")

        return {
            "success": False,
            "model_path": None,
            "metrics": None,
            "error": error_msg
        }

    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        print(f"\n[DOWNLOAD] ❌ {error_msg}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")

        return {
            "success": False,
            "model_path": None,
            "metrics": None,
            "error": error_msg
        }


def poll_job_status(job_id: str, server_url: str, interval: int = 5) -> None:
    """
    Hace polling al endpoint /job-data cada X segundos para obtener actualizaciones.

    Args:
        job_id: ID del job
        server_url: URL del servidor
        interval: Intervalo en segundos entre requests (default: 5)
    """
    import time

    endpoint = f"{server_url.rstrip('/')}/job-data"
    print(f"[POLL] 🔄 Iniciando polling cada {interval}s al endpoint /job-data")

    while True:
        try:
            response = requests.get(
                endpoint,
                params={"job_id": job_id},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                progress = data.get("progress")

                if progress:
                    print(f"[POLL] 📊 Status: {status} - Progress: {progress}")
                else:
                    print(f"[POLL] 📊 Status: {status}")

                # Si el job terminó (completed o error), detener polling
                if status in ["completed", "error", "failed"]:
                    print(f"[POLL] ✅ Job finalizado con status: {status}")
                    break
            else:
                print(f"[POLL] ⚠️ Error {response.status_code}: {response.text[:100]}")

        except Exception as e:
            print(f"[POLL] ❌ Error en polling: {e}")

        # Esperar antes del próximo request
        time.sleep(interval)


def connect_to_job_websocket(job_id: str, server_url: str, max_retries: int = 5) -> None:
    """
    Conecta al WebSocket del job para recibir actualizaciones en tiempo real.
    Reintenta la conexión si falla.

    Args:
        job_id: ID del job retornado por /upload-expr
        server_url: URL del servidor (ej: https://server:8000)
        max_retries: Número máximo de reintentos (default: 5)
    """
    if not WEBSOCKET_AVAILABLE:
        print(f"[WS] ⚠️ WebSocket no disponible. Instala: pip install websocket-client")
        return

    # Construir URL del WebSocket
    ws_url = server_url.replace("https://", "ws://").replace("http://", "ws://")
    ws_url = f"{ws_url.rstrip('/')}/ws/job/{job_id}"

    print(f"\n[WS] 🔌 Conectando a WebSocket...")
    print(f"[WS] 📡 URL: {ws_url}")

    retry_count = 0

    def on_message(ws, message):
        """Callback cuando se recibe un mensaje del WebSocket."""
        try:
            data = json.loads(message)
            event_type = data.get("type", "unknown")

            if event_type == "training_started":
                print(f"\n[WS] 🚀 Entrenamiento iniciado")
                print(f"[WS] 📊 Configuración: {data.get('config', {})}")

            elif event_type == "training_progress":
                epoch = data.get("epoch", "?")
                total_epochs = data.get("total_epochs", "?")
                metrics = data.get("metrics", {})
                print(f"[WS] 📈 Progreso: Epoch {epoch}/{total_epochs}")
                if metrics:
                    print(f"[WS] 📊 Métricas: {metrics}")

            elif event_type == "training_completed":
                print(f"\n[WS] ✅ ENTRENAMIENTO COMPLETADO")
                final_metrics = data.get("metrics", {})
                if final_metrics:
                    print(f"[WS] 📊 Métricas finales:")
                    for key, value in final_metrics.items():
                        print(f"[WS]   - {key}: {value}")
                model_path = data.get("model_path")
                if model_path:
                    print(f"[WS] 💾 Modelo guardado en: {model_path}")

            elif event_type == "error":
                error_msg = data.get("message", "Error desconocido")
                print(f"\n[WS] ❌ ERROR EN ENTRENAMIENTO")
                print(f"[WS] {error_msg}")

            else:
                print(f"[WS] 📬 Mensaje: {message}")

        except json.JSONDecodeError:
            print(f"[WS] 📬 {message}")
        except Exception as e:
            print(f"[WS] ⚠️ Error procesando mensaje: {e}")

    def on_error(ws, error):
        """Callback cuando ocurre un error."""
        nonlocal retry_count
        print(f"[WS] ❌ Error de WebSocket: {error}")

        # Reintentar si no se ha alcanzado el máximo
        if retry_count < max_retries:
            retry_count += 1
            import time
            wait_time = min(2 ** retry_count, 30)  # Exponential backoff, max 30s
            print(f"[WS] 🔄 Reintentando en {wait_time}s... (intento {retry_count}/{max_retries})")
            time.sleep(wait_time)
            # Reconectar
            ws.close()
            ws.run_forever()
        else:
            print(f"[WS] ❌ Máximo de reintentos alcanzado ({max_retries})")

    def on_close(ws, close_status_code, close_msg):
        """Callback cuando se cierra la conexión."""
        nonlocal retry_count
        print(f"\n[WS] 🔌 Conexión cerrada")
        if close_status_code:
            print(f"[WS] Código: {close_status_code}, Mensaje: {close_msg}")

        # Reintentar si fue un cierre inesperado y no se alcanzó el máximo
        if close_status_code and close_status_code not in [1000, 1001] and retry_count < max_retries:
            retry_count += 1
            import time
            wait_time = min(2 ** retry_count, 30)
            print(f"[WS] 🔄 Reconectando en {wait_time}s... (intento {retry_count}/{max_retries})")
            time.sleep(wait_time)
            ws.run_forever()

    def on_open(ws):
        """Callback cuando se abre la conexión."""
        print(f"[WS] ✅ Conectado exitosamente")
        print(f"[WS] ⏳ Esperando actualizaciones del servidor...\n")

    try:
        # Crear conexión WebSocket
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Ejecutar en thread separado
        wst = threading.Thread(target=ws.run_forever, daemon=True)
        wst.start()

    except Exception as e:
        print(f"[WS] ❌ Error al conectar: {e}")


def upload_experiment_to_cloud_async(
    classifier_type: str,
    model_name: str,
    dataset_name: Optional[str] = None
) -> None:
    """
    Envía el experimento al servidor en un thread separado y conecta al WebSocket para seguimiento.

    Esta función se ejecuta en segundo plano y no bloquea la UI.
    La simulación continúa mientras el upload real ocurre en paralelo.

    Args:
        classifier_type: "P300" o "InnerSpeech"
        model_name: Nombre del modelo
        dataset_name: Nombre del dataset (opcional)
    """
    def _upload_thread():
        """Thread worker que ejecuta el upload y conecta al WebSocket."""
        result = upload_experiment_to_cloud_sync(classifier_type, model_name, dataset_name)

        if result["success"]:
            print(f"\n🎉 EXPERIMENTO ENVIADO A LA NUBE")
            print(f"📋 Job ID: {result['jobid']}")
            print(f"💡 El entrenamiento se está ejecutando en el servidor\n")

            # Conectar al WebSocket para seguimiento en tiempo real
            server_url = os.getenv("EXTERNAL_SERVER_URL", "https://xx8mx485.usw3.devtunnels.ms:8000/")

            # Extraer job_id limpio (puede venir como JSON)
            job_id = result['jobid']
            if isinstance(job_id, str):
                # Si viene como JSON string, extraer el job_id
                try:
                    job_data = json.loads(job_id)
                    job_id = job_data.get('job_id', job_id)
                except:
                    pass  # Ya es un string limpio

            # Iniciar WebSocket con retry
            connect_to_job_websocket(job_id, server_url, max_retries=5)

            # Iniciar polling en paralelo (backup si WebSocket falla)
            polling_thread = threading.Thread(
                target=poll_job_status,
                args=(job_id, server_url, 5),
                daemon=True
            )
            polling_thread.start()

        else:
            print(f"\n⚠️ NO SE PUDO ENVIAR EL EXPERIMENTO")
            print(f"❌ Error: {result['error']}\n")

    # Ejecutar en thread separado
    thread = threading.Thread(target=_upload_thread, daemon=True)
    thread.start()
    print(f"[UPLOAD] 🔄 Iniciando envío en segundo plano...")
