"""
Utilidades para Simulación de Modelos P300 + Inner Speech
===========================================================

Funciones para cargar modelos entrenados, aplicar pipelines y realizar predicciones
en el contexto de simulación en tiempo real.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os


def list_available_models(model_type: str) -> List[Dict[str, Any]]:
    """
    Escanea directorio de modelos y retorna lista de modelos disponibles.

    Args:
        model_type: Tipo de modelo ("p300" o "inner")

    Returns:
        Lista de diccionarios con metadata de cada modelo:
        [
            {
                "experiment_id": "39",
                "model_name": "CNN",
                "model_type": "p300",
                "timestamp": "20251116_230930",
                "snapshot_path": str,
                "pkl_path": str,
                "metrics": {...},
                "window_size_samples": int,
                "pipeline_summary": str
            },
            ...
        ]
    """
    models_dir = Path(__file__).parent.parent / "models"

    if not models_dir.exists():
        print(f"[list_available_models] Directorio de modelos no encontrado: {models_dir}")
        return []

    available_models = []

    # Iterar sobre experimentos
    for exp_dir in sorted(models_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        model_type_dir = exp_dir / model_type
        if not model_type_dir.exists():
            continue

        # Buscar snapshot
        snapshot_path = model_type_dir / "experiment_snapshot.json"
        if not snapshot_path.exists():
            continue

        try:
            # Cargar snapshot
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                snapshot = json.load(f)

            # Extraer info del modelo
            classifier_config = snapshot.get("classifier_config", {})
            model_name = classifier_config.get("model_name", "Unknown")

            # Buscar archivo .pkl
            pkl_files = list(model_type_dir.glob(f"{model_name.lower()}_*.pkl"))
            if not pkl_files:
                print(f"[list_available_models] No se encontró .pkl para {exp_dir.name}/{model_type}/{model_name}")
                continue

            pkl_path = pkl_files[0]  # Tomar el primero

            # Cargar training_info para métricas
            training_info_path = model_type_dir / "training_info.json"
            metrics = {}
            timestamp = ""

            if training_info_path.exists():
                with open(training_info_path, 'r', encoding='utf-8') as f:
                    training_info = json.load(f)
                    if training_info and isinstance(training_info, list) and len(training_info) > 0:
                        latest_training = training_info[0]
                        metrics = latest_training.get("metrics", {})
                        timestamp = latest_training.get("timestamp", "")

            # Inferir window size
            window_size = infer_window_size_from_snapshot(snapshot)

            # Resumen del pipeline
            pipeline_summary = _get_pipeline_summary(snapshot)

            available_models.append({
                "experiment_id": exp_dir.name,
                "model_name": model_name,
                "model_type": model_type,
                "timestamp": timestamp,
                "snapshot_path": str(snapshot_path),
                "pkl_path": str(pkl_path),
                "metrics": metrics,
                "window_size_samples": window_size,
                "pipeline_summary": pipeline_summary
            })

        except Exception as e:
            print(f"[list_available_models] Error procesando {exp_dir.name}/{model_type}: {e}")
            continue

    print(f"[list_available_models] Encontrados {len(available_models)} modelos de tipo '{model_type}'")
    return available_models


def load_model_for_inference(snapshot_path: str, pkl_path: str) -> Dict[str, Any]:
    """
    Carga modelo y su configuración completa para inferencia.

    Args:
        snapshot_path: Ruta al experiment_snapshot.json
        pkl_path: Ruta al archivo .pkl del modelo

    Returns:
        {
            "model_instance": objeto modelo cargado,
            "model_name": str,
            "window_size_samples": int,
            "pipeline_config": {...},
            "model_metadata": {...},
            "frame_context": int or None (solo para CNN)
        }
    """
    # Cargar snapshot
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    # Cargar modelo desde pickle
    with open(pkl_path, 'rb') as f:
        model_instance = pickle.load(f)

    # Extraer info
    classifier_config = snapshot.get("classifier_config", {})
    model_name = classifier_config.get("model_name", "Unknown")
    window_size = infer_window_size_from_snapshot(snapshot)

    # Extraer frame_context si es CNN
    frame_context = None
    if model_name == "CNN":
        config = classifier_config.get("config", {})
        frame_context = config.get("frame_context", 8)

    # Pipeline config
    pipeline_config = {
        "filters": snapshot.get("filters", []),
        "transform": snapshot.get("transform", []),
        "dataset": snapshot.get("dataset", "")
    }

    # Metadata
    model_metadata = {
        "model_type": snapshot_path.split('/')[-2],  # "p300" o "inner"
        "experiment_id": snapshot.get("id", "unknown"),
        "classes": snapshot.get("subset_info", {}).get("classes", [])
    }

    return {
        "model_instance": model_instance,
        "model_name": model_name,
        "window_size_samples": window_size,
        "pipeline_config": pipeline_config,
        "model_metadata": model_metadata,
        "frame_context": frame_context
    }


def infer_window_size_from_snapshot(snapshot: Dict) -> int:
    """
    Extrae el tamaño de ventana esperado desde el experiment_snapshot.

    Args:
        snapshot: Diccionario del experiment_snapshot.json

    Returns:
        Número de samples esperados
    """
    transforms = snapshot.get("transform", [])

    if not transforms:
        print("[infer_window_size_from_snapshot] WARNING: No hay transforms en snapshot")
        return 2612  # Default fallback

    # Tomar el PRIMER transform (representa la entrada original)
    first_transform = transforms[0]

    # Buscar dimensionality_change
    dim_change = first_transform.get("dimensionality_change", {})
    input_shape = dim_change.get("input_shape", [])
    transposed = dim_change.get("transposed_from_input", False)

    if len(input_shape) >= 2:
        # Determinar índice según si fue transpuesto:
        # - transposed=False: input_shape es [n_channels, n_times] → leer index 1
        # - transposed=True:  input_shape es [n_times, n_channels] → leer index 0
        if transposed:
            # Después de ICA: [n_samples, n_channels]
            window_size = input_shape[0]
        else:
            # Sin ICA: [n_channels, n_samples]
            window_size = input_shape[1]

        print(f"[infer_window_size_from_snapshot] Window size detectado: {window_size} samples (transposed={transposed})")
        return window_size
    else:
        print(f"[infer_window_size_from_snapshot] WARNING: input_shape inválido: {input_shape}")
        return 2612  # Default fallback


def apply_pipeline_from_snapshot(
    raw_window: np.ndarray,
    snapshot_pipeline: Dict,
    model_type: str = "p300",
    experiment_id: Optional[str] = None
) -> np.ndarray:
    """
    Aplica el pipeline (filtros + transformación) a una ventana raw
    usando DIRECTAMENTE la configuración del snapshot (no del experimento actual).

    Args:
        raw_window: Array numpy (n_channels, n_samples)
        snapshot_pipeline: Dict con keys "filters" y "transform"
        model_type: "p300" o "inner"
        experiment_id: ID del experimento (necesario para cargar filtros ICA)

    Returns:
        Array numpy transformado según el pipeline
    """
    # Aplicar filtros (si los hay)
    filtered_data = raw_window
    filters = snapshot_pipeline.get("filters", [])

    for filter_config in filters:
        if "ICA" in filter_config:
            # Aplicar ICA desde archivo .fif guardado
            filtered_data = _apply_ica_filter(
                filtered_data,
                filter_config["ICA"],
                experiment_id,
                model_type
            )
        elif "BandPass" in filter_config:
            # Los filtros BandPass se aplican offline, no en simulación
            # (la señal raw ya viene del LSL que puede tener filtros aplicados)
            pass
        else:
            # Otros filtros: por ahora ignorar
            pass

    # Aplicar transformación
    transforms = snapshot_pipeline.get("transform", [])

    if not transforms:
        # Sin transformación, devolver raw
        return filtered_data

    # Tomar la última transformación (la que se usa para entrenamiento)
    last_transform = transforms[-1]

    # Detectar tipo de transformación
    if "WaveletTransform" in last_transform:
        transformed = _apply_wavelet_transform(filtered_data, last_transform["WaveletTransform"])
    elif "FFTTransform" in last_transform:
        transformed = _apply_fft_transform(filtered_data, last_transform["FFTTransform"])
    elif "DCTTransform" in last_transform:
        transformed = _apply_dct_transform(filtered_data, last_transform["DCTTransform"])
    elif "WindowingTransform" in last_transform:
        transformed = _apply_windowing_transform(filtered_data, last_transform["WindowingTransform"])
    else:
        # Transformación desconocida, devolver sin transformar
        print(f"[apply_pipeline_from_snapshot] WARNING: Transformación desconocida: {list(last_transform.keys())}")
        transformed = filtered_data

    return transformed


def _apply_wavelet_transform(raw_data: np.ndarray, config: Dict) -> np.ndarray:
    """
    Aplica WaveletTransform usando la configuración del snapshot.
    Extrae la lógica core de WaveletTransform.apply() para trabajar con numpy arrays.

    Args:
        raw_data: (n_channels, n_samples)
        config: Configuración de WaveletTransform del snapshot

    Returns:
        Datos transformados (n_frames, frame_length, n_channels)
    """
    import pywt

    X_raw = raw_data.astype(np.float64)
    n_channels, n_times = X_raw.shape

    # ===== Configuración de wavelet =====
    try:
        wave = pywt.Wavelet(config['wavelet'])
    except Exception as e:
        raise ValueError(f"Wavelet no válida: {config['wavelet']}") from e

    # Modo de borde
    valid_modes = set(m.lower() for m in pywt.Modes.modes)
    mode_in = (config.get('mode') or "symmetric").lower()
    aliases = {
        "periodic": "periodization", "per": "periodization", "ppd": "periodization",
        "reflect": "symmetric", "sym": "symmetric",
        "const": "constant", "zpd": "zero", "sp1": "smooth"
    }
    mode = aliases.get(mode_in, mode_in)
    if mode not in valid_modes:
        mode = "symmetric"  # fallback seguro

    # Nivel de descomposición
    max_level = pywt.dwt_max_level(data_len=n_times, filter_len=wave.dec_len)
    if config.get('level') is None:
        L = max(1, min(10, max_level))
    else:
        L = int(config['level'])
        if L < 1 or L > min(10, max_level):
            L = max(1, min(L, min(10, max_level)))

    # Threshold para denoising
    thr = float(config['threshold']) if config.get('threshold') is not None else None
    do_denoise = thr is not None and thr >= 0.0

    # ===== Reconstrucción Wavelet por canal =====
    Y_std = np.empty_like(X_raw, dtype=np.float64)
    for ch in range(n_channels):
        sig = X_raw[ch].astype(np.float64, copy=False)
        coeffs = pywt.wavedec(sig, wavelet=wave, mode=mode, level=L)

        if do_denoise:
            coeffs_d = [coeffs[0]]
            for cd in coeffs[1:]:
                coeffs_d.append(pywt.threshold(cd, value=thr, mode="soft"))
            rec = pywt.waverec(coeffs_d, wavelet=wave, mode=mode)
        else:
            rec = pywt.waverec(coeffs, wavelet=wave, mode=mode)

        # Ajuste de longitud
        if rec.shape[0] != n_times:
            if rec.shape[0] > n_times:
                rec = rec[:n_times]
            else:
                rec = np.pad(rec, (0, n_times - rec.shape[0]), mode="edge")

        Y_std[ch] = rec

    # Transponer a (n_times, n_channels)
    Y_continuous = Y_std.T.astype(np.float32, copy=False)

    # ===== Parámetros de ventaneo =====
    frame_length = int(config['frame_length'])

    if config.get('hop_samples') is not None:
        hop = int(config['hop_samples'])
    else:
        ov = float(config.get('overlap', 0.0))
        if ov == 1.0:
            ov = 0.99
        hop = max(1, int(round(frame_length * (1.0 - ov))))

    # Padding si n_times < frame_length
    if n_times < frame_length:
        pad_width = frame_length - n_times
        Y_padded = np.pad(Y_continuous, ((0, pad_width), (0, 0)), mode='edge')
        n_times_eff = frame_length
    else:
        Y_padded = Y_continuous
        n_times_eff = n_times

    # Calcular número de frames
    n_frames = 1 + (n_times_eff - frame_length) // hop
    if n_frames <= 0:
        n_frames = 1

    # ===== Ventaneo: (n_times, n_channels) → (n_frames, frame_length, n_channels) =====
    Y_windowed = np.empty((n_frames, frame_length, n_channels), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        end = start + frame_length
        if end <= Y_padded.shape[0]:
            Y_windowed[i] = Y_padded[start:end]
        else:
            # Último frame con padding
            remaining = Y_padded.shape[0] - start
            Y_windowed[i, :remaining] = Y_padded[start:]
            Y_windowed[i, remaining:] = 0.0

    return Y_windowed


def _apply_fft_transform(raw_data: np.ndarray, config: Dict) -> np.ndarray:
    """
    Aplica FFTTransform usando la configuración del snapshot.
    Extrae la lógica core de FFTTransform.apply() para trabajar con numpy arrays.

    Args:
        raw_data: (n_channels, n_samples)
        config: Configuración de FFTTransform del snapshot

    Returns:
        Datos transformados (n_frames, n_freqs, n_channels)
    """
    X_raw = raw_data.astype(np.float64)
    n_channels, n_times = X_raw.shape

    # ===== Parámetros de ventaneo =====
    # frame_length
    if config.get('frame_length') is not None:
        frame_length = int(config['frame_length'])
    else:
        if n_times >= 256:
            fl = 256
            while (fl << 1) <= n_times:
                fl <<= 1
            frame_length = max(256, fl)
        else:
            frame_length = 1 << int(np.floor(np.log2(max(16, n_times))))
            frame_length = max(16, int(frame_length))

    # hop
    if config.get('hop_samples') is not None:
        hop = int(config['hop_samples'])
    else:
        ov = float(config.get('overlap', 0.0))
        if ov == 1.0:
            ov = 0.99
        hop = max(1, int(round(frame_length * (1.0 - ov))))

    # nfft
    if config.get('nfft') is not None:
        nfft = int(config['nfft'])
    else:
        nfft = frame_length
    if nfft < frame_length:
        nfft = 1 << int(np.ceil(np.log2(frame_length)))

    # ===== Ventana =====
    wname = str(config.get('window', 'hann')).lower()
    if wname == "hann":
        win = np.hanning(frame_length)
    elif wname == "hamming":
        win = np.hamming(frame_length)
    elif wname == "blackman":
        win = np.blackman(frame_length)
    elif wname == "rectangular":
        win = np.ones(frame_length, dtype=float)
    else:
        win = np.hanning(frame_length)  # fallback
    win = win.astype(np.float64)

    # ===== Padding si es necesario =====
    if n_times < frame_length:
        pad_width = frame_length - n_times
        X_pad = np.pad(X_raw, ((0, 0), (0, pad_width)), mode="constant")
        n_times_eff = X_pad.shape[1]
    else:
        X_pad = X_raw
        n_times_eff = n_times

    # ===== Número de frames =====
    n_frames = 1 + (n_times_eff - frame_length) // hop
    if n_frames <= 0:
        n_frames = 1

    # Verificar padding del último frame
    last_frame_end = (n_frames - 1) * hop + frame_length
    if last_frame_end > n_times_eff:
        pad_needed = last_frame_end - n_times_eff
        X_pad = np.pad(X_pad, ((0, 0), (0, pad_needed)), mode="constant")
        n_times_eff = X_pad.shape[1]

    # ===== Calcular FFT =====
    freqs = np.fft.rfftfreq(nfft, d=1.0 / config.get('sp', 1024.0))
    n_freqs = int(freqs.shape[0])
    power = np.empty((n_frames, n_freqs, n_channels), dtype=np.float32)

    # Normalización
    win_norm = np.sqrt((win ** 2).sum())

    # FFT por canal
    for ch in range(n_channels):
        sig = X_pad[ch]
        frames = np.lib.stride_tricks.as_strided(
            sig,
            shape=(n_frames, frame_length),
            strides=(sig.strides[0] * hop, sig.strides[0]),
            writeable=False,
        ).copy()
        frames *= win
        spec = np.fft.rfft(frames, n=nfft, axis=1)
        pxx = (np.abs(spec) ** 2) / (win_norm ** 2 + 1e-12)
        power[:, :, ch] = pxx.astype(np.float32, copy=False)

    return power


def _apply_dct_transform(raw_data: np.ndarray, config: Dict) -> np.ndarray:
    """
    Aplica DCTTransform usando la configuración del snapshot.
    Extrae la lógica core de DCTTransform.apply() para trabajar con numpy arrays.

    Args:
        raw_data: (n_channels, n_samples)
        config: Configuración de DCTTransform del snapshot

    Returns:
        Datos transformados (n_frames, n_coeffs, n_channels)
    """
    from scipy.fft import dct as sp_dct

    X_raw = raw_data.astype(np.float64)
    n_channels, n_times = X_raw.shape

    # ===== Parámetros de ventaneo =====
    if config.get('frame_length') is not None:
        frame_length = int(config['frame_length'])
    else:
        if n_times >= 256:
            target = 1 << int(np.ceil(np.log2(256)))
            while target * 2 <= n_times:
                target *= 2
            frame_length = target
        else:
            target = 1 << int(np.floor(np.log2(max(16, n_times))))
            frame_length = max(16, int(target))

    # hop
    ov = float(config.get('overlap', 0.0))
    if not (0.0 <= ov < 1.0):
        ov = 0.0
    hop = max(1, int(round(frame_length * (1.0 - ov))))

    # ===== Ventana =====
    wname = str(config.get('window', 'rectangular')).lower()
    if wname == "hann":
        win = np.hanning(frame_length)
    elif wname == "hamming":
        win = np.hamming(frame_length)
    elif wname == "blackman":
        win = np.blackman(frame_length)
    elif wname == "rectangular":
        win = np.ones(frame_length, dtype=float)
    else:
        win = np.ones(frame_length, dtype=float)  # fallback
    win = win.astype(np.float64)

    # ===== Padding si es necesario =====
    if n_times < frame_length:
        pad_width = frame_length - n_times
        X_pad = np.pad(X_raw, ((0, 0), (0, pad_width)), mode="constant")
        n_times_eff = X_pad.shape[1]
    else:
        X_pad = X_raw
        n_times_eff = n_times

    # ===== Número de frames =====
    n_frames = 1 + (n_times_eff - frame_length) // hop
    if n_frames <= 0:
        n_frames = 1

    # Verificar padding del último frame
    last_frame_end = (n_frames - 1) * hop + frame_length
    if last_frame_end > n_times_eff:
        pad_needed = last_frame_end - n_times_eff
        X_pad = np.pad(X_pad, ((0, 0), (0, pad_needed)), mode="constant")
        n_times_eff = X_pad.shape[1]

    # ===== DCT =====
    dct_type = int(config.get('type', 2))
    if dct_type not in (1, 2, 3, 4):
        dct_type = 2
    norm = config.get('norm', None)

    n_coeffs = frame_length
    coeffs_cube = np.empty((n_frames, n_coeffs, n_channels), dtype=np.float32)

    for ch in range(n_channels):
        sig = X_pad[ch]
        frames = np.lib.stride_tricks.as_strided(
            sig,
            shape=(n_frames, frame_length),
            strides=(sig.strides[0] * hop, sig.strides[0]),
            writeable=False,
        ).copy()
        frames = (frames * win).astype(np.float64, copy=False)
        dct_frames = sp_dct(frames, type=dct_type, norm=norm, axis=1)
        coeffs_cube[:, :, ch] = dct_frames.astype(np.float32, copy=False)

    return coeffs_cube


def _apply_windowing_transform(raw_data: np.ndarray, config: Dict) -> np.ndarray:
    """
    Aplica WindowingTransform usando la configuración del snapshot.
    Extrae la lógica core de WindowingTransform.apply() para trabajar con numpy arrays.

    Args:
        raw_data: (n_channels, n_samples)
        config: Configuración de WindowingTransform del snapshot

    Returns:
        Datos transformados (n_frames, window_size, n_channels)
    """
    X_raw = raw_data.astype(np.float32)
    n_channels, n_times = X_raw.shape

    # ===== Parámetros =====
    window_size = int(config.get('window_size', 64))
    padding_mode = str(config.get('padding_mode', 'constant'))

    # hop
    if config.get('overlap') is not None:
        ov = float(config['overlap'])
        if not (0.0 <= ov < 1.0):
            ov = 0.0
        hop = max(1, int(round(window_size * (1.0 - ov))))
    else:
        hop = window_size  # sin overlap

    # ===== Padding si es necesario =====
    if n_times < window_size:
        pad_width = window_size - n_times
        X_pad = np.pad(X_raw, ((0, 0), (0, pad_width)), mode=padding_mode)
        n_times_eff = window_size
    else:
        X_pad = X_raw
        n_times_eff = n_times

    # ===== Número de frames =====
    n_frames = 1 + (n_times_eff - window_size) // hop
    if n_frames <= 0:
        n_frames = 1

    # ===== Ventaneo con stride_tricks =====
    X_windowed_list = []
    for ch in range(n_channels):
        sig = X_pad[ch]
        frames = np.lib.stride_tricks.as_strided(
            sig,
            shape=(n_frames, window_size),
            strides=(sig.strides[0] * hop, sig.strides[0]),
            writeable=False,
        ).copy()
        X_windowed_list.append(frames)

    # Stack: (n_channels, n_frames, window_size) → (n_frames, window_size, n_channels)
    X_windowed = np.stack(X_windowed_list, axis=-1).astype(np.float32)

    return X_windowed


def predict_with_model(
    model_instance: Any,
    model_name: str,
    transformed_data: np.ndarray,
    frame_context: Optional[int] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Realiza predicción con el modelo según su tipo.

    Args:
        model_instance: Instancia del modelo cargado
        model_name: Nombre del modelo ("CNN", "LSTM", "SVM", etc.)
        transformed_data: Datos transformados (frames)
        frame_context: Contexto de frames para CNN (ej: 4)
        class_names: Lista de nombres de clases (ej: ["rest", "arriba", "abajo", ...])

    Returns:
        {
            "prediction": int o str (clase predicha),
            "confidence": float (confianza de la predicción),
            "probabilities": list (probabilidades por clase),
            "predicted_class_name": str (nombre de la clase predicha, si class_names dado)
        }
    """
    # Preparar datos según tipo de modelo
    if model_name == "CNN":
        # CNN necesita imágenes construidas con frame_context
        from backend.classes.ClasificationModel.CNN import CNN

        if frame_context is None:
            frame_context = getattr(model_instance, 'frame_context', 8)

        # transformed_data esperado: (n_frames, freq, n_channels)
        # Necesitamos construir imágenes con contexto
        X_images = _build_cnn_images_from_frames(transformed_data, frame_context, model_instance.image_hw)

        # Predecir - CNN usa _tf_model
        tf_model = getattr(model_instance, '_tf_model', None)
        if tf_model is None:
            raise AttributeError(f"Modelo CNN no tiene atributo _tf_model")
        predictions = tf_model.predict(X_images, verbose=0)

    elif model_name in ["LSTM", "GRU"]:
        # LSTM/GRU esperan secuencias (n_frames, features_per_frame)
        # Si transformed_data es 3D (n_frames, frame_length, n_channels), aplanar a 2D
        if len(transformed_data.shape) == 3:
            n_frames = transformed_data.shape[0]
            # Aplanar cada frame: (n_frames, frame_length, n_channels) -> (n_frames, frame_length * n_channels)
            transformed_data_flat = transformed_data.reshape(n_frames, -1)
        else:
            transformed_data_flat = transformed_data

        # LSTM/GRU usan _tf_model
        tf_model = getattr(model_instance, '_tf_model', None)
        if tf_model is None:
            raise AttributeError(f"Modelo {model_name} no tiene atributo _tf_model")

        # Expandir batch dimension: (n_frames, features) -> (1, n_frames, features)
        predictions = tf_model.predict(np.expand_dims(transformed_data_flat, 0), verbose=0)

    elif model_name == "SVM":
        # SVM: aplanar frames
        n_frames = transformed_data.shape[0]
        X_flat = transformed_data.reshape(n_frames, -1)

        # SVM usa _svc_model
        svc_model = getattr(model_instance, '_svc_model', None)
        if svc_model is None:
            raise AttributeError("Modelo SVM no tiene atributo _svc_model")

        # Predecir
        predictions = svc_model.predict(X_flat)
        proba = svc_model.predict_proba(X_flat) if hasattr(svc_model, 'predict_proba') else None

    elif model_name == "RandomForest":
        # RandomForest: aplanar frames
        n_frames = transformed_data.shape[0]
        X_flat = transformed_data.reshape(n_frames, -1)

        # RandomForest usa _model
        rf_model = getattr(model_instance, '_model', None)
        if rf_model is None:
            raise AttributeError("Modelo RandomForest no tiene atributo _model")

        # Debug: imprimir clases del modelo (SILENCIADO para tiempo real)
        if hasattr(rf_model, 'classes_'):
            model_classes = rf_model.classes_
            # print(f"[predict_with_model DEBUG] RandomForest classes_: {model_classes}")
        else:
            model_classes = None

        # Predecir
        predictions = rf_model.predict(X_flat)
        proba = rf_model.predict_proba(X_flat) if hasattr(rf_model, 'predict_proba') else None

        # Debug silenciado
        # unique_preds = np.unique(predictions)
        # print(f"[predict_with_model DEBUG] Predictions unique values: {unique_preds}")

    elif model_name == "SVNN":
        # SVNN usa _keras_model
        keras_model = getattr(model_instance, '_keras_model', None)
        if keras_model is None:
            raise AttributeError("Modelo SVNN no tiene atributo _keras_model")
        predictions = keras_model.predict(transformed_data, verbose=0)

    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_name}")

    # Procesar salida
    # Para modelos ML tradicionales (SVM, RandomForest) que devuelven predicciones directas
    if model_name in ["SVM", "RandomForest"]:
        # predictions es un array 1D de clases predichas por frame
        # Tomar la predicción más frecuente (mayoría)
        from collections import Counter
        counts = Counter(predictions)
        prediction_value = int(counts.most_common(1)[0][0])

        # Calcular confianza y probabilidades si hay proba disponible
        if 'proba' in locals() and proba is not None:
            # Promediar probabilidades de todos los frames
            probabilities = np.mean(proba, axis=0)

            # Para RandomForest/SVM, las predicciones son los valores de clase, no índices
            # Necesitamos mapear el valor predicho al índice en classes_
            model_obj = rf_model if model_name == "RandomForest" else svc_model

            if hasattr(model_obj, 'classes_'):
                classes = model_obj.classes_

                # Buscar el índice del valor predicho en classes_
                if prediction_value in classes:
                    prediction = int(np.where(classes == prediction_value)[0][0])
                else:
                    print(f"[predict_with_model] WARNING: {model_name} predijo clase {prediction_value} que no está en classes_ {classes}. Usando última clase.")
                    prediction = len(probabilities) - 1
            else:
                # Fallback: asumir que prediction_value es el índice
                prediction = prediction_value
                if prediction >= len(probabilities):
                    print(f"[predict_with_model] WARNING: {model_name} predijo índice {prediction} pero proba tiene {len(probabilities)} clases. Usando última clase.")
                    prediction = len(probabilities) - 1

            confidence = float(probabilities[prediction])
        else:
            # Sin probabilidades, usar confianza basada en frecuencia
            prediction = prediction_value
            confidence = counts[prediction] / sum(counts.values())
            # Crear array de probabilidades sintético
            n_classes = max(counts.keys()) + 1
            probabilities = [confidence if i == prediction else (1-confidence)/(n_classes-1 if n_classes > 1 else 1)
                           for i in range(n_classes)]

    # Para modelos deep learning (CNN, LSTM, GRU, SVNN) que devuelven probabilidades
    elif len(predictions.shape) > 1 and predictions.shape[0] > 0:
        # Probabilidades
        probabilities = predictions[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
    else:
        # Predicción directa (fallback)
        prediction = int(predictions[0]) if hasattr(predictions[0], '__int__') else predictions[0]
        confidence = 1.0
        probabilities = [1.0]

    # Agregar nombre de clase si está disponible
    result = {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities
    }

    if class_names and prediction < len(class_names):
        result["predicted_class_name"] = class_names[prediction]
        # print(f"[predict_with_model] Predicción final: índice {prediction} -> clase '{class_names[prediction]}' (confianza: {confidence:.2%})")  # DEBUG: silenciado
    else:
        pass
        # print(f"[predict_with_model] Predicción final: índice {prediction} (confianza: {confidence:.2%})")  # DEBUG: silenciado

    return result


def extract_window_with_padding(
    raw_signal: np.ndarray,
    start: int,
    window_size: int
) -> np.ndarray:
    """
    Extrae ventana del raw signal con padding de ceros si es necesario.

    Args:
        raw_signal: Señal completa (n_channels, n_samples)
        start: Índice de inicio
        window_size: Tamaño de ventana en samples

    Returns:
        Ventana extraída (n_channels, window_size) con padding si necesario
    """
    end = start + window_size
    signal_length = raw_signal.shape[1]

    # Caso 1: Ventana completa dentro del rango
    if end <= signal_length:
        return raw_signal[:, start:end]

    # Caso 2: Necesita padding al final
    available = raw_signal[:, start:signal_length]
    padding_needed = window_size - available.shape[1]

    return np.pad(
        available,
        ((0, 0), (0, padding_needed)),
        mode='constant',
        constant_values=0
    )


# ============================================================================
# Funciones Helper Privadas
# ============================================================================

def _apply_ica_filter(
    raw_data: np.ndarray,
    ica_config: Dict,
    experiment_id: Optional[str],
    model_type: str
) -> np.ndarray:
    """
    Aplica filtro ICA usando el objeto ICA guardado durante entrenamiento.

    Args:
        raw_data: Datos raw (n_channels, n_samples)
        ica_config: Configuración del filtro ICA del snapshot
        experiment_id: ID del experimento
        model_type: "p300" o "inner"

    Returns:
        Datos transformados por ICA (n_components, n_samples)
    """
    import mne
    from pathlib import Path

    if experiment_id is None:
        print("[_apply_ica_filter] ERROR: experiment_id requerido para cargar ICA")
        return raw_data

    # Construir ruta al archivo .fif del ICA
    # Estructura: backend/models/{experiment_id}/{model_type}/Aux/.../file_ica_{filter_id}.fif
    # Necesitamos encontrar el archivo .fif correspondiente

    # Obtener directorio base de modelos
    models_dir = Path(__file__).parent.parent / "models"
    exp_dir = models_dir / experiment_id / model_type

    if not exp_dir.exists():
        print(f"[_apply_ica_filter] ERROR: Directorio del experimento no encontrado: {exp_dir}")
        return raw_data

    # Buscar archivo .fif del ICA
    # Patrón: *_ica_{filter_id}.fif
    filter_id = ica_config.get("id", "")
    ica_files = list(exp_dir.rglob(f"*_ica_{filter_id}.fif"))

    if not ica_files:
        print(f"[_apply_ica_filter] ERROR: No se encontró archivo ICA .fif para filter_id={filter_id} en {exp_dir}")
        print(f"[_apply_ica_filter] Buscando patrón: *_ica_{filter_id}.fif")
        return raw_data

    ica_path = ica_files[0]
    print(f"[_apply_ica_filter] Cargando ICA desde: {ica_path}")

    try:
        # Cargar ICA fitted object
        ica = mne.preprocessing.read_ica(str(ica_path), verbose=False)

        # Preparar datos para MNE
        n_channels, n_samples = raw_data.shape
        sfreq = float(ica_config.get("sp", 1024.0))

        # Crear info y RawArray
        ch_names = [f"EEG{idx+1:03d}" for idx in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(raw_data, info, verbose=False)

        # Aplicar ICA (extraer fuentes independientes)
        sources = ica.get_sources(raw).get_data()  # (n_components, n_samples)

        print(f"[_apply_ica_filter] ICA aplicado: {n_channels} canales → {sources.shape[0]} componentes")
        return sources

    except Exception as e:
        print(f"[_apply_ica_filter] ERROR aplicando ICA: {e}")
        import traceback
        traceback.print_exc()
        return raw_data


def _get_pipeline_summary(snapshot: Dict) -> str:
    """Genera resumen legible del pipeline"""
    transforms = snapshot.get("transform", [])

    if not transforms:
        return "Sin transformación"

    last_transform = transforms[-1]

    # Buscar tipo de transformación
    for key in last_transform.keys():
        if key not in ["id", "dimensionality_change"]:
            transform_config = last_transform[key]
            if isinstance(transform_config, dict):
                # Extraer parámetros clave
                if "wavelet" in transform_config:
                    wavelet = transform_config.get("wavelet", "?")
                    level = transform_config.get("level", "?")
                    return f"WaveletTransform({wavelet}, level={level})"
                elif "n_fft" in transform_config:
                    n_fft = transform_config.get("n_fft", "?")
                    return f"FFTTransform(n_fft={n_fft})"
                elif "n_dct" in transform_config:
                    n_dct = transform_config.get("n_dct", "?")
                    return f"DCTTransform(n_dct={n_dct})"
                else:
                    return key

    return "Transform desconocido"


def _build_cnn_images_from_frames(
    frames: np.ndarray,
    frame_context: int,
    target_hw: Tuple[int, int]
) -> np.ndarray:
    """
    Construye imágenes para CNN desde frames transformados.

    Args:
        frames: (n_frames, freq, n_channels) o (n_frames, time, freq)
        frame_context: Contexto (ej: 4 → imagen de ancho 2*4+1=9)
        target_hw: (height, width) destino de la imagen

    Returns:
        X_images: (n_valid_frames, H, W, 3) listo para CNN
    """
    from backend.classes.ClasificationModel.CNN import CNN

    # Detectar formato
    if len(frames.shape) == 3:
        F, dim1, dim2 = frames.shape

        # Determinar si es (F, freq, channels) o (F, time, freq)
        # Heurística: freq típicamente < 200, channels típicamente > 100
        if dim2 > dim1:
            # Probablemente (F, freq, channels)
            cube_fwc = frames  # (F, freq, C)
        else:
            # Probablemente (F, time, freq)
            cube_fwc = np.transpose(frames, (0, 2, 1))  # (F, freq, time) → (F, freq, C) asumiendo time como C
    else:
        raise ValueError(f"Shape de frames no reconocido: {frames.shape}")

    # Generar labels dummy (todos de misma clase para predicción)
    frame_labels = np.zeros(cube_fwc.shape[0], dtype=int)

    # Usar método de CNN para construir imágenes
    X_images, _ = CNN._images_from_spec_per_frame(
        cube_fwc=cube_fwc,
        frame_labels=frame_labels,
        k_ctx=frame_context,
        target_hw=target_hw
    )

    return X_images
