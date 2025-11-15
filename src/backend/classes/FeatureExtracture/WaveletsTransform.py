from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional
from pydantic import Field
from pathlib import Path
import numpy as np
import pywt

from backend.helpers.numeric_array import _load_numeric_array
from backend.classes.Experiment import Experiment
from collections import Counter


class WaveletTransform(Transform):
    """
    Descomposición Wavelet discreta por canal con opción de denoising (soft-threshold en detalles).
    Ventanea la señal reconstruida según parámetros de frame_length y hop/overlap.

    Artefactos:
      <stem>_wavelet_<id>.npy          # señal reconstruida ventaneada (n_frames, frame_length, n_channels)
      <stem>_wavelet_<id>_labels.npy   # etiquetas por frame (n_frames,)
    """
    wavelet: str = Field(
        ...,
        description="Nombre de la wavelet a usar (ej. db4, coif5, sym8, etc.)"
    )
    level: Optional[int] = Field(
        None, ge=1, le=10,
        description="Nivel de descomposición (opcional); si None, usa el máximo permitido"
    )
    mode: Optional[str] = Field(
        "symmetric",
        description="Modo de extensión de bordes: symmetric, periodization, reflect, etc."
    )
    threshold: Optional[float] = Field(
        None, ge=0.0,
        description="Umbral (>=0) para denoising por soft-thresholding de detalles"
    )

    # === NUEVOS PARÁMETROS DE VENTANEO (para etiquetas por frame) ===
    frame_length: Optional[int] = Field(
        None, ge=16,
        description="Tamaño de ventana (muestras) para etiquetado por frame. Si None, se infiere."
    )
    hop_samples: Optional[int] = Field(
        None, ge=1,
        description="Salto entre ventanas (muestras). Si se especifica, tiene prioridad sobre overlap."
    )
    overlap: Optional[float] = Field(
        0.0, ge=0.0, le=1.0,
        description="Solapamiento en [0,1]. Se usa si hop_samples es None."
    )

    @classmethod
    def apply(cls, instance: "WaveletTransform", file_path_in: str, directory_path_out: str, labels_directory: str, labels_out_path:str) -> bool:
        """
        - Entrada: 1D (n_times) o 2D (n_channels, n_times) o (n_times, n_channels)
        - Salida:
            * señal reconstruida ventaneada (n_frames, frame_length, n_channels)
            * etiquetas por frame (n_frames,) por mayoría en cada ventana
        """
        # ---------- resolver archivo de entrada ----------
        p_in = Path(str(file_path_in)).expanduser()
        if not p_in.exists():
            raise FileNotFoundError(f"No existe el archivo: {p_in}")

        # ---------- registrar la transform ----------
        Experiment.add_transform_config(instance)
        uid = instance.get_id()

        # ---------- cargar datos ----------
        X = _load_numeric_array(str(p_in))
        orig_was_1d = False
        if X.ndim == 1:
            X = X[np.newaxis, :]
            orig_was_1d = True
        elif X.ndim != 2:
            raise ValueError(f"Se esperaba 1D o 2D; recibido ndim={X.ndim}")

        input_shape = (int(X.shape[0]), int(X.shape[1]))

        # (n_channels, n_times)
        transposed = False
        if X.shape[0] > X.shape[1]:
            X_raw = X.T
            transposed = True
        else:
            X_raw = X

        n_channels, n_times = int(X_raw.shape[0]), int(X_raw.shape[1])

        # ---------- Wavelet y modo ----------
        try:
            wave = pywt.Wavelet(instance.wavelet)
        except Exception as e:
            raise ValueError(f"Wavelet no válida: {instance.wavelet}") from e

        valid_modes = set(m.lower() for m in pywt.Modes.modes)
        mode_in = (instance.mode or "symmetric").lower()
        aliases = {
            "periodic": "periodization", "per": "periodization", "ppd": "periodization",
            "reflect": "symmetric", "sym": "symmetric",
            "const": "constant", "zpd": "zero", "sp1": "smooth"
        }
        mode = aliases.get(mode_in, mode_in)
        if mode not in valid_modes:
            raise ValueError(
                f"Modo de borde inválido: '{instance.mode}'. "
                f"Usa uno de: {sorted(valid_modes)}. Tip: en PyWavelets es 'periodization', no 'periodic'."
            )

        # Nivel de descomposición
        max_level = pywt.dwt_max_level(data_len=n_times, filter_len=wave.dec_len)
        if instance.level is None:
            L = max(1, min(10, max_level))
        else:
            L = int(instance.level)
            if L < 1 or L > min(10, max_level):
                raise ValueError(f"Nivel inválido: {L}. Permitido hasta {min(10, max_level)} para longitud {n_times}.")

        thr = float(instance.threshold) if instance.threshold is not None else None
        do_denoise = thr is not None and thr >= 0.0

        # ---------- Reconstrucción Wavelet por canal ----------
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

            # ajuste de longitud
            if rec.shape[0] != n_times:
                if rec.shape[0] > n_times:
                    rec = rec[:n_times]
                else:
                    rec = np.pad(rec, (0, n_times - rec.shape[0]), mode="edge")
            Y_std[ch] = rec

        # salida tiempo x canal (señal continua reconstruida)
        Y_continuous = Y_std.T.astype(np.float32, copy=False)  # (n_times, n_channels)

        # ---------- Parámetros de ventaneo (para DATOS y etiquetas) ----------
        # frame_length
        if instance.frame_length is not None:
            frame_length = int(instance.frame_length)
        else:
            # heurística: si hay suficientes muestras, >=256; si no, pot2 <= n_times (min 16)
            if n_times >= 256:
                fl = 256
                while (fl << 1) <= n_times:
                    fl <<= 1
                frame_length = max(256, fl)
            else:
                frame_length = 1 << int(np.floor(np.log2(max(16, n_times))))
                frame_length = max(16, int(frame_length))
        if frame_length < 16:
            raise ValueError(f"frame_length demasiado pequeño: {frame_length}")

        # hop
        if instance.hop_samples is not None:
            hop = int(instance.hop_samples)
            if hop < 1:
                raise ValueError("hop_samples debe ser >= 1.")
        else:
            ov = float(instance.overlap or 0.0)
            if not (0.0 <= ov <= 1.0):
                raise ValueError("overlap debe estar en [0.0, 1.0].")
            if ov == 1.0:
                ov = 0.99
            hop = max(1, int(round(frame_length * (1.0 - ov))))

        # padding temporal (solo para cálculo de frames/labels; NO afecta Y_out)
        if n_times < frame_length:
            n_times_eff = frame_length
        else:
            n_times_eff = n_times

        n_frames = 1 + (n_times_eff - frame_length) // hop
        if n_frames <= 0:
            n_frames = 1

        # ---------- Ventaneo de DATOS: convertir (n_times, n_channels) → (n_frames, frame_length, n_channels) ----------
        # Padding si n_times < frame_length
        if n_times < frame_length:
            pad_width = frame_length - n_times
            Y_padded = np.pad(Y_continuous, ((0, pad_width), (0, 0)), mode='edge')
        else:
            Y_padded = Y_continuous

        # Crear ventanas usando stride_tricks
        Y_windowed = np.empty((n_frames, frame_length, n_channels), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop
            end = start + frame_length
            if end <= Y_padded.shape[0]:
                Y_windowed[i] = Y_padded[start:end]
            else:
                # Último frame con padding si es necesario
                remaining = Y_padded.shape[0] - start
                Y_windowed[i, :remaining] = Y_padded[start:]
                Y_windowed[i, remaining:] = 0.0  # padding con ceros

        Y_out = Y_windowed  # (n_frames, frame_length, n_channels)

        # ESTANDARIZACIÓN: Asegurar que todas las ventanas tengan exactamente (frame_length, n_channels)
        if Y_out.shape[1] != frame_length:
            # Si por alguna razón la dimensión 1 no es frame_length, ajustar con padding de ceros
            Y_fixed = np.zeros((Y_out.shape[0], frame_length, n_channels), dtype=np.float32)
            min_len = min(Y_out.shape[1], frame_length)
            Y_fixed[:, :min_len, :] = Y_out[:, :min_len, :]
            Y_out = Y_fixed

        output_shape = (int(Y_out.shape[0]), int(Y_out.shape[1]), int(Y_out.shape[2]))

        # ---------- Etiquetas por frame (mayoría) ----------
        frame_labels = None
        labels_dir = Path(str(labels_directory)).expanduser()
        candidates = [labels_dir / p_in.name, labels_dir / p_in.with_suffix(".npy").name]
        labels_path = next((c for c in candidates if c.exists()), None)

        if labels_path is not None:
            labels_arr = np.load(str(labels_path), allow_pickle=True)
            if labels_arr.ndim == 2 and labels_arr.shape[0] == 1:
                labels_arr = labels_arr.reshape(-1)
            elif labels_arr.ndim != 1:
                raise ValueError(f"Formato de etiquetas no soportado: shape={labels_arr.shape}")

            Llab = labels_arr.shape[0]
            # concordancia con n_times
            if Llab < n_times:
                labels_arr = np.concatenate([labels_arr, np.array(["None"] * (n_times - Llab), dtype=labels_arr.dtype)], axis=0)
                print(f"[WaveletTransform.apply] WARNING: etiquetas ({Llab}) < n_times ({n_times}). Pad con 'None'.")
            elif Llab > n_times:
                labels_arr = labels_arr[:n_times]
                print(f"[WaveletTransform.apply] WARNING: etiquetas ({Llab}) > n_times ({n_times}). Truncadas.")

            # si la señal es más corta que frame_length, pad etiquetas hasta frame_length para el primer frame
            if n_times_eff > n_times:
                extra = n_times_eff - n_times
                labels_arr = np.concatenate([labels_arr, np.array(['None'] * extra, dtype=labels_arr.dtype)], axis=0)

            # mayoría por ventana [i*hop, i*hop + frame_length)
            maj = []
            for i in range(n_frames):
                start = i * hop
                end = min(start + frame_length, labels_arr.shape[0])
                window_labels = labels_arr[start:end]
                if window_labels.size == 0:
                    maj.append("None")
                    continue
                counts = Counter([str(x) for x in window_labels])
                if "None" in counts and len(counts) > 1:
                    counts.pop("None", None)
                maj_label = max(counts.items(), key=lambda kv: kv[1])[0]
                maj.append(maj_label)
            frame_labels = np.array(maj, dtype=str)

            # ===== RE-ETIQUETAR A FORMATO NUMÉRICO =====
            # Pasar all_classes para mapeo consistente en multiclase
            frame_labels_numeric, id_to_class = instance.relabel_for_model(frame_labels, all_classes=instance.all_classes)
            print(f"[WaveletTransform.apply] Etiquetas convertidas a formato numérico:")
            print(f"   Mapeo: {id_to_class}")
        else:
            print(f"[WaveletTransform.apply] WARNING: No se encontró archivo de etiquetas en: {labels_dir} para {p_in.name}")
            frame_labels_numeric = None
            id_to_class = {}

        # ---------- guardar artefactos ----------
        dir_out = Path(str(directory_path_out)).expanduser()
        dir_labels_out = Path(str(labels_out_path)).expanduser()
        dir_out.mkdir(parents=True, exist_ok=True)
        dir_labels_out.mkdir(parents=True,exist_ok=True)

        out_npy = dir_out / f"{p_in.stem}_wavelet_{uid}.npy"
        np.save(str(out_npy), Y_out)
        print(f"[WaveletTransform.apply] Guardado señal wavelet: {out_npy}")

        if frame_labels_numeric is not None:
            # Guardar labels numéricas
            out_labels = dir_labels_out / f"{p_in.stem}_wavelet_{uid}_labels.npy"
            np.save(str(out_labels), frame_labels_numeric)
            print(f"[WaveletTransform.apply] Guardado etiquetas numéricas: {out_labels}")

            # Guardar mapping (id → clase string)
            import json
            out_mapping = dir_labels_out / f"{p_in.stem}_wavelet_{uid}_mapping.json"
            with open(str(out_mapping), 'w') as f:
                json.dump(id_to_class, f, indent=2)
            print(f"[WaveletTransform.apply] Guardado mapeo de clases: {out_mapping}")

        # ---------- registrar cambio de dimensionalidad (sin meta externo) ----------
        Experiment.set_last_transform_dimensionality_change(
            input_shape=input_shape,
            standardized_to="(n_channels, n_times)",
            transposed_from_input=bool(transposed),
            orig_was_1d=bool(orig_was_1d),
            output_shape=output_shape,
            output_axes_semantics={
                "axis0": "time_frames (start = i*hop, length = frame_length)",
                "axis1": "time_in_frame",
                "axis2": "channels"
            }
        )

        return True
