from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional, Literal
from pydantic import Field
from pathlib import Path
import numpy as np

from backend.helpers.numeric_array import _load_numeric_array
from backend.classes.Experiment import Experiment
from collections import Counter


class FFTTransform(Transform):
    """
    Calcula un espectrograma por canal usando ventanas solapadas y rFFT.

    Artefactos:
      <stem>_fft_<id>.npy          -> potencia (n_frames, n_freqs, n_channels)
      <stem>_fft_<id>_labels.npy   -> etiquetas por frame (n_frames,)
    """
    window: Literal["hann", "hamming", "blackman", "rectangular"] = Field(
        "hann",
        description="Tipo de ventana: hann, hamming, blackman, rectangular"
    )
    # --- NUEVOS PARÁMETROS DE VENTANEO ---
    frame_length: Optional[int] = Field(
        None, ge=16,
        description="Tamaño de ventana en muestras. Si None, se infiere (>=256 o pot2 <= n_times)."
    )
    hop_samples: Optional[int] = Field(
        None, ge=1,
        description="Salto entre ventanas en muestras. Si se especifica, tiene prioridad sobre 'overlap'."
    )
    # --- EXISTENTES (overlap ahora es secundario a hop_samples) ---
    overlap: Optional[float] = Field(
        0.0, ge=0.0, le=1.0,
        description="Solapamiento entre ventanas en [0,1]. Se usa si hop_samples es None."
    )
    nfft: Optional[int] = Field(
        None, ge=1,
        description="Puntos de FFT. Si None, se usa nfft=frame_length. Se fuerza nfft >= frame_length."
    )

    @classmethod
    def apply(cls, instance: "FFTTransform", file_path_in: str, directory_path_out: str, labels_directory, dir_out_labels:str) -> bool:
        """
        - Entrada: 1D (n_times) o 2D (n_channels, n_times) o (n_times, n_channels)
        - Salida:
            * <stem>_fft_<uid>.npy        -> potencia (n_frames, n_freqs, n_channels)
            * <stem>_fft_<uid}_labels.npy -> etiquetas por frame (n_frames,)
        """
        # ---------- resolver archivo de entrada ----------
        p_in = Path(str(file_path_in)).expanduser()
        if not p_in.exists():
            raise FileNotFoundError(f"No existe el archivo: {p_in}")

        # ---------- registrar la transform en el experimento ----------
        Experiment.add_transform_config(instance)
        uid = instance.get_id()

        # ---------- cargar datos ----------
        X = _load_numeric_array(str(p_in))
        orig_was_1d = False
        if X.ndim == 1:
            X = X[np.newaxis, :]  # -> (1, n_times)
            orig_was_1d = True
        elif X.ndim != 2:
            raise ValueError(f"Se esperaba 1D o 2D; recibido ndim={X.ndim}")

        input_shape = (int(X.shape[0]), int(X.shape[1]))

        # Asegurar (n_channels, n_times)
        transposed = False
        if X.shape[0] > X.shape[1]:
            X_raw = X.T
            transposed = True
        else:
            X_raw = X

        n_channels, n_times = int(X_raw.shape[0]), int(X_raw.shape[1])

        # ---------- frecuencia de muestreo ----------
        sfreq = float(instance.get_sp())
        if sfreq <= 0:
            raise ValueError(f"sfreq debe ser > 0; recibido {sfreq}")

        # ---------- PARAMETRIZACIÓN DE VENTANEO ----------
        # 1) frame_length
        if instance.frame_length is not None:
            frame_length = int(instance.frame_length)
        else:
            # Heurística: si hay muestras suficientes, usar >=256; si no, potencia de 2 <= n_times (min 16)
            if n_times >= 256:
                # elegir potencia de 2 más grande <= n_times y >= 256
                fl = 256
                while (fl << 1) <= n_times:
                    fl <<= 1
                frame_length = max(256, fl)
            else:
                frame_length = 1 << int(np.floor(np.log2(max(16, n_times))))
                frame_length = max(16, int(frame_length))

        if frame_length < 16:
            raise ValueError(f"frame_length demasiado pequeño: {frame_length}")

        # 2) hop_samples u overlap
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

        # 3) nfft (asegurar nfft >= frame_length)
        if instance.nfft is not None:
            nfft = int(instance.nfft)
        else:
            nfft = frame_length
        if nfft < frame_length:
            # Promoción mínima a potencia de 2 >= frame_length
            nfft = 1 << int(np.ceil(np.log2(frame_length)))
        if nfft < 1:
            raise ValueError(f"nfft inválido: {nfft}")

        # ---------- ventana ----------
        wname = str(instance.window).lower()
        if wname == "hann":
            win = np.hanning(frame_length)
        elif wname == "hamming":
            win = np.hamming(frame_length)
        elif wname == "blackman":
            win = np.blackman(frame_length)
        elif wname == "rectangular":
            win = np.ones(frame_length, dtype=float)
        else:
            raise ValueError(f"Ventana no soportada: {instance.window}")
        win = win.astype(np.float64)

        # ---------- cargar etiquetas desde labels_directory ----------
        labels_dir = Path(str(labels_directory)).expanduser()
        candidates = [
            labels_dir / p_in.name,
            labels_dir / p_in.with_suffix(".npy").name
        ]
        labels_path = next((c for c in candidates if c.exists()), None)

        frame_labels = None

        # ---------- preparar señal (y padding si es necesario) ----------
        if n_times < frame_length:
            pad_width = frame_length - n_times
            X_pad = np.pad(X_raw, ((0, 0), (0, pad_width)), mode="constant")
            n_times_eff = int(X_pad.shape[1])
        else:
            X_pad = X_raw
            n_times_eff = n_times

        # ---------- número de frames ----------
        n_frames = 1 + (n_times_eff - frame_length) // hop
        if n_frames <= 0:
            n_frames = 1

        # ---------- etiquetas -> ventaneo por mayoría ----------
        if labels_path is not None:
            labels_arr = np.load(str(labels_path), allow_pickle=True)
            # Esperado: (1, n_times) o (n_times,)
            if labels_arr.ndim == 2 and labels_arr.shape[0] == 1:
                labels_arr = labels_arr.reshape(-1)
            elif labels_arr.ndim != 1:
                raise ValueError(f"Formato de etiquetas no soportado: shape={labels_arr.shape}")

            # Concordancia de longitud con la señal (pre-padding de frames)
            L = labels_arr.shape[0]
            if L < n_times:
                pad = np.array(["None"] * (n_times - L), dtype=labels_arr.dtype)
                labels_arr = np.concatenate([labels_arr, pad], axis=0)
                print(f"[FFTTransform.apply] WARNING: etiquetas ({L}) < n_times ({n_times}). Pad con 'None'.")
            elif L > n_times:
                labels_arr = labels_arr[:n_times]
                print(f"[FFTTransform.apply] WARNING: etiquetas ({L}) > n_times ({n_times}). Truncadas.")

            # Si hubo padding por ser n_times < frame_length, igualar etiquetas
            if n_times_eff > n_times:
                extra = n_times_eff - n_times
                labels_arr = np.concatenate([labels_arr, np.array(["None"] * extra, dtype=labels_arr.dtype)], axis=0)

            # Mayoría en cada ventana [i*hop, i*hop+frame_length)
            majority = []
            for i in range(n_frames):
                start = i * hop
                end = min(start + frame_length, labels_arr.shape[0])
                window_labels = labels_arr[start:end]
                if window_labels.size == 0:
                    majority.append("None")
                    continue
                counts = Counter([str(x) for x in window_labels])
                if "None" in counts and len(counts) > 1:
                    counts.pop("None", None)
                maj_label = max(counts.items(), key=lambda kv: kv[1])[0]
                majority.append(maj_label)
            frame_labels = np.array(majority, dtype=str)
        else:
            print(f"[FFTTransform.apply] WARNING: No se encontró archivo de etiquetas en: {labels_dir} para {p_in.name}")

        # ---------- precomputos FFT ----------
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sfreq)
        n_freqs = int(freqs.shape[0])
        power = np.empty((n_frames, n_freqs, n_channels), dtype=np.float32)

        # normalización por energía de la ventana
        win_norm = np.sqrt((win ** 2).sum())

        # ---------- cálculo por canal ----------
        for ch in range(n_channels):
            sig = X_pad[ch]
            frames = np.lib.stride_tricks.as_strided(
                sig,
                shape=(n_frames, frame_length),
                strides=(sig.strides[0] * hop, sig.strides[0]),
                writeable=False,
            ).copy()
            frames *= win
            # rFFT con tamaño nfft (>= frame_length). Si nfft>frame_length, es zero-padding espectral.
            spec = np.fft.rfft(frames, n=nfft, axis=1)            # (n_frames, n_freqs)
            pxx = (np.abs(spec) ** 2) / (win_norm ** 2 + 1e-12)   # potencia
            power[:, :, ch] = pxx.astype(np.float32, copy=False)

        # ESTANDARIZACIÓN: Asegurar que todas las salidas tengan exactamente (n_frames, n_freqs, n_channels)
        # Si por alguna razón n_freqs difiere, ajustar con padding de ceros
        expected_n_freqs = n_freqs
        if power.shape[1] != expected_n_freqs:
            power_fixed = np.zeros((power.shape[0], expected_n_freqs, power.shape[2]), dtype=np.float32)
            min_freqs = min(power.shape[1], expected_n_freqs)
            power_fixed[:, :min_freqs, :] = power[:, :min_freqs, :]
            power = power_fixed

        output_shape = (int(power.shape[0]), int(power.shape[1]), int(power.shape[2]))  # (n_frames, n_freqs, n_channels)

        # ---------- guardar salidas ----------
        dir_out = Path(str(directory_path_out)).expanduser()
        dir_out_labels = Path(str(dir_out_labels)).expanduser()
        dir_out.mkdir(parents=True, exist_ok=True)
        dir_out_labels.mkdir(parents=True, exist_ok=True)

        out_npy = dir_out / f"{p_in.stem}_fft_{uid}.npy"
        np.save(str(out_npy), power)
        print(f"[FFTTransform.apply] Guardado potencia: {out_npy}")

        if frame_labels is not None:
            out_labels = dir_out_labels / f"{p_in.stem}_fft_{uid}_labels.npy"
            np.save(str(out_labels), frame_labels)
            print(f"[FFTTransform.apply] Guardado etiquetas por frame: {out_labels}")

        # ---------- registrar cambio de dimensionalidad (sin guardar meta) ----------
        Experiment.set_last_transform_dimensionality_change(
            input_shape=input_shape,
            standardized_to="(n_channels, n_times)",
            transposed_from_input=bool(transposed),
            orig_was_1d=bool(orig_was_1d),
            output_shape=output_shape,
            output_axes_semantics={
                "axis0": f"time_frames (start=i*hop, length={frame_length})",
                "axis1": "frequencies (Hz, rFFT)",
                "axis2": "channels"
            }
        )

        return True
