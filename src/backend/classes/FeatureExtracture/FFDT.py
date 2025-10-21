from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional, Literal
from pydantic import Field
from pathlib import Path
import numpy as np

from backend.helpers.numeric_array import _load_numeric_array
from backend.classes.Experiment import Experiment


class FFTTransform(Transform):
    """
    Calcula un espectrograma por canal usando ventanas solapadas y FFT real (rFFT).

    Artefacto binario único:
      <stem>_fft_<id>.npy  # potencia con forma (n_frames, n_freqs, n_channels)

    El *meta* (cambio de dimensionalidad) se registra en el experimento activo mediante
    `Experiment.add_transform_config` y `Experiment.set_last_transform_dimensionality_change`.
    """
    window: Literal["hann", "hamming", "blackman", "rectangular"] = Field(
        "hann",
        description="Tipo de ventana: hann, hamming, blackman, rectangular"
    )
    nfft: Optional[int] = Field(
        None,
        ge=1,
        description="Número de puntos para la FFT (opcional)"
    )
    overlap: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Porcentaje de solapamiento (entre 0.0 y 1.0)"
    )

    @classmethod
    def apply(cls, instance: "FFTTransform", file_path_in: str, directory_path_out: str) -> bool:
        """
        - Entrada: 1D (n_times) o 2D (n_channels, n_times) o (n_times, n_channels)
        - Salida: guarda un único .npy con potencia de forma (n_frames, n_freqs, n_channels)
        - Meta: inserta el bloque 'dimensionality_change' en el experimento activo
        """
        # ---------- resolver archivo de entrada ----------
        p_in = Path(str(file_path_in)).expanduser()
        if not p_in.exists():
            raise FileNotFoundError(f"No existe el archivo: {p_in}")

        # ---------- registrar la transform en el experimento (para obtener ID autoincremental) ----------
        # Esto asignará instance.id si la propiedad existe en la clase base
        Experiment.add_transform_config(instance)
        uid = instance.get_id()

        # ---------- cargar datos (robusto a pickle) ----------
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

        # ---------- parametrización de ventanas ----------
        if instance.nfft is not None:
            nfft = int(instance.nfft)
        else:
            target = int(max(256, 2 ** int(np.ceil(np.log2(max(16.0, sfreq))))))
            if n_times >= target:
                nfft = target
            else:
                nfft = n_times if n_times < 64 else 2 ** int(np.floor(np.log2(n_times)))
            nfft = max(16, int(nfft))

        if nfft < 1:
            raise ValueError(f"nfft inválido: {nfft}")

        ov = float(instance.overlap or 0.0)
        if not (0.0 <= ov <= 1.0):
            raise ValueError("overlap debe estar en [0.0, 1.0].")
        if ov == 1.0:
            ov = 0.99
        hop = max(1, int(round(nfft * (1.0 - ov))))

        # seleccionar ventana
        wname = str(instance.window).lower()
        if wname == "hann":
            win = np.hanning(nfft)
        elif wname == "hamming":
            win = np.hamming(nfft)
        elif wname == "blackman":
            win = np.blackman(nfft)
        elif wname == "rectangular":
            win = np.ones(nfft, dtype=float)
        else:
            raise ValueError(f"Ventana no soportada: {instance.window}")
        win = win.astype(np.float64)

        # ---------- generar frames ----------
        if n_times < nfft:
            pad_width = nfft - n_times
            X_pad = np.pad(X_raw, ((0, 0), (0, pad_width)), mode="constant")
            n_times_eff = int(X_pad.shape[1])
        else:
            X_pad = X_raw
            n_times_eff = n_times

        n_frames = 1 + (n_times_eff - nfft) // hop
        if n_frames <= 0:
            n_frames = 1

        # ---------- precomputos ----------
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sfreq)  # (n_freqs,)
        n_freqs = int(freqs.shape[0])

        power = np.empty((n_frames, n_freqs, n_channels), dtype=np.float32)

        # normalización por energía de la ventana
        win_norm = np.sqrt((win ** 2).sum())

        # ---------- cálculo por canal ----------
        for ch in range(n_channels):
            sig = X_pad[ch]
            frames = np.lib.stride_tricks.as_strided(
                sig,
                shape=(n_frames, nfft),
                strides=(sig.strides[0] * hop, sig.strides[0]),
                writeable=False,
            ).copy()
            frames *= win
            spec = np.fft.rfft(frames, n=nfft, axis=1)          # (n_frames, n_freqs)
            pxx = (np.abs(spec) ** 2) / (win_norm ** 2 + 1e-12) # potencia
            power[:, :, ch] = pxx.astype(np.float32, copy=False)

        output_shape = (int(power.shape[0]), int(power.shape[1]), int(power.shape[2]))  # (n_frames, n_freqs, n_channels)

        # ---------- guardar ÚNICO .npy ----------
        dir_out = Path(str(directory_path_out)).expanduser()
        dir_out.mkdir(parents=True, exist_ok=True)

        out_npy = dir_out / f"{p_in.stem}_fft_{uid}.npy"
        np.save(str(out_npy), power)

        # ---------- registrar cambio de dimensionalidad en el experimento ----------
        Experiment.set_last_transform_dimensionality_change(
            input_shape=input_shape,                        # forma original (pre-estandarización)
            standardized_to="(n_channels, n_times)",
            transposed_from_input=bool(transposed),
            orig_was_1d=bool(orig_was_1d),
            output_shape=output_shape,                      # (n_frames, n_freqs, n_channels)
            output_axes_semantics={
                "axis0": "time_frames (center of window)",
                "axis1": "frequencies (Hz, rFFT)",
                "axis2": "channels"
            }
        )

        print(f"[FFTTransform.apply] Guardado único: {out_npy}")
        return True
