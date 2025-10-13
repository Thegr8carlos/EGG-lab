from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional, Literal, Dict, Any
from pydantic import Field
from pathlib import Path
import numpy as np
import json

from backend.helpers.numeric_array import _load_numeric_array
from backend.classes.Experiment import Experiment


class FFTTransform(Transform):
    """
    Calcula un espectrograma por canal usando ventanas solapadas y FFT real (rFFT).

    Entradas:
      - Archivo con señal 1D (n_times) o 2D (n_channels, n_times) o (n_times, n_channels).
      - `sfreq` o `sp` disponible en la instancia (igual que en ICA):
          * sfreq = Hz directamente
          * sp     = periodo (s); si sp <= 1, sfreq = 1/sp  (p.ej. sp=0.002 -> 500 Hz)

    Salidas (en _fft):
      - <base>_fft_power.npy   : float32, forma (n_frames, n_freqs, n_channels)
      - <base>_fft_freqs.npy   : float64, forma (n_freqs,)
      - <base>_fft_times.npy   : float64, forma (n_frames,)  (centros de ventana en segundos)
      - <base>_fft_meta.json   : metadatos
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
    def apply(cls, instance: "FFTTransform", file_path: str) -> Dict[str, Any]:
        """
        Calcula espectrograma por canal y escribe matrices en disco.
        Guarda en: Data/_aux/<lastExperiment>/.../_fft/
        Devuelve rutas a los artefactos.

        Salida principal:
          power: (n_frames, n_freqs, n_channels)  -> para plot: tiempo x frecuencia x canal
          freqs: (n_freqs,)
          times: (n_frames,)
        """
        # ---------- resolver archivo de entrada ----------
        p_in = Path(str(file_path)).expanduser()
        if not p_in.exists():
            raise FileNotFoundError(f"No existe el archivo: {p_in}")
        print(f"[FFTTransform.apply] Archivo de entrada: {p_in}")

        # ---------- cargar datos (robusto a pickle) ----------
        X = _load_numeric_array(str(p_in))
        if X.ndim == 1:
            X = X[np.newaxis, :]  # -> (1, n_times)
        elif X.ndim != 2:
            raise ValueError(f"Se esperaba 1D o 2D; recibido ndim={X.ndim}")

        # Asegurar (n_channels, n_times)
        if X.shape[0] > X.shape[1]:
            # típico input (n_times, n_channels) -> transponer
            X_raw = X.T
            transposed = True
        else:
            X_raw = X
            transposed = False

        n_channels, n_times = X_raw.shape

        # ---------- inferir sfreq ----------
        sfreq = getattr(instance, "sfreq", None)
        if sfreq is None:
            sp = getattr(instance, "sp", None)
            if sp is None:
                raise ValueError("No se pudo inferir 'sfreq': instancia sin 'sfreq' ni 'sp'.")
            sp = float(sp)
            if sp <= 0:
                raise ValueError(f"Valor inválido de 'sp': {sp}")
            sfreq = sp if sp > 1.0 else (1.0 / sp)
        sfreq = float(sfreq)

        # ---------- parametrización de ventanas ----------
        # nfft por defecto: potencia de 2 más cercana >= 1 s de datos o 256, lo que sea mayor
        if instance.nfft is not None:
            nfft = int(instance.nfft)
        else:
            # heurística: al menos 256; si hay suficiente datos, ~1 s
            target = int(max(256, 2 ** int(np.ceil(np.log2(max(16, sfreq))))))
            nfft = min(target, n_times) if n_times >= target else min(2 ** int(np.floor(np.log2(n_times))) if n_times >= 64 else n_times, n_times)
            nfft = max(16, int(nfft))  # asegura mínimo razonable

        if nfft < 1:
            raise ValueError(f"nfft inválido: {nfft}")

        ov = float(instance.overlap or 0.0)
        if not (0.0 <= ov < 1.0):
            # permitir 1.0 solo como caso límite (ventana inmóvil) -> reducimos a 0.99
            if ov == 1.0:
                ov = 0.99
            else:
                raise ValueError("overlap debe estar en [0.0, 1.0].")

        hop = max(1, int(round(nfft * (1.0 - ov))))
        if hop <= 0:
            hop = 1

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

        # ---------- generar índices de frames ----------
        if n_times < nfft:
            # zero-pad hasta nfft para al menos 1 frame
            pad_width = nfft - n_times
            X_pad = np.pad(X_raw, ((0, 0), (0, pad_width)), mode="constant")
            n_times_eff = X_pad.shape[1]
        else:
            X_pad = X_raw
            n_times_eff = n_times

        n_frames = 1 + (n_times_eff - nfft) // hop
        if n_frames <= 0:
            n_frames = 1  # por seguridad

        # ---------- precomputos de FFT ----------
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sfreq)  # (n_freqs,)
        n_freqs = freqs.shape[0]

        # ---------- cálculo: por canal en bloques ----------
        power = np.empty((n_frames, n_freqs, n_channels), dtype=np.float32)
        times = np.empty((n_frames,), dtype=np.float64)

        # centros de ventana en muestras/segundos
        frame_starts = np.arange(n_frames) * hop
        frame_centers = frame_starts + (nfft // 2)
        times[:] = frame_centers / sfreq

        # ventana con corrección de energía (opcional): usar como está para densidad de potencia relativa
        win_norm = np.sqrt((win ** 2).sum())

        for ch in range(n_channels):
            sig = X_pad[ch]  # 1D
            # construir matriz de frames (n_frames, nfft) de forma explícita (evitar stride-tricks para simplicidad/robustez)
            frames = np.lib.stride_tricks.as_strided(
                sig,
                shape=(n_frames, nfft),
                strides=(sig.strides[0] * hop, sig.strides[0]),
                writeable=False,
            ).copy()

            # aplicar ventana
            frames *= win

            # FFT real a lo largo del eje de nfft
            spec = np.fft.rfft(frames, n=nfft, axis=1)  # (n_frames, n_freqs)
            # potencia (normalizada por energía de la ventana para comparabilidad)
            pxx = (np.abs(spec) ** 2) / (win_norm ** 2 + 1e-12)

            power[:, :, ch] = pxx.astype(np.float32, copy=False)

        # ---------- ruta de salida: Data/_aux/<lastExperiment>/.../_fft ----------
        lastExperiment = Experiment._get_last_experiment_id()

        parts = list(Path(*Path(p_in).parts).parts)
        try:
            idx = parts.index("_aux")
        except ValueError:
            try:
                idx = parts.index("Data")
            except ValueError:
                idx = 0
        parts.insert(idx + 1, str(lastExperiment))

        base_dir = Path(*parts[:-1])
        out_dir = base_dir / "_fft"
        out_dir.mkdir(parents=True, exist_ok=True)

        base = Path(parts[-1]).stem
        p_power = out_dir / f"{base}_fft_power.npy"
        p_freqs = out_dir / f"{base}_fft_freqs.npy"
        p_times = out_dir / f"{base}_fft_times.npy"
        p_meta  = out_dir / f"{base}_fft_meta.json"

        # ---------- guardar ----------
        np.save(str(p_power), power)  # (n_frames, n_freqs, n_channels)
        np.save(str(p_freqs), freqs)  # (n_freqs,)
        np.save(str(p_times), times)  # (n_frames,)

        meta = dict(
            input=str(p_in),
            sfreq=float(sfreq),
            n_times=int(n_times),
            n_channels=int(n_channels),
            transposed_from_input=bool(transposed),

            window=str(instance.window),
            nfft=int(nfft),
            overlap=float(ov),
            hop_samples=int(hop),

            n_frames=int(n_frames),
            n_freqs=int(n_freqs),

            outputs=dict(
                power=str(p_power),
                freqs=str(p_freqs),
                times=str(p_times),
            ),
            description="power shape = (n_frames, n_freqs, n_channels); freqs en Hz; times en segundos (centro de ventana)."
        )
        with open(p_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[FFTTransform.apply] Espectrograma guardado en: {p_power}")
        return {
            "power": str(p_power),
            "freqs": str(p_freqs),
            "times": str(p_times),
            "meta": str(p_meta),
        }
