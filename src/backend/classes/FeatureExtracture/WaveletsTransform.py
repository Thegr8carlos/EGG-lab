from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional, Dict, Any
from pydantic import Field
from pathlib import Path
import numpy as np
import json

from backend.helpers.numeric_array import _load_numeric_array
from backend.classes.Experiment import Experiment


class WaveletTransform(Transform):
    """
    Aplica una descomposición Wavelet discreta por canal usando PyWavelets.
    - Entrada: array 1D (n_times) o 2D (n_channels, n_times) o (n_times, n_channels).
    - Reordena internamente a (n_channels, n_times) para consistencia con el pipeline.
    - Guarda coeficientes por canal/escala y, si hay 'threshold', también la señal
      reconstruida luego del denoising (solo umbral en detalles).
    - Salidas principales en: Data/_aux/<lastExperiment>/.../_wavelet/

    Artefactos:
      - <base>_wavelet_coeffs.npz : coeficientes por canal y nivel
      - <base>_wavelet_denoised.npy (opcional) : (n_channels, n_times)
      - <base>_wavelet_denoised_plot.npy (opcional) : (n_times, n_channels)
      - <base>_wavelet_meta.json : metadatos
    """
    wavelet: str = Field(
        ...,
        description="Nombre de la wavelet a usar (ej. db4, coif5, sym8, etc.)"
    )
    level: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Nivel de descomposición (opcional)"
    )
    mode: Optional[str] = Field(
        "symmetric",
        description="Modo de extensión de bordes: symmetric, periodic, reflect, etc."
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        description="Umbral (>=0) para denoising por soft-thresholding de detalles"
    )

    @classmethod
    def apply(cls, instance: "WaveletTransform", file_path: str) -> Dict[str, Any]:
        """
        Aplica la descomposición DWT por canal y guarda coeficientes/metadata.
        Si 'threshold' está definido, realiza denoising (soft) SOLO en detalles.
        """
        # ---------- resolver archivo de entrada ----------
        p_in = Path(str(file_path)).expanduser()
        if not p_in.exists():
            raise FileNotFoundError(f"No existe el archivo: {p_in}")
        print(f"[WaveletTransform.apply] Archivo de entrada: {p_in}")

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

        # ---------- parámetros generales ----------
        # sfreq/sp (opcional): lo incluimos en meta si está disponible para trazabilidad
        sfreq = getattr(instance, "sfreq", None)
        if sfreq is None and hasattr(instance, "sp") and instance.sp is not None:
            sp = float(instance.sp)
            if sp <= 0:
                raise ValueError(f"Valor inválido de 'sp': {sp}")
            sfreq = sp if sp > 1.0 else (1.0 / sp)
        sfreq = float(sfreq) if sfreq is not None else None

        # ---------- PyWavelets ----------
        try:
            import pywt
        except Exception as e:
            raise ImportError(
                "Se requiere PyWavelets (pywt) para usar WaveletTransform."
            ) from e

        try:
            wave = pywt.Wavelet(instance.wavelet)
        except Exception as e:
            raise ValueError(f"Wavelet no válida: {instance.wavelet}") from e

        mode = str(instance.mode or "symmetric")

        # Nivel por defecto: el máximo permitido por la señal y el filtro
        if instance.level is None:
            max_level = pywt.dwt_max_level(data_len=n_times, filter_len=wave.dec_len)
            L = max(1, min(10, max_level))  # acotar a [1,10] como tu validación
        else:
            L = int(instance.level)
            max_level = pywt.dwt_max_level(data_len=n_times, filter_len=wave.dec_len)
            if L < 1 or L > min(10, max_level):
                raise ValueError(f"Nivel inválido: {L}. Permitido hasta {min(10, max_level)} para longitud {n_times}.")

        thr = float(instance.threshold) if instance.threshold is not None else None
        do_denoise = thr is not None and thr >= 0.0

        # ---------- cálculo por canal ----------
        # Guardaremos los coeficientes en un .npz con claves por canal/escala
        coeffs_store: Dict[str, np.ndarray] = {}
        denoised = np.empty_like(X_raw, dtype=np.float64) if do_denoise else None

        for ch in range(n_channels):
            sig = X_raw[ch].astype(np.float64, copy=False)

            # Descomposición: [cA_L, cD_L, cD_{L-1}, ..., cD_1]
            coeffs = pywt.wavedec(sig, wavelet=wave, mode=mode, level=L)

            # Guardado de coeficientes por canal
            # Aprox
            coeffs_store[f"ch{ch:03d}_A{L}"] = coeffs[0].astype(np.float32, copy=False)
            # Detalles
            for i, cd in enumerate(coeffs[1:], start=1):
                lev = L - i + 1  # mapeo: coeffs[1] -> D_L, ..., coeffs[-1] -> D_1
                coeffs_store[f"ch{ch:03d}_D{lev}"] = cd.astype(np.float32, copy=False)

            # Denoising (si aplica): umbral suave solo en detalles
            if do_denoise:
                coeffs_d = [coeffs[0]]  # no umbral a la aproximación
                for cd in coeffs[1:]:
                    cd_thr = pywt.threshold(cd, value=thr, mode="soft")
                    coeffs_d.append(cd_thr)
                rec = pywt.waverec(coeffs_d, wavelet=wave, mode=mode)
                # Ajustar longitud: waverec puede devolver N' ≈ N (según padding)
                if rec.shape[0] != n_times:
                    if rec.shape[0] > n_times:
                        rec = rec[:n_times]
                    else:
                        rec = np.pad(rec, (0, n_times - rec.shape[0]), mode="constant")
                denoised[ch] = rec

        # ---------- ruta de salida: Data/_aux/<lastExperiment>/.../_wavelet ----------
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
        out_dir = base_dir / "_wavelet"
        out_dir.mkdir(parents=True, exist_ok=True)

        base = Path(parts[-1]).stem
        p_coeffs = out_dir / f"{base}_wavelet_coeffs.npz"
        p_meta   = out_dir / f"{base}_wavelet_meta.json"

        # Denos opcional
        p_deno   = out_dir / f"{base}_wavelet_denoised.npy"
        p_deno_p = out_dir / f"{base}_wavelet_denoised_plot.npy"

        # ---------- guardar ----------
        # Coeficientes: un único NPZ con claves legibles
        np.savez_compressed(str(p_coeffs), **coeffs_store)

        outputs = {
            "coeffs": str(p_coeffs),
            "meta": str(p_meta),
        }

        if do_denoise and denoised is not None:
            np.save(str(p_deno), denoised.astype(np.float32, copy=False))        # (n_channels, n_times)
            np.save(str(p_deno_p), denoised.T.astype(np.float32, copy=False))    # (n_times, n_channels)
            outputs.update({
                "denoised": str(p_deno),
                "denoised_plot": str(p_deno_p),
            })

        # Metadatos
        meta = dict(
            input=str(p_in),
            sfreq=float(sfreq) if sfreq is not None else None,
            n_times=int(n_times),
            n_channels=int(n_channels),
            transposed_from_input=bool(transposed),

            wavelet=str(instance.wavelet),
            wavelet_dec_len=int(wave.dec_len),
            mode=str(mode),
            level_requested=int(instance.level) if instance.level is not None else None,
            level_applied=int(L),
            threshold=float(thr) if do_denoise else None,

            coeffs_format="NPZ con claves ch{idx:03d}_A{L} y ch{idx:03d}_D{lev} (lev = 1..L)",
            outputs=outputs
        )
        with open(p_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[WaveletTransform.apply] Coeficientes guardados en: {p_coeffs}")
        if do_denoise:
            print(f"[WaveletTransform.apply] Señal denoised guardada en: {p_deno}")

        return outputs
