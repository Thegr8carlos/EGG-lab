from backend.classes.Filter.Filter import Filter
from pydantic import Field
from typing import Optional, Literal, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import pywt
from backend.helpers.numeric_array import _load_numeric_array

class WaveletsBase(Filter):
    wavelet: Literal['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db8', 'sym2', 'sym3', 'sym4', 'sym5', 'coif1', 'coif2', 'coif3', 'coif5', 'haar'] = Field(
        'db4',
        description="Nombre de la wavelet a usar"
    )
    level: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Nivel de descomposición (opcional). Si es None, se calcula el máximo posible."
    )
    mode: Literal['symmetric', 'periodic', 'constant', 'zero', 'smooth'] = Field(
        'symmetric',
        description="Modo de extensión de bordes para PyWavelets"
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        description="Umbral global para denoising (si se omite, se usa umbral universal por canal)."
    )

    @classmethod
    def apply(cls, instance: "WaveletsBase", file_path: str, directory_path_out: str) -> bool:
        """
        Aplica denoising por wavelets canal-por-canal sobre un .npy (1D o 2D) y
        guarda únicamente un archivo .npy en `directory_path_out` con el patrón:
            <stem>_wav_<id>.npy

        Devuelve True si se guardó correctamente.
        """
        # ---------- resolver archivo ----------
        p_in = Path(str(file_path)).expanduser()
        if not p_in.exists():
            raise FileNotFoundError(f"No existe el archivo: {p_in}")

        # ---------- cargar datos (robusto a pickle) ----------
        X = _load_numeric_array(str(p_in))
        orig_was_1d = False
        if X.ndim == 1:
            X = X[np.newaxis, :]           # (1, n_times)
            orig_was_1d = True
        elif X.ndim != 2:
            raise ValueError(f"Se esperaba señal 1D o 2D, recibido ndim={X.ndim}")

        # Estándar: (n_channels, n_times)
        transposed = False
        if X.shape[0] > X.shape[1]:
            X = X.T
            transposed = True

        n_channels, n_times = X.shape

        # ---------- parámetros wavelet ----------
        wavelet = pywt.Wavelet(instance.wavelet)

        valid_modes = set(m.lower() for m in pywt.Modes.modes)
        mode_in = (instance.mode or "symmetric").lower()
        aliases = {
            "periodic": "periodization",
            "per": "periodization",
            "reflect": "symmetric",
            "sym": "symmetric",
            "const": "constant",
            "zpd": "zero",
            "sp1": "smooth",
            "ppd": "periodization",
        }
        mode = aliases.get(mode_in, mode_in)
        if mode not in valid_modes:
            raise ValueError(
                f"Modo de borde inválido: '{instance.mode}'. "
                f"Usa uno de: {sorted(valid_modes)}. "
                f"Tip: en PyWavelets es 'periodization', no 'periodic'."
            )

        max_level_possible = pywt.dwt_max_level(data_len=n_times, filter_len=wavelet.dec_len)
        level = instance.level if instance.level is not None else max(1, max_level_possible)

        # ---------- denoising canal-por-canal ----------
        X_denoised = np.empty_like(X, dtype=float)

        for ch in range(n_channels):
            sig = X[ch, :]

            coeffs = pywt.wavedec(sig, wavelet=wavelet, mode=mode, level=level)
            # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]
            cA = coeffs[0]
            details = coeffs[1:]

            # umbral
            if instance.threshold is None:
                d1 = details[-1] if details else np.array([])
                if d1.size == 0:
                    sigma = 0.0
                else:
                    sigma = np.median(np.abs(d1)) / 0.6745
                thr = sigma * np.sqrt(2 * np.log(max(1, n_times)))
            else:
                thr = float(instance.threshold)

            new_details = [pywt.threshold(d, value=thr, mode="soft") for d in details]
            rec = pywt.waverec([cA] + new_details, wavelet=wavelet, mode=mode)

            # recorte / pad para igualar longitud
            if rec.shape[0] < n_times:
                pad = n_times - rec.shape[0]
                rec = np.pad(rec, (0, pad), mode="edge")
            elif rec.shape[0] > n_times:
                rec = rec[:n_times]

            X_denoised[ch, :] = rec

        # ---------- restaurar orientación ----------
        Y = X_denoised
        if transposed:
            Y = Y.T
        if orig_was_1d:
            Y = np.squeeze(Y)

        # ---------- guardar ÚNICAMENTE .npy en directorio indicado ----------
        dir_out = Path(str(directory_path_out)).expanduser()
        dir_out.mkdir(parents=True, exist_ok=True)

        out_name = f"{p_in.stem}_wav_{instance.get_id()}.npy"
        out_path = dir_out / out_name

        np.save(str(out_path), Y)
        print(f"[WaveletsBase.apply] Señal denoised guardada en: {out_path}")
        return True
