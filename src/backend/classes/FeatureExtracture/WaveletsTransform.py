from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional
from pydantic import Field
from pathlib import Path
import numpy as np
import pywt

from backend.helpers.numeric_array import _load_numeric_array
from backend.classes.Experiment import Experiment


class WaveletTransform(Transform):
    """
    Descomposición Wavelet discreta por canal con opción de denoising (soft-threshold en detalles).

    Artefacto binario único:
      <stem>_wavelet_<id>.npy    # señal resultante con forma (n_times, n_channels)

    El meta (bloque de *cambio de dimensionalidad*) se registra en el experimento activo
    usando Experiment.add_transform_config y Experiment.set_last_transform_dimensionality_change.
    """
    wavelet: str = Field(
        ...,
        description="Nombre de la wavelet a usar (ej. db4, coif5, sym8, etc.)"
    )
    level: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Nivel de descomposición (opcional); si None, usa el máximo permitido"
    )
    mode: Optional[str] = Field(
        "symmetric",
        description="Modo de extensión de bordes: symmetric, periodization, reflect, etc."
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        description="Umbral (>=0) para denoising por soft-thresholding de detalles"
    )

    @classmethod
    def apply(cls, instance: "WaveletTransform", file_path_in: str, directory_path_out: str) -> bool:
        """
        - Entrada: 1D (n_times) o 2D (n_channels, n_times) o (n_times, n_channels)
        - Salida: guarda un único .npy con la señal reconstruida (tiempo x canal) de forma (n_times, n_channels)
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

        # ---------- cargar datos (robusto) ----------
        X = _load_numeric_array(str(p_in))
        orig_was_1d = False
        if X.ndim == 1:
            X = X[np.newaxis, :]  # (1, n_times)
            orig_was_1d = True
        elif X.ndim != 2:
            raise ValueError(f"Se esperaba 1D o 2D; recibido ndim={X.ndim}")

        input_shape = (int(X.shape[0]), int(X.shape[1]))

        # Estandarizar a (n_channels, n_times)
        transposed = False
        if X.shape[0] > X.shape[1]:
            X_raw = X.T
            transposed = True
        else:
            X_raw = X

        n_channels, n_times = int(X_raw.shape[0]), int(X_raw.shape[1])

        # ---------- PyWavelets: wavelet y modo ----------
        try:
            wave = pywt.Wavelet(instance.wavelet)
        except Exception as e:
            raise ValueError(f"Wavelet no válida: {instance.wavelet}") from e

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

        # ---------- DWT + denoising opcional + reconstrucción ----------
        Y_std = np.empty_like(X_raw, dtype=np.float64)  # (n_channels, n_times)

        for ch in range(n_channels):
            sig = X_raw[ch].astype(np.float64, copy=False)
            coeffs = pywt.wavedec(sig, wavelet=wave, mode=mode, level=L)  # [cA_L, cD_L, ..., cD_1]

            if do_denoise:
                coeffs_d = [coeffs[0]]  # no umbral en aproximación
                for cd in coeffs[1:]:
                    coeffs_d.append(pywt.threshold(cd, value=thr, mode="soft"))
                rec = pywt.waverec(coeffs_d, wavelet=wave, mode=mode)
            else:
                rec = pywt.waverec(coeffs, wavelet=wave, mode=mode)

            # Ajuste de longitud por padding interno de la DWT
            if rec.shape[0] != n_times:
                if rec.shape[0] > n_times:
                    rec = rec[:n_times]
                else:
                    rec = np.pad(rec, (0, n_times - rec.shape[0]), mode="edge")

            Y_std[ch] = rec

        # Salida final (tiempo x canal)
        Y_out = Y_std.T.astype(np.float32, copy=False)
        output_shape = (int(Y_out.shape[0]), int(Y_out.shape[1]))  # (n_times, n_channels)

        # ---------- guardar: un .npy ----------
        dir_out = Path(str(directory_path_out)).expanduser()
        dir_out.mkdir(parents=True, exist_ok=True)

        out_npy = dir_out / f"{p_in.stem}_wavelet_{uid}.npy"
        np.save(str(out_npy), Y_out)

        # ---------- registrar cambio de dimensionalidad en el experimento ----------
        Experiment.set_last_transform_dimensionality_change(
            input_shape=input_shape,                       # forma original
            standardized_to="(n_channels, n_times)",       # estandarización interna usada aquí
            transposed_from_input=bool(transposed),
            orig_was_1d=bool(orig_was_1d),
            output_shape=output_shape,                     # (n_times, n_channels)
            output_axes_semantics={
                "axis0": "time",
                "axis1": "channels"
            }
        )

        print(f"[WaveletTransform.apply] Guardado único: {out_npy}")
        return True
