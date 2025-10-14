from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional, Literal, Dict, Any
from pydantic import Field
from pathlib import Path
import numpy as np
import json

from backend.helpers.numeric_array import _load_numeric_array
from backend.classes.Experiment import Experiment


# ------------------- DCT Transform -------------------

class DCTTransform(Transform):
    """
    Aplica la Transformada Discreta del Coseno (DCT) a un arreglo 1D o 2D.
    - Acepta entrada (n_times,) o (n_channels, n_times) o (n_times, n_channels).
    - Reordena internamente a (n_channels, n_times) para consistencia.
    - Aplica DCT a lo largo del eje indicado por `axis` (por defecto el último).
    - Guarda coeficientes en: Data/_aux/<lastExperiment>/.../_dct/

    Salidas:
      - <base>_dct_coeffs.npy      : coeficientes en forma (n_channels, n_times) tras normalizar forma
      - <base>_dct_coeffs_plot.npy : (n_times, n_channels) (útil para plotters tiempo x canal)
      - <base>_dct_meta.json       : metadatos de ejecución
    """
    type: Optional[Literal[1, 2, 3, 4]] = Field(
        2,
        description="Tipo de DCT: 1, 2, 3 o 4"
    )
    norm: Optional[Literal["ortho", None]] = Field(
        None,
        description="Tipo de normalización: 'ortho' o None"
    )
    axis: Optional[int] = Field(
        -1,
        ge=-3,
        le=3,
        description="Eje sobre el que se aplica la DCT (ej. -1 para último eje)"
    )

    @classmethod
    def apply(cls, instance: "DCTTransform", file_path: str) -> Dict[str, Any]:
        """
        Aplica la DCT y guarda coeficientes + metadatos.
        """
        # ---------- resolver archivo de entrada ----------
        p_in = Path(str(file_path)).expanduser()
        if not p_in.exists():
            raise FileNotFoundError(f"No existe el archivo: {p_in}")
        print(f"[DCTTransform.apply] Archivo de entrada: {p_in}")

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

        # ---------- importar DCT de SciPy ----------
        try:
            # SciPy moderno
            from scipy.fft import dct as sp_dct
        except Exception:
            try:
                # compatibilidad
                from scipy.fftpack import dct as sp_dct
            except Exception as e:
                raise ImportError(
                    "Se requiere SciPy para DCT (scipy.fft.dct o scipy.fftpack.dct)."
                ) from e

        # ---------- parámetros DCT ----------
        dct_type = int(instance.type if instance.type is not None else 2)
        if dct_type not in (1, 2, 3, 4):
            raise ValueError(f"Tipo de DCT no soportado: {dct_type}")

        norm = instance.norm  # 'ortho' o None
        axis = int(instance.axis if instance.axis is not None else -1)

        # Normalizar axis al rango de X_raw (2D): {0, 1} o negativos equivalentes
        if axis < 0:
            axis_norm = X_raw.ndim + axis
        else:
            axis_norm = axis
        if axis_norm < 0 or axis_norm >= X_raw.ndim:
            raise ValueError(f"Eje fuera de rango para datos 2D: axis={axis} -> {axis_norm}")

        # ---------- aplicar DCT ----------
        # Nota: sp_dct aplica por eje especificado; no cambia la forma.
        coeffs = sp_dct(X_raw, type=dct_type, norm=norm, axis=axis_norm)

        # Para conveniencia del pipeline/plotter, guardamos también transpuesto
        # (tiempo x canal) si nuestros datos estándar son (canal x tiempo).
        if coeffs.shape == (n_channels, n_times):
            coeffs_plot = coeffs.T
        else:
            # En la práctica, para 2D y eje válido, la forma se mantiene igual.
            coeffs_plot = coeffs.T

        # ---------- ruta de salida: Data/_aux/<lastExperiment>/.../_dct ----------
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
        out_dir = base_dir / "_dct"
        out_dir.mkdir(parents=True, exist_ok=True)

        base = Path(parts[-1]).stem
        p_coeffs      = out_dir / f"{base}_dct_coeffs.npy"
        p_coeffs_plot = out_dir / f"{base}_dct_coeffs_plot.npy"
        #p_meta        = out_dir / f"{base}_dct_meta.json"

        # ---------- guardar ----------
        np.save(str(p_coeffs), coeffs.astype(np.float32, copy=False))
        np.save(str(p_coeffs_plot), coeffs_plot.astype(np.float32, copy=False))
        ##--------------In this section, if we need it, we can save the metadata as Experiment.
        # meta = dict(
        #     input=str(p_in),
        #     n_times=int(n_times),
        #     n_channels=int(n_channels),
        #     transposed_from_input=bool(transposed),

        #     dct_type=int(dct_type),
        #     norm=str(norm) if norm is not None else None,
        #     axis_requested=int(axis),
        #     axis_applied=int(axis_norm),

        #     outputs=dict(
        #         coeffs=str(p_coeffs),
        #         coeffs_plot=str(p_coeffs_plot),
        #     ),
        #     description="coeffs shape = (n_channels, n_times); coeffs_plot = (n_times, n_channels)."
        # )
        # with open(p_meta, "w", encoding="utf-8") as f:
        #     json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[DCTTransform.apply] Coeficientes guardados en: {p_coeffs}")
        return {
            #"coeffs": str(p_coeffs), # Only saving coeffs for now, becose we use channels x times
            "coeffs": str(p_coeffs_plot),
            #"meta": str(p_meta),
        }
