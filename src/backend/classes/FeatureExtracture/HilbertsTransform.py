from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional, Literal
from pydantic import Field
from pathlib import Path
import numpy as np
from backend.helpers.numeric_array import _load_numeric_array
from scipy.fft import dct as sp_dct
from backend.classes.Experiment import Experiment

class DCTTransform(Transform):
    """
    Aplica la Transformada Discreta del Coseno (DCT) a un arreglo 1D o 2D.

    Artefacto binario único:
      <stem>_dct_<id>.npy    # coeficientes con forma (n_times, n_channels) (tiempo x canal)

    El meta (bloque de *cambio de dimensionalidad*) se registra en el experimento activo
    usando Experiment.add_transform_config y Experiment.set_last_transform_dimensionality_change.
    """
    type: Optional[Literal[1, 2, 3, 4]] = Field(
        2,
        description="Tipo de DCT: 1, 2, 3 o 4"
    )
    norm: Optional[Literal["ortho", None]] = Field(
        None,
        description="Normalización: 'ortho' o None"
    )
    axis: Optional[int] = Field(
        -1,
        ge=-3,
        le=3,
        description="Eje sobre el que se aplica la DCT (ej. -1 para último eje)"
    )

    @classmethod
    def apply(cls, instance: "DCTTransform", file_path_in: str, directory_path_out: str) -> bool:
        """
        - Entrada: 1D (n_times) o 2D (n_channels, n_times) o (n_times, n_channels)
        - Salida: guarda un único .npy con coeficientes de forma (n_times, n_channels)
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
            X = X[np.newaxis, :]  # -> (1, n_times)
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
        coeffs = sp_dct(X_raw, type=dct_type, norm=norm, axis=axis_norm)

        # Salida estándar para el pipeline/plot: (n_times, n_channels)
        coeffs_out = coeffs.T  # partimos de (n_channels, n_times) -> (n_times, n_channels)
        output_shape = (int(coeffs_out.shape[0]), int(coeffs_out.shape[1]))

        # ---------- guardar: un .npy ----------
        dir_out = Path(str(directory_path_out)).expanduser()
        dir_out.mkdir(parents=True, exist_ok=True)

        out_npy = dir_out / f"{p_in.stem}_dct_{uid}.npy"
        np.save(str(out_npy), coeffs_out.astype(np.float32, copy=False))

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

        print(f"[DCTTransform.apply] Guardado único: {out_npy}")
        return True
