from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional, Literal
from pydantic import Field
from pathlib import Path
import numpy as np
from backend.helpers.numeric_array import _load_numeric_array
from scipy.fft import dct as sp_dct
from backend.classes.Experiment import Experiment
from collections import Counter

class DCTTransform(Transform):
    """
    DCT por ventanas sobre el eje temporal.

    Artefactos binarios:
      <stem>_dct_<id>.npy          # coeficientes (n_frames, n_coeffs, n_channels)
      <stem>_dct_<id>_labels.npy   # etiquetas por frame (n_frames,)
    """
    type: Optional[Literal[1, 2, 3, 4]] = Field(
        2, description="Tipo de DCT: 1, 2, 3 o 4"
    )
    norm: Optional[Literal["ortho", None]] = Field(
        None, description="Normalización: 'ortho' o None"
    )
    # Forzaremos el eje temporal a ser el último de X_raw (axis=1 con X_raw=(n_channels,n_times)).
    # Se deja el parámetro para compatibilidad pero se ignora si no es temporal.
    axis: Optional[int] = Field(
        -1, ge=-3, le=3, description="Eje solicitado para DCT (se forzará al eje temporal)."
    )
    frame_length: Optional[int] = Field(
        None, ge=16, description="Tamaño de ventana (muestras). Si None, usa potencia de 2 >= 256 o <= n_times."
    )
    overlap: Optional[float] = Field(
        0.0, ge=0.0, le=1.0, description="Traslape entre ventanas en [0,1). 1.0 no permitido."
    )
    window: Optional[Literal["rectangular","hann","hamming","blackman"]] = Field(
        "rectangular", description="Ventana aplicada antes de la DCT dentro de cada frame."
    )

    @classmethod
    def apply(cls, instance: "DCTTransform", file_path_in: str, directory_path_out: str, labels_directory: str,labels_out_path:str) -> bool:
        """
        - Entrada: 1D (n_times) o 2D (n_channels, n_times) o (n_times, n_channels)
        - Salida:
            * coeficientes DCT: (n_frames, n_coeffs, n_channels)
            * etiquetas por frame (mayoría): (n_frames,)
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

        # ---------- parámetros de ventana ----------
        # Elegimos frame_length si no viene: pow2 >= 256 y <= n_times; si n_times<256 usamos pow2 <= n_times (>=16).
        if instance.frame_length is not None:
            frame_length = int(instance.frame_length)
        else:
            if n_times >= 256:
                # siguiente potencia de 2 >= 256 y <= n_times
                target = 1 << int(np.ceil(np.log2(256)))
                while target * 2 <= n_times:
                    target *= 2
                frame_length = target
            else:
                # potencia de 2 <= n_times (mínimo 16)
                target = 1 << int(np.floor(np.log2(max(16, n_times))))
                frame_length = max(16, int(target))

        if frame_length < 16:
            raise ValueError(f"frame_length demasiado pequeño: {frame_length}")
        if frame_length > n_times:
            # permitimos padding posterior
            pass

        ov = float(instance.overlap or 0.0)
        if not (0.0 <= ov < 1.0):
            raise ValueError("overlap debe estar en [0.0, 1.0).")
        hop = max(1, int(round(frame_length * (1.0 - ov))))

        # ventana opcional
        wname = (instance.window or "rectangular").lower()
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

        # ---------- preparar señal (padding si hace falta para al menos 1 frame) ----------
        if n_times < frame_length:
            pad_width = frame_length - n_times
            X_pad = np.pad(X_raw, ((0, 0), (0, pad_width)), mode="constant")
            n_times_eff = int(X_pad.shape[1])
        else:
            X_pad = X_raw
            n_times_eff = n_times

        n_frames = 1 + (n_times_eff - frame_length) // hop
        if n_frames <= 0:
            n_frames = 1

        # ---------- cargar etiquetas ----------
        labels_dir = Path(str(labels_directory)).expanduser()
        candidates = [
            labels_dir / p_in.name,
            labels_dir / p_in.with_suffix(".npy").name
        ]
        labels_path = next((c for c in candidates if c.exists()), None)

        frame_labels = None
        if labels_path is not None:
            labels_arr = np.load(str(labels_path), allow_pickle=True)
            # Formatos esperados: (1, n_times) o (n_times,)
            if labels_arr.ndim == 2 and labels_arr.shape[0] == 1:
                labels_arr = labels_arr.reshape(-1)
            elif labels_arr.ndim != 1:
                raise ValueError(f"Formato de etiquetas no soportado: shape={labels_arr.shape}")

            # Concordancia con n_times (antes del padding por ventana)
            L = labels_arr.shape[0]
            if L < n_times:
                labels_arr = np.concatenate([labels_arr, np.array(["None"] * (n_times - L), dtype=labels_arr.dtype)], axis=0)
                print(f"[DCTTransform.apply] WARNING: etiquetas ({L}) < n_times ({n_times}). Pad con 'None'.")
            elif L > n_times:
                labels_arr = labels_arr[:n_times]
                print(f"[DCTTransform.apply] WARNING: etiquetas ({L}) > n_times ({n_times}). Truncadas.")

            # Si hubo padding de señal para frame_length, igualar etiquetas
            if n_times_eff > n_times:
                extra = n_times_eff - n_times
                labels_arr = np.concatenate([labels_arr, np.array(["None"] * extra, dtype=labels_arr.dtype)], axis=0)

            # Ventaneo de etiquetas: mayoría en [i*hop, i*hop + frame_length)
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
        else:
            print(f"[DCTTransform.apply] WARNING: No se encontró archivo de etiquetas en: {labels_dir} para {p_in.name}")

        # ---------- DCT por frames sobre eje temporal ----------
        dct_type = int(instance.type if instance.type is not None else 2)
        if dct_type not in (1, 2, 3, 4):
            raise ValueError(f"Tipo de DCT no soportado: {dct_type}")
        norm = instance.norm  # 'ortho' o None

        # Para evitar confusiones, forzamos el eje temporal a columnas (axis=1) en X_pad (n_channels, n_times_eff)
        # y dct se aplicará por frame (longitud frame_length).
        n_coeffs = frame_length  # por defecto conservamos todos los coeficientes
        coeffs_cube = np.empty((n_frames, n_coeffs, n_channels), dtype=np.float32)

        for ch in range(n_channels):
            sig = X_pad[ch]
            # ventanas con strides
            frames = np.lib.stride_tricks.as_strided(
                sig,
                shape=(n_frames, frame_length),
                strides=(sig.strides[0] * hop, sig.strides[0]),
                writeable=False,
            ).copy()
            # ventana (si aplica)
            frames = (frames * win).astype(np.float64, copy=False)
            # DCT por frame (eje 1)
            dct_frames = sp_dct(frames, type=dct_type, norm=norm, axis=1)  # (n_frames, n_coeffs)
            coeffs_cube[:, :, ch] = dct_frames.astype(np.float32, copy=False)

        output_shape = (int(coeffs_cube.shape[0]), int(coeffs_cube.shape[1]), int(coeffs_cube.shape[2]))  # (n_frames, n_coeffs, n_channels)

        # ---------- guardar artefactos ----------
        dir_out = Path(str(directory_path_out)).expanduser()
        dir_out_labels = Path(str(labels_out_path)).expanduser()
        dir_out.mkdir(parents=True, exist_ok=True)
        dir_out_labels.mkdir(parents=True, exist_ok=True)

        out_npy = dir_out / f"{p_in.stem}_dct_{uid}.npy"
        np.save(str(out_npy), coeffs_cube)
        print(f"[DCTTransform.apply] Guardado coeficientes (frames): {out_npy}")

        if frame_labels is not None:
            out_labels = dir_out_labels / f"{p_in.stem}_dct_{uid}_labels.npy"
            np.save(str(out_labels), frame_labels)
            print(f"[DCTTransform.apply] Guardado etiquetas por frame: {out_labels}")

        # ---------- registrar cambio de dimensionalidad (sin meta externo) ----------
        Experiment.set_last_transform_dimensionality_change(
            input_shape=input_shape,
            standardized_to="(n_channels, n_times)",
            transposed_from_input=bool(transposed),
            orig_was_1d=bool(orig_was_1d),
            output_shape=output_shape,  # (n_frames, n_coeffs, n_channels)
            output_axes_semantics={
                "axis0": "time_frames (start = i*hop, length = frame_length)",
                "axis1": "dct_coeff_index",
                "axis2": "channels"
            }
        )

        return True
