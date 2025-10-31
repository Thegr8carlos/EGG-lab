from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from typing import Optional
from pydantic import Field
from pathlib import Path
import numpy as np

from backend.helpers.numeric_array import _load_numeric_array
from backend.classes.Experiment import Experiment
from collections import Counter


class WindowingTransform(Transform):
    """
    Transformada simple que ventanea señales crudas sin aplicar ningún procesamiento.
    Convierte señales 2D en 3D mediante ventaneo sin overlap (hop = window_size).

    Útil para:
    - Convertir señales crudas a formato consistente con otras transformadas
    - Generar ejemplos para modelos que esperan ventanas
    - Pre-procesar datos sin aplicar FFT/DCT/Wavelet

    Artefactos:
      <stem>_window_<id>.npy          # ventanas (n_frames, window_size, n_channels)
      <stem>_window_<id>_labels.npy   # etiquetas por frame (n_frames,)
    """
    window_size: int = Field(
        64, ge=16,
        description="Tamaño de cada ventana en muestras. Default: 64"
    )

    @classmethod
    def apply(cls, instance: "WindowingTransform", file_path_in: str, directory_path_out: str, labels_directory: str, labels_out_path: str) -> bool:
        """
        - Entrada: 1D (n_times) o 2D (n_channels, n_times) o (n_times, n_channels)
        - Salida:
            * ventanas sin overlap (n_frames, window_size, n_channels)
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

        # Asegurar (n_channels, n_times)
        transposed = False
        if X.shape[0] > X.shape[1]:
            X_raw = X.T
            transposed = True
        else:
            X_raw = X

        n_channels, n_times = int(X_raw.shape[0]), int(X_raw.shape[1])

        # ---------- Parámetros de ventaneo ----------
        window_size = int(instance.window_size)
        hop = window_size  # Sin overlap

        # Calcular número de frames
        n_frames = n_times // window_size
        if n_frames < 1:
            # Si la señal es más corta que window_size, pad y crear 1 frame
            pad = window_size - n_times
            X_raw = np.pad(X_raw, ((0, 0), (0, pad)), mode='edge')
            n_frames = 1
            n_times_usable = window_size
        else:
            # Truncar al múltiplo más cercano
            n_times_usable = n_frames * window_size

        # ---------- Ventaneo de datos sin overlap ----------
        X_truncated = X_raw[:, :n_times_usable]  # (n_channels, n_frames * window_size)

        # Reshape: (n_channels, n_frames * window_size) → (n_frames, window_size, n_channels)
        X_windowed = X_truncated.T.reshape(n_frames, window_size, n_channels).astype(np.float32)

        # ESTANDARIZACIÓN: Asegurar que todas las ventanas tengan exactamente (window_size, n_channels)
        if X_windowed.shape[1] != window_size:
            X_fixed = np.zeros((X_windowed.shape[0], window_size, n_channels), dtype=np.float32)
            min_len = min(X_windowed.shape[1], window_size)
            X_fixed[:, :min_len, :] = X_windowed[:, :min_len, :]
            X_windowed = X_fixed

        output_shape = (int(X_windowed.shape[0]), int(X_windowed.shape[1]), int(X_windowed.shape[2]))

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
            # Concordancia con n_times
            if Llab < n_times:
                labels_arr = np.concatenate([labels_arr, np.array(["None"] * (n_times - Llab), dtype=labels_arr.dtype)], axis=0)
                print(f"[WindowingTransform.apply] WARNING: etiquetas ({Llab}) < n_times ({n_times}). Pad con 'None'.")
            elif Llab > n_times:
                labels_arr = labels_arr[:n_times]
                print(f"[WindowingTransform.apply] WARNING: etiquetas ({Llab}) > n_times ({n_times}). Truncadas.")

            # Truncar etiquetas al mismo tamaño que datos
            labels_arr = labels_arr[:n_times_usable]

            # Mayoría por ventana sin overlap
            maj = []
            for i in range(n_frames):
                start = i * window_size
                end = start + window_size
                window_labels = labels_arr[start:end]
                if window_labels.size == 0:
                    maj.append("None")
                    continue
                counts = Counter([str(x) for x in window_labels])
                if "None" in counts and len(counts) > 1:
                    counts.pop("None", None)
                if counts:
                    maj_label = max(counts.items(), key=lambda kv: kv[1])[0]
                else:
                    maj_label = "None"
                maj.append(maj_label)
            frame_labels = np.array(maj, dtype=str)
        else:
            print(f"[WindowingTransform.apply] WARNING: No se encontró archivo de etiquetas en: {labels_dir} para {p_in.name}")

        # ---------- guardar artefactos ----------
        dir_out = Path(str(directory_path_out)).expanduser()
        dir_labels_out = Path(str(labels_out_path)).expanduser()
        dir_out.mkdir(parents=True, exist_ok=True)
        dir_labels_out.mkdir(parents=True, exist_ok=True)

        out_npy = dir_out / f"{p_in.stem}_window_{uid}.npy"
        np.save(str(out_npy), X_windowed)
        print(f"[WindowingTransform.apply] Guardado ventanas: {out_npy}")

        if frame_labels is not None:
            out_labels = dir_labels_out / f"{p_in.stem}_window_{uid}_labels.npy"
            np.save(str(out_labels), frame_labels)
            print(f"[WindowingTransform.apply] Guardado etiquetas por frame: {out_labels}")

        # ---------- registrar cambio de dimensionalidad ----------
        Experiment.set_last_transform_dimensionality_change(
            input_shape=input_shape,
            standardized_to="(n_channels, n_times)",
            transposed_from_input=bool(transposed),
            orig_was_1d=bool(orig_was_1d),
            output_shape=output_shape,
            output_axes_semantics={
                "axis0": "time_frames (start = i*window_size, length = window_size, no overlap)",
                "axis1": "time_in_frame",
                "axis2": "channels"
            }
        )

        return True
