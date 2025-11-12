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
    Convierte señales 2D en 3D mediante ventaneo con soporte de overlap.

    Útil para:
    - Convertir señales crudas a formato consistente con otras transformadas
    - Generar ejemplos para modelos que esperan ventanas
    - Pre-procesar datos sin aplicar FFT/DCT/Wavelet

    Artefactos:
      <stem>_window_<id>.npy          # ventanas (n_frames, window_size, n_channels)
      <stem>_window_<id>_labels.npy   # etiquetas por frame (n_frames,)
    """
    window_size: int = Field(
        64, ge=2,
        description="Tamaño de cada ventana en muestras. Default: 64"
    )
    overlap: Optional[float] = Field(
        None,
        description="Solapamiento entre ventanas [0.0, 1.0). Si None, sin overlap (backward compatible). Default: None"
    )
    padding_mode: str = Field(
        'constant',
        description="Modo de padding cuando la señal no es múltiplo de window_size. Opciones: 'constant', 'edge', 'reflect'. Default: 'constant'"
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
        padding_mode = str(instance.padding_mode) if hasattr(instance, 'padding_mode') else 'constant'

        # Calcular hop (salto entre ventanas)
        if instance.overlap is not None:
            ov = float(instance.overlap)
            if not (0.0 <= ov < 1.0):
                raise ValueError(f"overlap debe estar en [0.0, 1.0), recibido: {ov}")
            hop = max(1, int(round(window_size * (1.0 - ov))))
        else:
            # Backward compatible: sin overlap
            hop = window_size

        # ---------- Preparar señal con padding si es necesario ----------
        if n_times < window_size:
            # Señal más corta que window_size: padding hasta window_size
            pad_width = window_size - n_times
            X_pad = np.pad(X_raw, ((0, 0), (0, pad_width)), mode=padding_mode)
            n_times_eff = window_size
            print(f"[WindowingTransform] ⚠️  Señal corta ({n_times} < {window_size}), padding {pad_width} muestras con mode='{padding_mode}'")
        else:
            X_pad = X_raw
            n_times_eff = n_times

        # ---------- Calcular número de frames (fórmula estándar) ----------
        n_frames = 1 + (n_times_eff - window_size) // hop
        if n_frames <= 0:
            n_frames = 1

        # ---------- Ventaneo con stride_tricks ----------
        # Crear ventanas usando stride_tricks para eficiencia
        X_windowed_list = []
        for ch in range(n_channels):
            sig = X_pad[ch]
            # Usar stride_tricks para crear ventanas
            frames = np.lib.stride_tricks.as_strided(
                sig,
                shape=(n_frames, window_size),
                strides=(sig.strides[0] * hop, sig.strides[0]),
                writeable=False,
            ).copy()
            X_windowed_list.append(frames)

        # Stack: (n_channels, n_frames, window_size) → (n_frames, window_size, n_channels)
        X_windowed = np.stack(X_windowed_list, axis=-1).astype(np.float32)

        # ---------- Verificar si necesitamos padding en el último frame ----------
        last_frame_start = (n_frames - 1) * hop
        last_frame_end = last_frame_start + window_size

        if last_frame_end > n_times_eff:
            # El último frame excede los datos disponibles
            pad_needed = last_frame_end - n_times_eff
            print(f"[WindowingTransform] ⚠️  Último frame incompleto, padding {pad_needed} muestras con mode='{padding_mode}'")
            # El padding ya se aplicó con stride_tricks si n_times_eff fue ajustado

        output_shape = (int(X_windowed.shape[0]), int(X_windowed.shape[1]), int(X_windowed.shape[2]))

        # Mensaje informativo sobre overlap
        if instance.overlap is not None and instance.overlap > 0:
            overlap_samples = window_size - hop
            print(f"[WindowingTransform] ℹ️  Ventaneo con overlap={instance.overlap:.2f} ({overlap_samples} muestras, hop={hop})")
        else:
            print(f"[WindowingTransform] ℹ️  Ventaneo sin overlap (hop={hop})")

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
            # Concordancia con n_times (usar n_times_eff que incluye padding)
            if Llab < n_times_eff:
                # Padding de etiquetas - usar la última etiqueta válida
                last_valid_label = labels_arr[-1] if len(labels_arr) > 0 else "None"
                labels_arr = np.concatenate([labels_arr, np.array([last_valid_label] * (n_times_eff - Llab), dtype=labels_arr.dtype)], axis=0)
                print(f"[WindowingTransform.apply] ℹ️  Etiquetas ({Llab}) < n_times_eff ({n_times_eff}). Padding con última etiqueta válida.")
            elif Llab > n_times_eff:
                labels_arr = labels_arr[:n_times_eff]
                print(f"[WindowingTransform.apply] ℹ️  Etiquetas ({Llab}) > n_times_eff ({n_times_eff}). Truncadas.")

            # Mayoría por ventana (con soporte de overlap)
            maj = []
            for i in range(n_frames):
                start = i * hop
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
        # Documentar overlap en la semántica de ejes
        overlap_desc = f" overlap={instance.overlap:.2f}" if instance.overlap is not None and instance.overlap > 0 else " no overlap"
        axis0_desc = f"time_frames (start = i*{hop}, length = {window_size},{overlap_desc})"

        Experiment.set_last_transform_dimensionality_change(
            input_shape=input_shape,
            standardized_to="(n_channels, n_times)",
            transposed_from_input=bool(transposed),
            orig_was_1d=bool(orig_was_1d),
            output_shape=output_shape,
            output_axes_semantics={
                "axis0": axis0_desc,
                "axis1": "time_in_frame",
                "axis2": "channels"
            }
        )

        return True
