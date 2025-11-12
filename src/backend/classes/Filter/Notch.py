
from backend.classes.Filter.Filter import Filter
from pydantic import  Field
from typing import Optional,  Literal, Union, List
from backend.classes.dataset import Dataset
from pathlib import Path
import numpy as np
import json, os
import mne
from scipy.signal import iirnotch, filtfilt
from backend.classes.Experiment import Experiment
from backend.helpers.numeric_array import _load_numeric_array
class Notch(Filter):
    freqs: Union[float, List[float]] = Field(
        ...,
        description="Frecuencia o lista de frecuencias a atenuar (Hz)"
    )
    quality: Optional[float] = Field(
        30.0,
        ge=1.0,
        le=1000.0,
        description="Factor de calidad (Q). BW = f0 / Q. Usado para derivar notch_widths (FIR) o en iirnotch (IIR)."
    )
    method: Literal['fir', 'iir'] = Field(
        'fir',
        description="Método para aplicar el filtro notch"
    )

    @classmethod
    def apply(cls, instance: "Notch", file_path: str, directory_path_out: str) -> bool:
        """
        Aplica uno o varios notch y guarda la señal en `directory_path_out` con el patrón:
            <stem>_notch_<id>.npy
        Devuelve True si se guardó correctamente. Lanza excepciones para entradas inválidas.
        """
        # --- Resolver archivo de entrada ---
        p_in = Path(str(file_path)).expanduser()
        if not p_in.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {p_in}")

        # --- Cargar datos (robusto a .npy/.npz/pickle si tu helper lo permite) ---
        data = _load_numeric_array(str(p_in))
        print(f"[Notch.apply] Señal cargada desde: {p_in}, shape={data.shape}, dtype={data.dtype}, ndim={data.ndim}")

        # --- Convertir a float64 (MNE requiere double precision) ---
        if data.dtype != np.float64:
            data = data.astype(np.float64)
            print(f"[Notch.apply] Convertido a float64 para compatibilidad con MNE")

        # --- Asegurar 2D (n_channels, n_times) ---
        orig_was_1d = False
        if data.ndim == 1:
            data = data[np.newaxis, :]
            orig_was_1d = True
        elif data.ndim != 2:
            raise ValueError(f"Se esperaba 1D o 2D, pero data.ndim={data.ndim}")

        # Canales en filas
        transposed = False
        if data.shape[0] > data.shape[1]:
            data = data.T
            transposed = True

        # --- Frecuencia de muestreo ---
        sfreq = float(instance.get_sp())
        if sfreq <= 0:
            raise ValueError(f"sfreq debe ser > 0; recibido {sfreq}")
        nyq = sfreq / 2.0

        # --- Normalizar lista de frecuencias y validar Nyquist ---
        freqs = instance.freqs if isinstance(instance.freqs, (list, tuple, np.ndarray)) else [instance.freqs]
        freqs = [float(f) for f in freqs]
        for f0 in freqs:
            if not (0 < f0 < nyq):
                raise ValueError(f"Frecuencia fuera de rango: {f0} Hz (Nyquist={nyq} Hz)")

        method = instance.method.lower()
        Q = float(instance.quality) if instance.quality is not None else None

        # --- Filtrado ---
        if method == "fir":
            # Si hay Q, convertir a notch_widths en Hz: BW = f0 / Q
            notch_widths = None
            if Q is not None and Q > 0:
                notch_widths = np.array([f0 / Q for f0 in freqs], dtype=float)

            out = mne.filter.notch_filter(
                x=data,
                Fs=sfreq,
                freqs=np.array(freqs, dtype=float),
                notch_widths=notch_widths,  # si None, MNE usa freqs/200 (~Q≈200)
                method="fir",
                phase="zero",
                fir_window="hamming",
                verbose=False,
            )
        elif method == "iir":
            if Q is None or Q <= 0:
                Q = 30.0
            out = data.copy()
            for f0 in freqs:
                b, a = iirnotch(w0=f0, Q=Q, fs=sfreq)
                for ch in range(out.shape[0]):
                    out[ch, :] = filtfilt(b, a, out[ch, :])
        else:
            raise ValueError(f"method debe ser 'fir' o 'iir', recibido: {instance.method}")

        # --- Restaurar orientación original ---
        if transposed:
            out = out.T
        if orig_was_1d:
            out = np.squeeze(out)

        # --- Guardar en el directorio indicado con sufijo único ---
        dir_out = Path(str(directory_path_out)).expanduser()
        dir_out.mkdir(parents=True, exist_ok=True)

        out_name = f"{p_in.stem}_notch_{instance.get_id()}.npy"
        out_path = dir_out / out_name

        np.save(str(out_path), out)
        print(f"[Notch.apply] Señal (n={len(freqs)} notch) guardada en: {out_path}")
        return True
