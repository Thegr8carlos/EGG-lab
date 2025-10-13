
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
    def apply(cls, instance: "Notch", file_path: str) -> str:
        """
        Aplica un filtro notch (uno o varios) sobre un .npy de señal (mono o multicanal).
        Guarda <nombre>_notch.npy y devuelve la ruta.
        - Soporta rutas tipo 'Data/...', y mapeo a '_aux/<archivo>.npy' si procede.
        - Infiera sfreq desde 'instance.sfreq' o 'instance.sp' (igual que en PassFilters).
        - FIR: usa mne.filter.notch_filter con zero-phase; si hay Q, lo traduce a notch_widths.
        - IIR: aplica un iirnotch por cada frecuencia con filtfilt (cero-fase).
        """
        
        

        # We normalize and resolve input path
        p_in = Path(str(file_path)).expanduser()
        

        if not p_in.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {p_in}")
        
        

        # First we load the data
        data = np.load(str(p_in), mmap_mode=None)
        print(f"[Notch.apply] Señal cargada desde: {p_in}, shape={data.shape}, dtype={data.dtype}, ndim={data.ndim}")
        # Then, we ensure data is 2D (n_channels, n_times)
        # therefore, with data.shape = (nChannels, nTimes)
        orig_was_1d = False
        if data.ndim == 1:
            data = data[np.newaxis, :]
            orig_was_1d = True
        elif data.ndim != 2:
            raise ValueError(f"Se esperaba 1D o 2D, pero data.ndim={data.ndim}")

        # we ensure channels are rows (n_channels, n_times)
        transposed = False
        if data.shape[0] > data.shape[1]:
            data = data.T
            transposed = True

        # we get sp (sampling period) from instance 
        sp_attr = getattr(instance, "sp", None)
        if sp_attr is None:
            raise ValueError(
                "No se pudo inferir 'sfreq': la instancia no tiene 'sfreq' ni 'sp'. "
                "Define Filter.sp (periodo o frecuencia) o agrega sidecar .npz/.json."
            )
        sp_attr = float(sp_attr)
        if sp_attr <= 0:
            raise ValueError(f"Valor inválido de 'sp': {sp_attr}")

        nyq = sp_attr / 2.0

        # We ensure freqs is a list of valid frequencies with nyquist check
        freqs = instance.freqs if isinstance(instance.freqs, (list, tuple, np.ndarray)) else [instance.freqs]
        freqs = [float(f) for f in freqs]
        for f0 in freqs:
            if not (0 < f0 < nyq):
                raise ValueError(f"Frecuencia fuera de rango: {f0} Hz (Nyquist={nyq} Hz)")

        # We prepare parameters for filtering
        method = instance.method.lower()
        Q = float(instance.quality) if instance.quality is not None else None

        if method == "fir":
            # Si hay Q, convertirlo a notch_widths (BW = f0/Q) en Hz; MNE acepta lista por-frecuencia
            notch_widths = None
            if Q is not None and Q > 0:
                notch_widths = np.array([f0 / Q for f0 in freqs], dtype=float)
            # MNE acepta múltiples bandas en FIR
            out = mne.filter.notch_filter(
                x=data,
                Fs=float(sp_attr),
                freqs=np.array(freqs, dtype=float),
                notch_widths=notch_widths,           # si None, MNE usa freqs/200 (≈Q=200)
                method="fir",
                phase="zero",
                fir_window="hamming",
                verbose=False,
            )
        elif method == "iir":
            # Aplicar un biquad notch por frecuencia en serie, por canal (cero-fase via filtfilt)
            if Q is None or Q <= 0:
                Q = 30.0  # default razonable para red
            out = data.copy()
            for f0 in freqs:
                b, a = iirnotch(w0=f0, Q=Q, fs=float(sp_attr))
                # filtfilt opera sobre el último eje (tiempo). Procesamos cada canal.
                for ch in range(out.shape[0]):
                    out[ch, :] = filtfilt(b, a, out[ch, :])
        else:
            raise ValueError(f"method debe ser 'fir' o 'iir', recibido: {instance.method}")

        # restaurar orientación
        if transposed:
            out = out.T
        if orig_was_1d:
            out = np.squeeze(out)

    
        
       # obtenemos el último experimento
        lastExperiment = Experiment._get_last_experiment_id()

        # descomponer p_in
        parts = list(p_in.parts)

        # buscar "aux" e insertar la carpeta de experimento justo después
        
        idx = parts.index("_aux")
        parts.insert(idx + 1, lastExperiment)
        print(f"[Notch.apply] parts after inserting experiment: {parts}")

        # reconstruir la ruta con la nueva carpeta
        out_dir = Path(*parts[:-1])  # todo menos el archivo
        out_dir.mkdir(parents=True, exist_ok=True)

        # definir la ruta final (con sufijo _notch)
        out_path = out_dir / (parts[-1] + "_notch" + p_in.suffix)

        # guardar el archivo filtrado
        np.save(str(out_path), out)
        print(f"[Notch.apply] Señal (n={len(freqs)} notch) guardada en: {out_path}")
        return str(out_path)
