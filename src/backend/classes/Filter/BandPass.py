from backend.classes.Filter.Filter import Filter
from pydantic import Field
from typing import Optional, Literal, Union, Tuple
from pathlib import Path
import numpy as np
import mne

class BandPass(Filter):
    filter_type: Literal['lowpass', 'highpass', 'bandpass'] = Field(
        'bandpass',
        description="Tipo de filtro: lowpass, highpass o bandpass"
    )
    freq: Union[float, Tuple[float, float]] = Field(
        ...,
        description="Frecuencia de corte: una sola (float) o un par (low, high)"
    )
    method: Literal['fir', 'iir'] = Field(
        'fir',
        description="Método de diseño del filtro"
    )
    order: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Orden del filtro (opcional, depende del método)"
    )
    phase: Literal['zero', 'minimum'] = Field(
        'zero',
        description="Tipo de fase para el filtro FIR"
    )
    fir_window: Literal['hamming', 'hann', 'blackman', 'bartlett', 'flattop'] = Field(
        'hamming',
        description="Ventana para diseño FIR"
    )

    @classmethod
    def apply(cls, instance: "BandPass", file_path: str, directory_path_out: str) -> bool:
        """
        Aplica el filtro (low/high/band-pass) sobre un archivo .npy y guarda el
        resultado en `directory_path_out` con el patrón:
            <stem>_bandpass_<id>.npy
        Devuelve True si se guardó correctamente. Lanza excepciones para entradas inválidas.
        """
        # --- Normaliza y valida la ruta de entrada ---
        p_in = Path(str(file_path)).expanduser()
        if not p_in.exists():
            raise FileNotFoundError(f"No existe el archivo de entrada: {p_in}")
        resolved = p_in
        print(f"[BandPass.apply] Archivo de entrada resuelto a: {resolved}")

        # --- Carga de datos ---
        data = np.load(str(resolved), mmap_mode=None)

        # --- Convertir a float64 (MNE requiere double precision) ---
        if data.dtype != np.float64:
            data = data.astype(np.float64)
            print(f"[BandPass.apply] Convertido a float64 para compatibilidad con MNE")

        orig_was_1d = False
        if data.ndim == 1:
            data = data[np.newaxis, :]
            orig_was_1d = True
        elif data.ndim != 2:
            raise ValueError(f"Se esperaba 1D o 2D, pero data.ndim={data.ndim}")

        # Asegurar forma (n_channels, n_times)
        transposed = False
        if data.shape[0] > data.shape[1]:
            data = data.T
            transposed = True

        # --- Frecuencia de muestreo ---
        sfreq = instance.get_sp()
        if sfreq <= 0:
            raise ValueError(f"sfreq debe ser > 0, pero sfreq={sfreq}")
        nyq = sfreq / 2.0
        print(f"[BandPass.apply] sfreq = {sfreq} Hz (Nyquist = {nyq} Hz)")

        # --- Validación de cortes ---
        ft = instance.filter_type
        if ft == "lowpass":
            # Asegura límite bajo None y alto < Nyquist
            h = float(instance.freq)
            if h <= 0:
                raise ValueError("En 'lowpass', la frecuencia de corte debe ser > 0.")
            h_freq = min(h, nyq * 0.999)
            l_freq = None
        elif ft == "highpass":
            # Asegura límite alto None y bajo > 0 y < Nyquist
            l = float(instance.freq)
            if l <= 0 or l >= nyq:
                raise ValueError("En 'highpass', la frecuencia de corte debe estar en (0, Nyquist).")
            l_freq, h_freq = l, None
        else:  # bandpass
            low, high = instance.freq  # type: ignore[assignment]
            low, high = float(low), float(high)
            if low <= 0 or low >= high:
                raise ValueError("En 'bandpass', se requiere 0 < low < high.")
            if high >= nyq:
                # Ajusta suavemente el límite superior si roza Nyquist
                high = nyq * 0.999
            l_freq, h_freq = low, high

        # --- Filtrado ---
        if instance.method == "fir":
            filter_length = instance.order if instance.order is not None else "auto"
            out = mne.filter.filter_data(
                data=data,
                sfreq=sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                method="fir",
                phase=instance.phase,
                fir_window=instance.fir_window or "hamming",
                filter_length=filter_length,
                verbose=False,
            )
        else:
            iir_order = instance.order if instance.order is not None else 4
            iir_params = dict(order=iir_order, ftype="butter")
            out = mne.filter.filter_data(
                data=data,
                sfreq=sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                method="iir",
                iir_params=iir_params,
                verbose=False,
            )

        # --- Restaurar orientación original ---
        if transposed:
            out = out.T
        if orig_was_1d:
            out = np.squeeze(out)

        # --- Preparar salida en el directorio indicado ---
        dir_out = Path(str(directory_path_out)).expanduser()
        dir_out.mkdir(parents=True, exist_ok=True)

        # Usa tu get_id() existente para el sufijo único
        out_name = f"{resolved.stem}_bandpass_{instance.get_id()}.npy"
        out_path = dir_out / out_name

        # --- Guardar y devolver bool ---
        np.save(str(out_path), out)
        print(f"[BandPass.apply] Señal filtrada guardada en: {out_path}")
        return True
