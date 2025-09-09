
from backend.classes.Filter.Filter import Filter
from pydantic import Field
from typing import Optional,  Literal, Union, Tuple

from pathlib import Path
import os, json, numpy as np, mne

# --------------------- Bandpass ---------------------

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
    fir_window: Optional[str] = Field(
        'hamming',
        description="Ventana para diseño FIR: hamming, hann, blackman, etc."
    )
    @classmethod
    def apply(cls, instance: "BandPass", file_path: str) -> str:
        """
        Aplica un filtro sobre un .npy de señal (mono o multicanal).
        Guarda <nombre>_filt.npy y devuelve la ruta.
        Ajuste: resolución robusta de rutas tipo 'Data/...' o '_aux/...'.
        """
        # ---------- helpers de ruta ----------
        def _maybe_with_data_prefix(p: Path) -> Path:
            # si es relativo y no empieza en Data/, probamos Data/p
            if not p.is_absolute():
                p_data = Path("Data") / p
                if p_data.exists():
                    return p_data
            return p

        def _to_aux_npy(p_original: Path) -> Path:
            """
            Inserta carpeta '_aux' al nivel del archivo y agrega '.npy' al nombre completo.
            Ej: Data/sub-01/.../sub-01_ses-02_task-innerspeech_eeg.bdf
                -> Data/sub-01/.../_aux/sub-01_ses-02_task-innerspeech_eeg.bdf.npy
            """
            parent = p_original.parent / "_aux"
            parent.mkdir(parents=True, exist_ok=True)
            fname = p_original.name + ".npy" if not p_original.name.endswith(".npy") else p_original.name
            return parent / fname

        # Normaliza a Path
        p_in = Path(str(file_path)).expanduser()
        tried = []

        # 1) Tal cual
        if p_in.exists():
            resolved = p_in
        else:
            tried.append(str(p_in))
            # 2) Con prefijo Data/
            p2 = _maybe_with_data_prefix(p_in)
            if p2.exists():
                resolved = p2
            else:
                tried.append(str(p2))
                # 3) Si no es .npy, intentar mapear a _aux/<archivo>.npy
                if not p_in.name.endswith(".npy"):
                    p3 = _to_aux_npy(_maybe_with_data_prefix(p_in))
                    if p3.exists():
                        resolved = p3
                    else:
                        tried.append(str(p3))
                        # última chance: si pasaron ya una ruta que es .npy pero faltó Data/
                        if p_in.name.endswith(".npy"):
                            p4 = Path("Data") / p_in
                            if p4.exists():
                                resolved = p4
                            else:
                                tried.append(str(p4))
                                raise FileNotFoundError(
                                    f"No existe el archivo (probados): {tried}"
                                )
                        else:
                            raise FileNotFoundError(
                                f"No existe el archivo (probados): {tried}"
                            )
                else:
                    # ya es .npy, probar con Data/
                    p4 = Path("Data") / p_in
                    if p4.exists():
                        resolved = p4
                    else:
                        tried.append(str(p4))
                        raise FileNotFoundError(f"No existe el archivo (probados): {tried}")

        print(f"[BandPass.apply] Archivo de entrada resuelto a: {resolved}")

        # -------- inferir sfreq --------
        def _infer_sfreq_from_instance(inst) -> float:
            """
            Obtiene sfreq a partir de la instancia del filtro.
            Prioridad:
            1) attr 'sfreq' si existe,
            2) attr 'sp' (sampling period o sample rate, inferido).
            """
            # 1) si el modelo tiene 'sfreq' explícito
            sfreq_attr = getattr(inst, "sfreq", None)
            if sfreq_attr is not None:
                return float(sfreq_attr)

            # 2) usa 'sp' de la superclase Filter
            sp_attr = getattr(inst, "sp", None)
            if sp_attr is None:
                raise ValueError(
                    "No se pudo inferir 'sfreq': la instancia no tiene 'sfreq' ni 'sp'. "
                    "Define Filter.sp (periodo o frecuencia) o agrega sidecar .npz/.json."
                )
            sp = float(sp_attr)
            if sp <= 0:
                raise ValueError(f"Valor inválido de 'sp': {sp}")

            # Heurística robusta: sp como periodo si <= 1, como frecuencia si > 1
            return (1.0 / sp) if sp <= 1.0 else sp


        # -------- carga datos --------
        data = np.load(str(resolved), mmap_mode=None)
        orig_was_1d = False
        if data.ndim == 1:
            data = data[np.newaxis, :]
            orig_was_1d = True
        elif data.ndim != 2:
            raise ValueError(f"Se esperaba 1D o 2D, pero data.ndim={data.ndim}")

        # Asegurar (n_channels, n_times)
        transposed = False
        if data.shape[0] > data.shape[1]:
            data = data.T
            transposed = True

        sfreq = _infer_sfreq_from_instance(instance)

        nyq = sfreq / 2.0

        # -------- cortes --------
        ft = instance.filter_type
        if ft == "lowpass":
            l_freq, h_freq = None, float(instance.freq)
        elif ft == "highpass":
            l_freq, h_freq = float(instance.freq), None
        else:
            low, high = instance.freq
            low, high = float(low), float(high)
            if low >= high:
                raise ValueError("En 'bandpass', low debe ser < high.")
            l_freq, h_freq = low, min(high, nyq * 0.999)

        # -------- filtrado --------
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

        # restaurar orientación
        if transposed:
            out = out.T
        if orig_was_1d:
            out = np.squeeze(out)

        # -------- guardar y devolver ruta --------
        out_path = resolved.with_name(resolved.stem + "_filt" + resolved.suffix)
        np.save(str(out_path), out)
        print(f"[BandPass.apply] Señal filtrada guardada en: {out_path}")
        return str(out_path)