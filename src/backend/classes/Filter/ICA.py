from backend.classes.Filter.Filter import Filter
from pydantic import Field
from typing import Optional, Literal, Dict, Any
from pathlib import Path
import numpy as np
import json, os
import mne
from backend.helpers.numeric_array import _load_numeric_array
from backend.classes.Experiment import Experiment
class ICA(Filter):
    numeroComponentes: Optional[int] = Field(
        None, ge=1, description="Número de componentes independientes (opcional)"
    )
    method: Literal['fastica', 'picard', 'infomax'] = Field(
        'fastica', description="Método ICA: fastica, picard o infomax"
    )
    random_state: Optional[int] = Field(
        None, description="Semilla aleatoria para reproducibilidad"
    )
    max_iter: Optional[int] = Field(
        200, ge=1, le=10000, description="Número máximo de iteraciones"
    )

    @classmethod
    def apply(cls, instance: "ICA", file_path: str) -> Dict[str, Any]:
        """
        Descompone con ICA y escribe activaciones (S) en forma (n_times, n_components).
        Guarda en: Data/_aux/<lastExperiment>/.../_ica/
        """
        # ---------- resolver archivo de entrada ----------
        p_in = Path(str(file_path)).expanduser()
        if not p_in.exists():
            raise FileNotFoundError(f"No existe el archivo: {p_in}")
        print(f"[ICA.apply] Archivo de entrada: {p_in}")

        # ---------- cargar datos (robusto a pickle) ----------
        X = _load_numeric_array(str(p_in))  # <-- tu helper
        if X.ndim == 1:
            X = X[np.newaxis, :]  # (1, n_times)  -> channels x times tentativo
        elif X.ndim != 2:
            raise ValueError(f"Se esperaba 1D o 2D; recibido ndim={X.ndim}")

        # Asegurar (n_channels, n_times) para MNE
        if X.shape[0] > X.shape[1]:
            # típico: (n_times, n_channels) -> transponer
            X_raw = X.T
            transposed = True
        else:
            X_raw = X
            transposed = False

        n_channels, n_times = X_raw.shape

        # ---------- inferir sfreq ----------
        sfreq = getattr(instance, "sfreq", None)
        if sfreq is None:
            sp = getattr(instance, "sp", None)
            if sp is None:
                raise ValueError("No se pudo inferir 'sfreq': instancia sin 'sfreq' ni 'sp'.")
            sp = float(sp)
            if sp <= 0:
                raise ValueError(f"Valor inválido de 'sp': {sp}")
            sfreq = sp if sp > 1.0 else (1.0 / sp)
        sfreq = float(sfreq)

        # ---------- construir RawArray mínimo ----------
        ch_names = [f"EEG{idx+1:03d}" for idx in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(X_raw, info, verbose=False)

        # ---------- ajustar ICA ----------
        ica = mne.preprocessing.ICA(
            n_components=instance.numeroComponentes,
            method=instance.method,
            random_state=instance.random_state,
            max_iter=instance.max_iter,
        )
        ica.fit(raw, verbose=False)

        # fuentes S: (n_components, n_times) -> para plotter: (n_times, n_components)
        S = ica.get_sources(raw, verbose=False).get_data()
        n_components = S.shape[0]
        S_plot = S.T

        A = ica.mixing_matrix_      # (n_channels, n_components)
        W = ica.unmixing_matrix_    # (n_components, n_channels)

        # ---------- ruta de salida: Data/_aux/<lastExperiment>/.../_ica ----------
        lastExperiment = Experiment._get_last_experiment_id()
        parts = list(p_in.parts)
        try:
            idx = parts.index("_aux")
        except ValueError:
            try:
                idx = parts.index("Data")
            except ValueError:
                idx = 0
        parts.insert(idx + 1, str(lastExperiment))   # inserta experimento

        base_dir = Path(*parts[:-1])                 # sin el archivo
        out_dir = base_dir / "_ica"                  # carpeta _ica
        out_dir.mkdir(parents=True, exist_ok=True)

        base = Path(parts[-1]).stem
        p_sources = out_dir / f"{base}_ica_sources.npy"
        p_mixing  = out_dir / f"{base}_ica_mixing.npy"
        p_unmix   = out_dir / f"{base}_ica_unmixing.npy"
        p_fif     = out_dir / f"{base}_ica.fif"
        p_meta    = out_dir / f"{base}_ica_meta.json"

        # ---------- guardar ----------
        np.save(str(p_sources), S_plot)  # (n_times, n_components)
        np.save(str(p_mixing), A if A is not None else np.array([]))
        np.save(str(p_unmix),  W if W is not None else np.array([]))
        ica.save(str(p_fif), overwrite=True)

        meta = dict(
            input=str(p_in),
            sfreq=float(sfreq),
            n_times=int(n_times),
            n_channels=int(n_channels),
            n_components=int(n_components),
            transposed_from_input=bool(transposed),
            method=str(instance.method),
            n_components_param=int(instance.numeroComponentes) if instance.numeroComponentes else None,
            random_state=int(instance.random_state) if instance.random_state is not None else None,
            max_iter=int(instance.max_iter) if instance.max_iter is not None else None,
        )
        with open(p_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[ICA.apply] Fuentes guardadas en: {p_sources}")
        return {
            "sources": str(p_sources),
            "mixing": str(p_mixing),
            "unmixing": str(p_unmix),
            "ica_fif": str(p_fif),
            "meta": str(p_meta),
        }
