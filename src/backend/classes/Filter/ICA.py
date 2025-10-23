from backend.classes.Filter.Filter import Filter
from pydantic import Field
from typing import Optional, Literal, Dict, Any
from pathlib import Path
import numpy as np
import json  # ← mantenido aunque ya no se usa; líneas relacionadas quedan comentadas
import mne
from backend.helpers.numeric_array import _load_numeric_array

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
    def apply(cls, instance: "ICA", file_path: str, directory_path_out: str) -> bool:
        """
        Descompone con ICA y guarda ÚNICAMENTE un archivo:
            <stem>_ica_<id>.npy         # contiene S_plot con forma (n_times, n_components)
        Devuelve True si se guardó correctamente.
        """
        # ---------- resolver archivo de entrada ----------
        p_in = Path(str(file_path)).expanduser()
        if not p_in.exists():
            raise FileNotFoundError(f"No existe el archivo: {p_in}")
        print(f"[ICA.apply] Archivo de entrada: {p_in}")

        # ---------- cargar datos ----------
        X = _load_numeric_array(str(p_in))
        if X.ndim == 1:
            X = X[np.newaxis, :]
            orig_was_1d = True
        elif X.ndim == 2:
            orig_was_1d = False
        else:
            raise ValueError(f"Se esperaba 1D o 2D; recibido ndim={X.ndim}")

        # Asegurar (n_channels, n_times) para MNE
        if X.shape[0] > X.shape[1]:
            X_raw = X.T
            transposed = True
        else:
            X_raw = X
            transposed = False

        n_channels, n_times = X_raw.shape

        # ---------- frecuencia de muestreo ----------
        sfreq = float(instance.get_sp())
        if sfreq <= 0:
            raise ValueError(f"sfreq debe ser > 0; recibido {sfreq}")

        # ---------- construir RawArray ----------
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

        # fuentes S: (n_components, n_times) -> guardar como (n_times, n_components)
        S = ica.get_sources(raw).get_data()
        # A = ica.mixing_matrix_      # (n_channels, n_components)       # ← no se guarda
        # W = ica.unmixing_matrix_    # (n_components, n_channels)       # ← no se guarda
        S_plot = S.T

        # ---------- preparar salida ----------
        out_dir = Path(str(directory_path_out)).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        uid = instance.get_id()
        base = p_in.stem

        p_sources_only = out_dir / f"{base}_ica_{uid}.npy"
        # p_mixing  = out_dir / f"{base}_ica_{uid}_mixing.npy"          # ← comentado
        # p_unmix   = out_dir / f"{base}_ica_{uid}_unmixing.npy"        # ← comentado
        # p_fif     = out_dir / f"{base}_ica_{uid}.fif"                 # ← comentado
        # p_meta    = out_dir / f"{base}_ica_{uid}_meta.json"           # ← comentado

        # ---------- guardar ÚNICO .npy ----------
        np.save(str(p_sources_only), S_plot)

        # --- Todo lo siguiente queda comentado por requerimiento ---
        # np.save(str(p_mixing),  A if A is not None else np.array([]))
        # np.save(str(p_unmix),   W if W is not None else np.array([]))
        # ica.save(str(p_fif), overwrite=True)
        # meta: Dict[str, Any] = dict(
        #     input=str(p_in),
        #     sfreq=sfreq,
        #     n_times=int(n_times),
        #     n_channels=int(n_channels),
        #     n_components=int(S.shape[0]),
        #     transposed_from_input=bool(transposed),
        #     method=str(instance.method),
        #     n_components_param=int(instance.numeroComponentes) if instance.numeroComponentes else None,
        #     random_state=int(instance.random_state) if instance.random_state is not None else None,
        #     max_iter=int(instance.max_iter) if instance.max_iter is not None else None,
        #     output=str(p_sources_only),
        #     id=str(uid),
        # )
        # with open(p_meta, "w", encoding="utf-8") as f:
        #     json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[ICA.apply] Guardado único: {p_sources_only}")
        return True
