from pydantic import BaseModel, Field
import os
import json
from typing import List, Optional, Dict, Any
from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from backend.classes.ClasificationModel.ClsificationModels import Classifier
from backend.classes.Filter.Filter import Filter
from backend.classes.Metrics import EvaluationMetrics
from backend.classes.dataset import Dataset


class Experiment(BaseModel):
    id: str = Field(..., description="Unique identifier for the experiment")
    transform: List[dict] = Field(
        default_factory=list,
        description="Feature transformations applied to the data"
    )
    P300Classifier: Optional[dict] = Field(
        None,
        description="Classification model configuration for P300 detection"
    )
    innerSpeachClassifier: Optional[dict] = Field(
        None,
        description="Classification model configuration for Inner Speech detection"
    )
    filters: List[dict] = Field(
        default_factory=list,
        description="List of filter configurations used before transformation"
    )
    evaluation: Optional[dict] = Field(
        None,
        description="Evaluation metrics for the experiment"
    )
    dataset: Optional[dict] = Field(
        None,
        description="Dataset configuration (path, name, extensions_enabled)"
    )

    # -------------------- Paths & I/O --------------------

    @staticmethod
    def get_experiments_dir() -> str:
        """
        Returns the absolute path to the Experiments directory and ensures it exists.
        """
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        experiments_dir = os.path.join(base, "backend", "Experiments")
        os.makedirs(experiments_dir, exist_ok=True)
        return experiments_dir

    @staticmethod
    def _get_last_experiment_path() -> str:
        """
        Returns the absolute path to the 'last_experiment.txt' file.
        """
        return os.path.join(Experiment.get_experiments_dir(), "last_experiment.txt")

    @classmethod
    def _set_last_experiment_id(cls, experiment_id: str) -> None:
        """
        Persists the given experiment_id as the last experiment.
        """
        path = cls._get_last_experiment_path()
        with open(path, "w") as f:
            f.write(experiment_id)

    @classmethod
    def _get_last_experiment_id(cls) -> str:
        """
        Returns the last experiment ID previously stored.
        """
        path = cls._get_last_experiment_path()
        if not os.path.exists(path):
            raise FileNotFoundError("No experiment has been created yet.")
        with open(path, "r") as f:
            return f.read().strip()

    @classmethod
    def create_blank_json(cls) -> None:
        """
        Creates a new experiment with autoincremented ID and saves it.
        Stores the ID for future access.
        """
        directory = cls.get_experiments_dir()
        existing_files = [
            f for f in os.listdir(directory)
            if f.startswith("experiment_") and f.endswith(".json")
        ]

        existing_ids: List[int] = []
        for f in existing_files:
            try:
                num = int(f.replace("experiment_", "").replace(".json", ""))
                existing_ids.append(num)
            except ValueError:
                continue

        next_id = max(existing_ids, default=0) + 1
        experiment_id = str(next_id)

        experiment = cls(
            id=experiment_id,
            transform=[],
            P300Classifier=None,
            innerSpeachClassifier=None,
            filters=[],
            evaluation=None,
            dataset=None,
        )

        path = os.path.join(directory, f"experiment_{experiment_id}.json")
        with open(path, "w") as f:
            json.dump(experiment.dict(), f, indent=4)

        cls._set_last_experiment_id(experiment_id)
        print(f"✅ Experimento {experiment_id} creado y registrado como el actual.")

    @classmethod
    def _load_latest_experiment(cls) -> "Experiment":
        """
        Loads and returns the last experiment as an Experiment object.
        """
        experiment_id = cls._get_last_experiment_id()
        path = os.path.join(cls.get_experiments_dir(), f"experiment_{experiment_id}.json")
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def _save_latest_experiment(cls, experiment: "Experiment") -> None:
        """
        Persists the given Experiment instance to its JSON file.
        """
        path = os.path.join(cls.get_experiments_dir(), f"experiment_{experiment.id}.json")
        with open(path, "w") as f:
            json.dump(experiment.dict(), f, indent=4)

    # -------------------- Helpers (IDs & standard blocks) --------------------

    @staticmethod
    def _blank_dimensionality_change() -> dict:
        """
        Standard block to be stored alongside each transform (not inside the transform object).
        """
        return {
            "input_shape": None,                     # e.g. (n_times,) | (n_channels, n_times) | (n_times, n_channels)
            "standardized_to": "(n_channels, n_times)",
            "transposed_from_input": None,           # bool
            "orig_was_1d": None,                     # bool
            "output_shape": None,                    # e.g. (n_times, n_channels)
            "output_axes_semantics": {
                "axis0": "time",
                "axis1": "channels"
            }
        }

    @staticmethod
    def _extract_last_id_from_list(items: Optional[List[dict]]) -> int:
        """
        Returns the max 'id' (int) found in a list of entries (transforms or filters).
        If none found, returns -1 (so next becomes 0).
        Entries are expected to contain a top-level 'id'.
        """
        last = -1
        if not items:
            return last
        for entry in items:
            if isinstance(entry, dict):
                val = entry.get("id")
                try:
                    if val is not None:
                        last = max(last, int(val))
                except Exception:
                    continue
        return last

    @staticmethod
    def _ensure_set_object_id(obj: Any, new_id: int) -> None:
        """
        If obj has attribute 'id', sets it to str(new_id). Silently ignores if not present.
        """
        try:
            setattr(obj, "id", str(new_id))
        except Exception:
            pass

    # -------------------- Dataset --------------------

    @classmethod
    def set_dataset(cls, dataset: Dataset) -> None:
        """
        Sets the dataset configuration on the latest experiment.
        """
        experiment = cls._load_latest_experiment()
        experiment.dataset = {
            "path": dataset.path,
            "name": dataset.name,
           
        }
        cls._save_latest_experiment(experiment)

    @classmethod
    def set_dataset_values(
        cls,
        path: str,
        name: str,
        
    ) -> None:
        """
        Convenience method to set the dataset using raw values.
        """
        ds = Dataset(path=path, name=name)
        
        cls.set_dataset(ds)

    # -------------------- Transforms --------------------

    @classmethod
    def add_transform_config(cls, transform: Transform) -> None:
        """
        Adds a transform configuration with an autoincremented top-level ID and
        a blank 'dimensionality_change' block (standard, outside the transform object).
        If previous transform ID is missing, it starts at 0.
        """
        experiment = cls._load_latest_experiment()

        # Autoincrement ID
        prev_id = cls._extract_last_id_from_list(experiment.transform)
        new_id = prev_id + 1  # if prev_id == -1 -> 0

        # Reflect this ID back into the transform object, if it exposes 'id'
        cls._ensure_set_object_id(transform, new_id)

        transform_name = transform.__class__.__name__
        transform_data = transform.dict() if isinstance(transform, BaseModel) else vars(transform)

        if experiment.transform is None:
            experiment.transform = []

        # 🔥 PREVENIR DUPLICADOS: Verificar si el último transform es idéntico
        if len(experiment.transform) > 0:
            last_transform = experiment.transform[-1]
            # Comparar nombre y configuración (excluyendo id)
            last_transform_name = None
            last_transform_config = {}
            for key, value in last_transform.items():
                if key not in ["id", "dimensionality_change"]:
                    last_transform_name = key
                    last_transform_config = value
                    break

            # Crear copia sin id para comparar
            new_config = {k: v for k, v in transform_data.items() if k != "id"}
            last_config = {k: v for k, v in last_transform_config.items() if k != "id"}

            if last_transform_name == transform_name and last_config == new_config:
                print(f"⚠️ [Experiment] Transform duplicado detectado, NO se agregará: {transform_name}")
                return  # No agregar duplicado

        experiment.transform.append({
            "id": new_id,
            transform_name: transform_data,
            "dimensionality_change": cls._blank_dimensionality_change(),
        })

        cls._save_latest_experiment(experiment)

    @classmethod
    def set_transform_dimensionality_change(cls, index: int, **kwargs) -> None:
        """
        Updates the 'dimensionality_change' block of the transform at the given index.
        Only the standard keys are accepted.
        """
        allowed_keys = {
            "input_shape",
            "standardized_to",
            "transposed_from_input",
            "orig_was_1d",
            "output_shape",
            "output_axes_semantics",
        }

        experiment = cls._load_latest_experiment()

        if not experiment.transform or index < 0 or index >= len(experiment.transform):
            raise IndexError(f"Índice de transform fuera de rango: {index}")

        entry = experiment.transform[index]

        if "dimensionality_change" not in entry or not isinstance(entry["dimensionality_change"], dict):
            entry["dimensionality_change"] = cls._blank_dimensionality_change()

        dimchg = entry["dimensionality_change"]
        for k, v in kwargs.items():
            if k not in allowed_keys:
                raise KeyError(f"Clave no permitida en dimensionality_change: {k}")
            dimchg[k] = v

        experiment.transform[index] = entry
        cls._save_latest_experiment(experiment)

    @classmethod
    def set_last_transform_dimensionality_change(cls, **kwargs) -> None:
        """
        Convenience method to update the 'dimensionality_change' of the last added transform.
        """
        experiment = cls._load_latest_experiment()
        if not experiment.transform:
            raise ValueError("No hay transforms para actualizar.")
        last_idx = len(experiment.transform) - 1
        cls.set_transform_dimensionality_change(last_idx, **kwargs)

    # -------------------- Filters --------------------

    @classmethod
    def add_filter_config(cls, filter_instance: Filter) -> None:
        """
        Adds a filter configuration with an autoincremented top-level ID.
        If previous filter ID is missing, it starts at 0.
        """
        experiment = cls._load_latest_experiment()

        prev_id = cls._extract_last_id_from_list(experiment.filters)
        new_id = prev_id + 1  # if prev_id == -1 -> 0

        # Reflect this ID back into the filter object, if it exposes 'id'
        cls._ensure_set_object_id(filter_instance, new_id)

        filter_name = filter_instance.__class__.__name__
        filter_data = (filter_instance.dict()
                       if isinstance(filter_instance, BaseModel)
                       else vars(filter_instance))

        if experiment.filters is None:
            experiment.filters = []

        # 🔥 PREVENIR DUPLICADOS: Verificar si el último filtro es idéntico
        if len(experiment.filters) > 0:
            last_filter = experiment.filters[-1]
            # Comparar nombre y configuración (excluyendo id)
            last_filter_name = None
            last_filter_config = {}
            for key, value in last_filter.items():
                if key != "id":
                    last_filter_name = key
                    last_filter_config = value
                    break

            # Crear copia sin id para comparar
            new_config = {k: v for k, v in filter_data.items() if k != "id"}
            last_config = {k: v for k, v in last_filter_config.items() if k != "id"}

            if last_filter_name == filter_name and last_config == new_config:
                print(f"⚠️ [Experiment] Filtro duplicado detectado, NO se agregará: {filter_name}")
                return  # No agregar duplicado

        experiment.filters.append({
            "id": new_id,
            filter_name: filter_data
        })

        cls._save_latest_experiment(experiment)

    # -------------------- Classifiers --------------------

    @classmethod
    def add_P300_classifier(cls, classifier: Classifier) -> None:
        """
        Stores the P300 classifier configuration.
        """
        experiment = cls._load_latest_experiment()
        classifier_name = classifier.__class__.__name__
        classifier_data = classifier.dict() if isinstance(classifier, BaseModel) else vars(classifier)
        experiment.P300Classifier = {
            classifier_name: classifier_data
        }
        cls._save_latest_experiment(experiment)

    @classmethod
    def add_inner_speech_classifier(cls, classifier: Classifier) -> None:
        """
        Stores the Inner Speech classifier configuration.
        """
        experiment = cls._load_latest_experiment()
        classifier_name = classifier.__class__.__name__
        classifier_data = classifier.dict() if isinstance(classifier, BaseModel) else vars(classifier)
        experiment.innerSpeachClassifier = {
            classifier_name: classifier_data
        }
        cls._save_latest_experiment(experiment)

    # -------------------- Pipeline History System --------------------

    @classmethod
    def _get_pipeline_cache_path(cls, file_path: str, experiment_id: str) -> Dict[str, str]:
        """
        Constructs cache file paths for a given event file and experiment.

        Returns:
            dict with keys: cache_file, metadata_file, cache_dir, intermediates_dir
        """
        import hashlib
        from pathlib import Path

        # Get base dataset directory (parent of the event file)
        event_path = Path(file_path)
        dataset_dir = event_path.parent

        # Create Aux/experiment_{id}/ structure
        aux_dir = dataset_dir / "Aux" / f"experiment_{experiment_id}"
        cache_dir = aux_dir / "pipeline_cache"
        intermediates_dir = aux_dir / "intermediates"

        # Hash the event filename for uniqueness
        file_hash = hashlib.md5(event_path.name.encode()).hexdigest()[:8]

        return {
            "cache_file": str(cache_dir / f"{event_path.stem}_{file_hash}_final.npy"),
            "metadata_file": str(cache_dir / f"{event_path.stem}_{file_hash}_metadata.json"),
            "cache_dir": str(cache_dir),
            "intermediates_dir": str(intermediates_dir)
        }

    @classmethod
    def _reconstruct_filter_instance(cls, filter_name: str, config: dict) -> Filter:
        """
        Reconstructs a Pydantic filter instance from JSON configuration.

        Args:
            filter_name: Name of the filter class (e.g., "ICA", "WaveletsBase")
            config: Dictionary with filter configuration

        Returns:
            Instantiated filter object
        """
        from backend.classes.Filter.FilterSchemaFactory import FilterSchemaFactory

        filter_class = FilterSchemaFactory.available_filters.get(filter_name)
        if filter_class is None:
            raise ValueError(f"Filter '{filter_name}' not found in available filters")

        return filter_class(**config)

    @classmethod
    def _reconstruct_transform_instance(cls, transform_name: str, config: dict) -> Transform:
        """
        Reconstructs a Pydantic transform instance from JSON configuration.

        Args:
            transform_name: Name of the transform class (e.g., "WaveletTransform", "FFTTransform")
            config: Dictionary with transform configuration

        Returns:
            Instantiated transform object
        """
        from backend.classes.FeatureExtracture.TransformSchemaFactory import TransformSchemaFactory

        transform_class = TransformSchemaFactory.available_transforms.get(transform_name)
        if transform_class is None:
            raise ValueError(f"Transform '{transform_name}' not found in available transforms")

        return transform_class(**config)

    @classmethod
    def apply_history_pipeline(
        cls,
        file_path: str,
        force_recalculate: bool = False,
        save_intermediates: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Applies ALL filters and transforms from experiment history to a signal file.
        Uses intelligent caching to avoid redundant computation.

        Workflow:
        1. Check if cache exists and is valid
        2. If valid cache exists and force_recalculate=False, return cached result
        3. Otherwise, execute full pipeline:
           - Apply all filters in sequence
           - Apply all transforms in sequence
           - Save final result to cache
           - Optionally save intermediate steps

        Args:
            file_path: Path to the event .npy file
            force_recalculate: If True, ignores cache and recalculates
            save_intermediates: If True, saves intermediate results after each step
            verbose: If True, prints progress messages

        Returns:
            dict with keys:
                - signal: Final transformed signal array
                - metadata: Pipeline execution info
                - cache_used: Boolean indicating if cache was used
                - cache_path: Path to cached file
        """
        import numpy as np
        import time
        import hashlib
        from pathlib import Path

        # Load experiment
        experiment = cls._load_latest_experiment()
        experiment_id = experiment.id

        # 🔍 DEBUG: Mostrar información del experimento
        if verbose:
            print(f"\n{'='*60}")
            print(f"🧪 EXPERIMENTO #{experiment_id}")
            print(f"📂 Archivo: {Path(file_path).name}")
            print(f"{'='*60}")

            # Mostrar historial completo
            n_filters = len(experiment.filters)
            n_transforms = len(experiment.transform)
            print(f"📋 Pipeline completo: {n_filters} filtros + {n_transforms} transformadas")

            if n_filters > 0:
                print(f"\n🔧 Filtros a aplicar:")
                for i, filter_entry in enumerate(experiment.filters):
                    filter_id = filter_entry.get("id", "?")
                    filter_name = None
                    for key, value in filter_entry.items():
                        if key != "id" and isinstance(value, dict):
                            filter_name = key
                            break
                    print(f"  {i+1}. F{filter_id}: {filter_name}")

            if n_transforms > 0:
                print(f"\n🎨 Transformadas a aplicar:")
                for i, transform_entry in enumerate(experiment.transform):
                    transform_id = transform_entry.get("id", "?")
                    transform_name = None
                    for key, value in transform_entry.items():
                        if key not in ["id", "dimensionality_change"] and isinstance(value, dict):
                            transform_name = key
                            break
                    print(f"  {i+1}. T{transform_id}: {transform_name}")

            print(f"{'='*60}\n")

        # Get cache paths
        cache_paths = cls._get_pipeline_cache_path(file_path, experiment_id)
        cache_file = cache_paths["cache_file"]
        metadata_file = cache_paths["metadata_file"]
        cache_dir = Path(cache_paths["cache_dir"])
        intermediates_dir = Path(cache_paths["intermediates_dir"])

        # Create directories
        cache_dir.mkdir(parents=True, exist_ok=True)
        if save_intermediates:
            intermediates_dir.mkdir(parents=True, exist_ok=True)

        # Check cache validity
        cache_exists = os.path.exists(cache_file) and os.path.exists(metadata_file)

        if cache_exists and not force_recalculate:
            # Load metadata to verify experiment hash
            with open(metadata_file, "r") as f:
                cached_metadata = json.load(f)

            # Simple hash: serialize filters + transforms config
            current_config = {
                "filters": experiment.filters,
                "transforms": experiment.transform
            }
            current_hash = hashlib.md5(
                json.dumps(current_config, sort_keys=True).encode()
            ).hexdigest()

            if cached_metadata.get("pipeline_hash") == current_hash:
                # Valid cache found
                if verbose:
                    print(f"✅ Cache válido encontrado: {cache_file}")

                signal = np.load(cache_file, allow_pickle=False)

                return {
                    "signal": signal,
                    "metadata": cached_metadata,
                    "cache_used": True,
                    "cache_path": cache_file
                }

        # No valid cache - execute full pipeline
        if verbose:
            print(f"🔄 Ejecutando pipeline completo para: {Path(file_path).name}")

        start_time = time.time()

        # Load original signal
        current_signal = np.load(file_path, allow_pickle=False)
        original_shape = current_signal.shape

        step_count = 0
        execution_log = []

        # Phase 1: Apply all filters
        if experiment.filters:
            if verbose:
                print(f"📍 Fase 1: Aplicando {len(experiment.filters)} filtros")

            for filter_entry in experiment.filters:
                filter_id = filter_entry.get("id")

                # Extract filter name and config
                filter_name = None
                filter_config = None
                for key, value in filter_entry.items():
                    if key != "id" and isinstance(value, dict):
                        filter_name = key
                        filter_config = value
                        break

                if not filter_name or not filter_config:
                    if verbose:
                        print(f"⚠️ Filtro {filter_id} sin configuración válida, saltando")
                    continue

                try:
                    # Reconstruct filter instance
                    filter_instance = cls._reconstruct_filter_instance(filter_name, filter_config)

                    if verbose:
                        print(f"  → Aplicando {filter_name} (ID: {filter_id})")

                    # Get filter class for apply method
                    from backend.classes.Filter.FilterSchemaFactory import FilterSchemaFactory
                    filter_class = FilterSchemaFactory.available_filters[filter_name]

                    # Save current signal to temp file
                    temp_input = intermediates_dir / f"temp_step_{step_count}_input.npy"
                    np.save(str(temp_input), current_signal)

                    # Create temp output directory
                    temp_output_dir = intermediates_dir / f"temp_step_{step_count}_output"
                    temp_output_dir.mkdir(exist_ok=True)

                    # Apply filter (filters take file_path and directory_path_out)
                    success = filter_class.apply(
                        filter_instance,
                        file_path=str(temp_input),
                        directory_path_out=str(temp_output_dir)
                    )

                    if success:
                        # Find the output file (filters create it automatically)
                        output_files = list(temp_output_dir.glob("*.npy"))
                        if output_files:
                            temp_output = output_files[0]
                            current_signal = np.load(str(temp_output), allow_pickle=False)

                            # Save intermediate if requested
                            if save_intermediates:
                                intermediate_file = intermediates_dir / f"step_{step_count:02d}_{filter_name}_{filter_id}.npy"
                                np.save(str(intermediate_file), current_signal)

                            execution_log.append({
                                "step": step_count,
                                "type": "filter",
                                "name": filter_name,
                                "id": filter_id,
                                "shape": list(current_signal.shape),
                                "status": "success"
                            })

                            if verbose:
                                print(f"    ✅ {filter_name} aplicado: {current_signal.shape}")

                            step_count += 1
                        else:
                            if verbose:
                                print(f"    ⚠️ {filter_name} no generó archivo de salida")
                            execution_log.append({
                                "step": step_count,
                                "type": "filter",
                                "name": filter_name,
                                "id": filter_id,
                                "status": "no_output"
                            })
                    else:
                        if verbose:
                            print(f"    ❌ {filter_name} falló")
                        execution_log.append({
                            "step": step_count,
                            "type": "filter",
                            "name": filter_name,
                            "id": filter_id,
                            "status": "failed"
                        })

                    # Cleanup temp files
                    if temp_input.exists():
                        temp_input.unlink()
                    if temp_output_dir.exists():
                        import shutil
                        shutil.rmtree(temp_output_dir)

                except Exception as e:
                    if verbose:
                        print(f"  ❌ Error aplicando filtro {filter_name}: {e}")
                    continue

        # Phase 2: Apply all transforms
        if experiment.transform:
            if verbose:
                print(f"📍 Fase 2: Aplicando {len(experiment.transform)} transformadas")

            for transform_entry in experiment.transform:
                transform_id = transform_entry.get("id")

                # Extract transform name and config
                transform_name = None
                transform_config = None
                for key, value in transform_entry.items():
                    if key not in ["id", "dimensionality_change"] and isinstance(value, dict):
                        transform_name = key
                        transform_config = value
                        break

                if not transform_name or not transform_config:
                    if verbose:
                        print(f"⚠️ Transformada {transform_id} sin configuración válida, saltando")
                    continue

                try:
                    # Reconstruct transform instance
                    transform_instance = cls._reconstruct_transform_instance(transform_name, transform_config)

                    if verbose:
                        print(f"  → Aplicando {transform_name} (ID: {transform_id})")

                    # Get transform class for apply method
                    from backend.classes.FeatureExtracture.TransformSchemaFactory import TransformSchemaFactory
                    transform_class = TransformSchemaFactory.available_transforms[transform_name]

                    # Save current signal to temp file
                    temp_input = intermediates_dir / f"temp_step_{step_count}_input.npy"
                    np.save(str(temp_input), current_signal)

                    temp_output_dir = intermediates_dir / "temp_output"
                    temp_output_dir.mkdir(exist_ok=True)

                    # Create temp labels (all same class from filename)
                    event_name = Path(file_path).stem
                    event_class = event_name.split('[')[0].strip() if '[' in event_name else event_name

                    temp_labels_dir = intermediates_dir / "temp_labels"
                    temp_labels_dir.mkdir(exist_ok=True)

                    n_samples = current_signal.shape[1] if current_signal.ndim == 2 else current_signal.shape[0]
                    labels_array = np.array([event_class] * n_samples, dtype=str)
                    temp_labels_file = temp_labels_dir / Path(file_path).name
                    np.save(str(temp_labels_file), labels_array)

                    # Apply transform
                    try:
                        success = transform_class.apply(
                            transform_instance,
                            file_path_in=str(temp_input),
                            directory_path_out=str(temp_output_dir),
                            labels_directory=str(temp_labels_dir),
                            labels_out_path=str(temp_output_dir)
                        )
                    except TypeError:
                        # Try with dir_out_labels instead
                        success = transform_class.apply(
                            transform_instance,
                            file_path_in=str(temp_input),
                            directory_path_out=str(temp_output_dir),
                            labels_directory=str(temp_labels_dir),
                            dir_out_labels=str(temp_output_dir)
                        )

                    if success:
                        # Find output file
                        transform_suffixes = {
                            "WaveletTransform": "wavelet",
                            "FFTTransform": "fft",
                            "DCTTransform": "dct",
                            "WindowingTransform": "window"
                        }
                        suffix = transform_suffixes.get(transform_name, "transformed")

                        output_files = sorted(
                            temp_output_dir.glob("*.npy"),
                            key=lambda x: x.stat().st_mtime,
                            reverse=True
                        )

                        if output_files:
                            current_signal = np.load(str(output_files[0]), allow_pickle=False)

                            # Handle 3D arrays (windowed transforms)
                            if current_signal.ndim == 3:
                                n_frames, frame_size, n_channels = current_signal.shape
                                current_signal = current_signal.transpose(2, 0, 1).reshape(n_channels, n_frames * frame_size)

                            # Save intermediate if requested
                            if save_intermediates:
                                intermediate_file = intermediates_dir / f"step_{step_count:02d}_{transform_name}_{transform_id}.npy"
                                np.save(str(intermediate_file), current_signal)

                            execution_log.append({
                                "step": step_count,
                                "type": "transform",
                                "name": transform_name,
                                "id": transform_id,
                                "shape": list(current_signal.shape),
                                "status": "success"
                            })

                            if verbose:
                                print(f"    ✅ {transform_name} aplicada: {current_signal.shape}")

                            step_count += 1
                        else:
                            if verbose:
                                print(f"    ⚠️ {transform_name} no generó archivo de salida")
                            execution_log.append({
                                "step": step_count,
                                "type": "transform",
                                "name": transform_name,
                                "id": transform_id,
                                "status": "no_output"
                            })
                    else:
                        if verbose:
                            print(f"    ❌ {transform_name} falló")
                        execution_log.append({
                            "step": step_count,
                            "type": "transform",
                            "name": transform_name,
                            "id": transform_id,
                            "status": "failed"
                        })

                    # Cleanup temp files
                    import shutil
                    if temp_input.exists():
                        temp_input.unlink()
                    if temp_output_dir.exists():
                        shutil.rmtree(str(temp_output_dir))
                    if temp_labels_file.exists():
                        temp_labels_file.unlink()

                except Exception as e:
                    if verbose:
                        print(f"  ❌ Error aplicando transformada {transform_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Save final result to cache
        np.save(cache_file, current_signal)

        # Save metadata
        current_config = {
            "filters": experiment.filters,
            "transforms": experiment.transform
        }
        pipeline_hash = hashlib.md5(
            json.dumps(current_config, sort_keys=True).encode()
        ).hexdigest()

        execution_time = time.time() - start_time

        metadata = {
            "pipeline_hash": pipeline_hash,
            "experiment_id": experiment_id,
            "original_file": file_path,
            "original_shape": list(original_shape),
            "final_shape": list(current_signal.shape),
            "execution_time_seconds": execution_time,
            "steps_applied": step_count,
            "execution_log": execution_log,
            "timestamp": time.time()
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        if verbose:
            # Resumen final del pipeline
            successful_steps = [log for log in execution_log if log.get("status") == "success"]
            failed_steps = [log for log in execution_log if log.get("status") == "failed"]
            no_output_steps = [log for log in execution_log if log.get("status") == "no_output"]

            print(f"\n{'='*60}")
            print(f"📊 RESUMEN DEL PIPELINE")
            print(f"{'='*60}")
            print(f"⏱️  Tiempo total: {execution_time:.2f}s")
            print(f"📏 Shape: {original_shape} → {current_signal.shape}")
            print(f"✅ Pasos exitosos: {len(successful_steps)}/{len(execution_log)}")

            if failed_steps:
                print(f"❌ Pasos fallidos: {len(failed_steps)}")
                for step in failed_steps:
                    print(f"   - {step['name']} (ID: {step['id']})")

            if no_output_steps:
                print(f"⚠️  Sin salida: {len(no_output_steps)}")
                for step in no_output_steps:
                    print(f"   - {step['name']} (ID: {step['id']})")

            print(f"💾 Cache guardado: {Path(cache_file).name}")
            print(f"{'='*60}\n")

        return {
            "signal": current_signal,
            "metadata": metadata,
            "cache_used": False,
            "cache_path": cache_file
        }

    @classmethod
    def get_experiment_summary(cls, calculate_cache: bool = False) -> Dict[str, Any]:
        """
        Returns a summary of the current experiment for UI display.

        Args:
            calculate_cache: If True, calculates cache size (slower). Default False for speed.

        Returns:
            dict with keys:
                - experiment_id: Current experiment ID
                - filters: List of filter summaries (name, id, config)
                - transforms: List of transform summaries (name, id, config)
                - total_steps: Total number of pipeline steps
                - cache_info: Cache size and location (only if calculate_cache=True)
        """
        experiment = cls._load_latest_experiment()

        # Summarize filters
        filters_summary = []
        for filter_entry in experiment.filters:
            filter_id = filter_entry.get("id")
            for key, value in filter_entry.items():
                if key != "id" and isinstance(value, dict):
                    filters_summary.append({
                        "id": filter_id,
                        "name": key,
                        "config": value
                    })
                    break

        # Summarize transforms
        transforms_summary = []
        for transform_entry in experiment.transform:
            transform_id = transform_entry.get("id")
            for key, value in transform_entry.items():
                if key not in ["id", "dimensionality_change"] and isinstance(value, dict):
                    transforms_summary.append({
                        "id": transform_id,
                        "name": key,
                        "config": value
                    })
                    break

        # Calculate cache size only if requested (performance optimization)
        cache_info = {"size_mb": 0, "files_count": 0}

        if calculate_cache:
            from pathlib import Path
            experiments_dir = cls.get_experiments_dir()
            cache_size_mb = 0
            cache_files_count = 0

            # Check all Aux/experiment_{id}/ directories (only if needed)
            base_dir = Path(experiments_dir).parent.parent
            aux_pattern = f"**/Aux/experiment_{experiment.id}/pipeline_cache"

            for cache_dir in base_dir.glob(aux_pattern):
                if cache_dir.exists():
                    for cache_file in cache_dir.rglob("*.npy"):
                        cache_size_mb += cache_file.stat().st_size / (1024 * 1024)
                        cache_files_count += 1

            cache_info = {
                "size_mb": round(cache_size_mb, 2),
                "files_count": cache_files_count
            }

        return {
            "experiment_id": experiment.id,
            "filters": filters_summary,
            "transforms": transforms_summary,
            "total_steps": len(filters_summary) + len(transforms_summary),
            "cache_info": cache_info
        }

    @classmethod
    def clear_pipeline_cache(cls, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Deletes all cached pipeline results for a specific experiment or all experiments.

        Args:
            experiment_id: If provided, only clears cache for this experiment.
                          If None, clears cache for current experiment.

        Returns:
            dict with keys:
                - files_deleted: Number of files deleted
                - space_freed_mb: Space freed in megabytes
                - experiments_affected: List of experiment IDs whose cache was cleared
        """
        import shutil
        from pathlib import Path

        if experiment_id is None:
            experiment = cls._load_latest_experiment()
            experiment_id = experiment.id

        experiments_dir = cls.get_experiments_dir()
        base_dir = Path(experiments_dir).parent.parent

        files_deleted = 0
        space_freed = 0
        experiments_affected = []

        # Find all cache directories
        aux_pattern = f"**/Aux/experiment_{experiment_id}"

        for aux_dir in base_dir.glob(aux_pattern):
            if aux_dir.exists():
                cache_dir = aux_dir / "pipeline_cache"
                intermediates_dir = aux_dir / "intermediates"

                # Count and delete cache
                for cache_file in [cache_dir, intermediates_dir]:
                    if cache_file.exists():
                        for file in cache_file.rglob("*"):
                            if file.is_file():
                                space_freed += file.stat().st_size
                                files_deleted += 1

                        shutil.rmtree(str(cache_file))

                experiments_affected.append(experiment_id)

        space_freed_mb = space_freed / (1024 * 1024)

        print(f"🧹 Cache limpiado: {files_deleted} archivos, {space_freed_mb:.2f} MB liberados")

        return {
            "files_deleted": files_deleted,
            "space_freed_mb": round(space_freed_mb, 2),
            "experiments_affected": list(set(experiments_affected))
        }
