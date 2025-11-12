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
        description="Feature transformations applied to the data (legacy, deprecated)"
    )
    P300Classifier: Optional[dict] = Field(
        None,
        description="Classification model configuration for P300 detection. Structure: {model_name: {config, transform: {...}}}"
    )
    innerSpeachClassifier: Optional[dict] = Field(
        None,
        description="Classification model configuration for Inner Speech detection. Structure: {model_name: {config, transform: {...}}}"
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
        The transform field (if exists) will be preserved from previous configuration.
        """
        import numpy as np

        def convert_numpy_to_list(obj):
            """Recursivamente convierte numpy arrays a listas."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_to_list(item) for item in obj]
            else:
                return obj

        experiment = cls._load_latest_experiment()
        classifier_name = classifier.__class__.__name__

        # Usar model_dump (Pydantic v2) y luego convertir numpy arrays
        if isinstance(classifier, BaseModel):
            try:
                # Pydantic v2
                classifier_data = classifier.model_dump()
            except AttributeError:
                # Fallback Pydantic v1
                classifier_data = classifier.dict()

            # Convertir todos los numpy arrays a listas
            classifier_data = convert_numpy_to_list(classifier_data)
        else:
            classifier_data = vars(classifier)

        # Preservar transformación existente si hay una
        existing_transform = None
        if experiment.P300Classifier and isinstance(experiment.P300Classifier, dict):
            for model_name, model_config in experiment.P300Classifier.items():
                if isinstance(model_config, dict) and "transform" in model_config:
                    existing_transform = model_config["transform"]
                    break

        # Agregar transformación preservada al nuevo config
        if existing_transform is not None:
            classifier_data["transform"] = existing_transform

        experiment.P300Classifier = {
            classifier_name: classifier_data
        }
        cls._save_latest_experiment(experiment)

    @classmethod
    def set_P300_transform(cls, transform: Transform) -> None:
        """
        Sets or updates the transform configuration for the P300 classifier.
        Creates an empty classifier entry if it doesn't exist yet.
        """
        experiment = cls._load_latest_experiment()

        transform_name = transform.__class__.__name__
        transform_data = transform.dict() if isinstance(transform, BaseModel) else vars(transform)

        # Si no hay clasificador P300 aún, crear estructura básica
        if not experiment.P300Classifier or not isinstance(experiment.P300Classifier, dict):
            experiment.P300Classifier = {
                "pending_model": {
                    "transform": {transform_name: transform_data}
                }
            }
        else:
            # Actualizar transform en el modelo existente
            for model_name, model_config in experiment.P300Classifier.items():
                if isinstance(model_config, dict):
                    model_config["transform"] = {transform_name: transform_data}
                    break

        cls._save_latest_experiment(experiment)
        print(f"✅ Transformación '{transform_name}' configurada para P300")

    @classmethod
    def get_P300_transform(cls) -> Optional[dict]:
        """
        Retrieves the transform configuration from the P300 classifier.
        Returns None if no transform is configured.
        """
        try:
            experiment = cls._load_latest_experiment()
            if experiment.P300Classifier and isinstance(experiment.P300Classifier, dict):
                for model_name, model_config in experiment.P300Classifier.items():
                    if isinstance(model_config, dict) and "transform" in model_config:
                        return model_config["transform"]
            return None
        except Exception:
            return None

    @classmethod
    def add_inner_speech_classifier(cls, classifier: Classifier) -> None:
        """
        Stores the Inner Speech classifier configuration.
        The transform field (if exists) will be preserved from previous configuration.
        """
        import numpy as np

        def convert_numpy_to_list(obj):
            """Recursivamente convierte numpy arrays a listas."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_to_list(item) for item in obj]
            else:
                return obj

        experiment = cls._load_latest_experiment()
        classifier_name = classifier.__class__.__name__

        # Usar model_dump (Pydantic v2) y luego convertir numpy arrays
        if isinstance(classifier, BaseModel):
            try:
                # Pydantic v2
                classifier_data = classifier.model_dump()
            except AttributeError:
                # Fallback Pydantic v1
                classifier_data = classifier.dict()

            # Convertir todos los numpy arrays a listas
            classifier_data = convert_numpy_to_list(classifier_data)
        else:
            classifier_data = vars(classifier)

        # Preservar transformación existente si hay una
        existing_transform = None
        if experiment.innerSpeachClassifier and isinstance(experiment.innerSpeachClassifier, dict):
            for model_name, model_config in experiment.innerSpeachClassifier.items():
                if isinstance(model_config, dict) and "transform" in model_config:
                    existing_transform = model_config["transform"]
                    break

        # Agregar transformación preservada al nuevo config
        if existing_transform is not None:
            classifier_data["transform"] = existing_transform

        experiment.innerSpeachClassifier = {
            classifier_name: classifier_data
        }
        cls._save_latest_experiment(experiment)

    @classmethod
    def set_inner_speech_transform(cls, transform: Transform) -> None:
        """
        Sets or updates the transform configuration for the Inner Speech classifier.
        Creates an empty classifier entry if it doesn't exist yet.
        """
        experiment = cls._load_latest_experiment()

        transform_name = transform.__class__.__name__
        transform_data = transform.dict() if isinstance(transform, BaseModel) else vars(transform)

        # Si no hay clasificador Inner Speech aún, crear estructura básica
        if not experiment.innerSpeachClassifier or not isinstance(experiment.innerSpeachClassifier, dict):
            experiment.innerSpeachClassifier = {
                "pending_model": {
                    "transform": {transform_name: transform_data}
                }
            }
        else:
            # Actualizar transform en el modelo existente
            for model_name, model_config in experiment.innerSpeachClassifier.items():
                if isinstance(model_config, dict):
                    model_config["transform"] = {transform_name: transform_data}
                    break

        cls._save_latest_experiment(experiment)
        print(f"✅ Transformación '{transform_name}' configurada para Inner Speech")

    @classmethod
    def get_inner_speech_transform(cls) -> Optional[dict]:
        """
        Retrieves the transform configuration from the Inner Speech classifier.
        Returns None if no transform is configured.
        """
        try:
            experiment = cls._load_latest_experiment()
            if experiment.innerSpeachClassifier and isinstance(experiment.innerSpeachClassifier, dict):
                for model_name, model_config in experiment.innerSpeachClassifier.items():
                    if isinstance(model_config, dict) and "transform" in model_config:
                        return model_config["transform"]
            return None
        except Exception:
            return None

    # -------------------- Pipeline System for Models --------------------

    @classmethod
    def dapply_model_pipeline(
        cls,
        file_path: str,
        model_type: str = "p300",
        force_recalculate: bool = False,
        save_intermediates: bool = True,
        verbose: bool = True,
        *,
        flatten_output: bool = True
    ) -> Dict[str, Any]:
        """
        Applies the complete pipeline for a specific model type: filters → model's transform → ready for model.

        This is the NEW method that should be used for P300 and Inner Speech models.
        It replaces apply_history_pipeline() for model-specific workflows.

        Workflow:
        1. Apply all filters from experiment.filters
        2. Apply the specific transform from experiment.P300Classifier.transform or experiment.innerSpeachClassifier.transform
        3. Return processed signal ready for the model

        Args:
            file_path: Path to the event .npy file
            model_type: "p300" or "inner" to determine which model's transform to use
            force_recalculate: If True, ignores cache and recalculates
            save_intermediates: If True, saves intermediate results after each step
            verbose: If True, prints progress messages

                Returns:
                        dict with keys:
                                - signal: Final processed signal array. Si flatten_output=False y la transform genera 3D,
                                    se retorna el cubo 3D (n_frames, features, n_channels) SIN aplanar.
                                - metadata: Pipeline execution info
                                - cache_used: Boolean indicating if cache was used
                                - cache_path: Path to cached file
                                - labels_path: Path to labels file (if generated by transform)
        """
        import numpy as np
        import time
        import hashlib
        from pathlib import Path

        # Load experiment
        experiment = cls._load_latest_experiment()
        experiment_id = experiment.id

        if verbose:
            print(f"\n{'='*60}")
            print(f"🧪 PIPELINE DE MODELO: {model_type.upper()}")
            print(f"📂 Archivo: {Path(file_path).name}")
            print(f"{'='*60}")

        # Get the model's transform
        if model_type == "p300":
            model_transform_dict = cls.get_P300_transform()
            model_name = "P300"
        elif model_type == "inner":
            model_transform_dict = cls.get_inner_speech_transform()
            model_name = "Inner Speech"
        else:
            raise ValueError(f"model_type debe ser 'p300' o 'inner', recibido: {model_type}")

        if verbose:
            n_filters = len(experiment.filters)
            has_transform = model_transform_dict is not None
            print(f"📋 Pipeline: {n_filters} filtros + {'1 transformación' if has_transform else 'sin transformación'}")

            if n_filters > 0:
                print(f"\n🔧 Filtros a aplicar:")
                for i, filter_entry in enumerate(experiment.filters):
                    filter_id = filter_entry.get("id", "?")
                    filter_name = next((k for k, v in filter_entry.items() if k != "id" and isinstance(v, dict)), "Unknown")
                    print(f"  {i+1}. F{filter_id}: {filter_name}")

            if has_transform:
                transform_name = next(iter(model_transform_dict.keys()))
                print(f"\n🎨 Transformación del modelo {model_name}:")
                print(f"  - {transform_name}")

            print(f"{'='*60}\n")

        # Get cache paths (usando model_type en el hash para diferenciar)
        cache_paths = cls._get_model_pipeline_cache_path(file_path, experiment_id, model_type)
        cache_file = cache_paths["cache_file"]
        metadata_file = cache_paths["metadata_file"]
        cache_dir = Path(cache_paths["cache_dir"])
        intermediates_dir = Path(cache_paths["intermediates_dir"])

        # Create directories
        cache_dir.mkdir(parents=True, exist_ok=True)
        # IMPORTANTE: Crear intermediates_dir SIEMPRE porque las transformaciones lo necesitan para archivos temporales
        intermediates_dir.mkdir(parents=True, exist_ok=True)

        # Check cache validity
        cache_exists = os.path.exists(cache_file) and os.path.exists(metadata_file)

        if cache_exists and not force_recalculate:
            # Load metadata to verify pipeline hash
            with open(metadata_file, "r") as f:
                cached_metadata = json.load(f)

            # Hash actual config (filters + model transform)
            current_config = {
                "filters": experiment.filters,
                "model_transform": model_transform_dict,
                "model_type": model_type
            }
            current_hash = hashlib.md5(
                json.dumps(current_config, sort_keys=True).encode()
            ).hexdigest()

            if cached_metadata.get("pipeline_hash") == current_hash:
                # Valid cache found
                if verbose:
                    print(f"✅ Cache válido encontrado: {Path(cache_file).name}")

                signal = np.load(cache_file, allow_pickle=False)
                labels_path = cached_metadata.get("labels_file")

                return {
                    "signal": signal,
                    "metadata": cached_metadata,
                    "cache_used": True,
                    "cache_path": cache_file,
                    "labels_path": labels_path
                }

        # No valid cache - execute full pipeline
        if verbose:
            print(f"🔄 Ejecutando pipeline completo...")

        start_time = time.time()

        # Load original signal
        current_signal = np.load(file_path, allow_pickle=False)
        original_shape = current_signal.shape

        current_labels_file = None
        step_count = 0
        execution_log = []
        # Visualization payload (if a model transform outputs 3D, we'll keep a limited copy for UI)
        viz_payload = None

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

                    # Save CURRENT INPUT for this filter
                    # Mantener forma REAL (sin adaptación 3D→2D) para diagnóstico transparente
                    if verbose:
                        try:
                            print(f"    ↪ Entrada filtro: shape={current_signal.shape}, ndim={getattr(current_signal,'ndim', 'na')}")
                        except Exception:
                            pass
                    temp_input = intermediates_dir / f"temp_step_{step_count}_input.npy"
                    np.save(str(temp_input), current_signal)

                    # (Eliminado intento previo de adaptación; usar señal directa)

                    # Create temp output directory
                    temp_output_dir = intermediates_dir / f"temp_step_{step_count}_output"
                    temp_output_dir.mkdir(exist_ok=True)

                    # Apply filter
                    success = filter_class.apply(
                        filter_instance,
                        file_path=str(temp_input),
                        directory_path_out=str(temp_output_dir)
                    )

                    if success:
                        # Find the output file
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

        # Phase 2: Apply model's transform (if exists)
        if model_transform_dict:
            if verbose:
                print(f"📍 Fase 2: Aplicando transformación del modelo {model_name}")

            # Extract transform name and config
            transform_name = next(iter(model_transform_dict.keys()))
            transform_config = model_transform_dict[transform_name]

            try:
                # Reconstruct transform instance
                transform_instance = cls._reconstruct_transform_instance(transform_name, transform_config)

                if verbose:
                    print(f"  → Aplicando {transform_name}")

                # Get transform class for apply method
                from backend.classes.FeatureExtracture.TransformSchemaFactory import TransformSchemaFactory
                transform_class = TransformSchemaFactory.available_transforms[transform_name]

                # Save current signal to temp file
                temp_input = intermediates_dir / f"temp_step_{step_count}_input.npy"
                np.save(str(temp_input), current_signal)

                temp_output_dir = intermediates_dir / "temp_output"
                temp_output_dir.mkdir(exist_ok=True)

                # Create temp labels (nombre debe coincidir con temp_input para que transform lo encuentre)
                event_name = Path(file_path).stem
                event_class = event_name.split('[')[0].strip() if '[' in event_name else event_name

                temp_labels_dir = intermediates_dir / "temp_labels"
                temp_labels_dir.mkdir(exist_ok=True)

                # Determinar longitud de labels según dimensionalidad:
                # - Para señales 2D (channels, time): usar la dimensión mayor (normalmente tiempo)
                # - Para señales 3D ya venteadas: usar primera dim (frames)
                if isinstance(current_signal, np.ndarray):
                    if current_signal.ndim == 1:
                        n_times = int(current_signal.shape[0])
                    elif current_signal.ndim == 2:
                        # En EEG 2D, típicamente (n_channels, n_times) o (n_times, n_channels)
                        # Usar la dimensión mayor como tiempo
                        n_times = int(max(current_signal.shape))
                    elif current_signal.ndim == 3:
                        # Ya venteado: (n_frames, feature_len, n_channels)
                        n_times = int(current_signal.shape[0])
                    else:
                        n_times = int(current_signal.shape[0])
                else:
                    try:
                        n_times = int(len(current_signal))
                    except Exception:
                        n_times = 0

                labels_array = np.array([event_class] * n_times, dtype=str)
                # CRÍTICO: El nombre debe ser temp_input.name para que transform lo encuentre
                temp_labels_file = temp_labels_dir / temp_input.name
                np.save(str(temp_labels_file), labels_array)

                if verbose:
                    print(f"    🏷️  Labels creadas: clase='{event_class}', n={n_times}, archivo={temp_labels_file.name}")

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
                    output_files = sorted(
                        [f for f in temp_output_dir.glob("*.npy") if "_labels" not in f.name],
                        key=lambda x: x.stat().st_mtime,
                        reverse=True
                    )

                    # Find labels file
                    label_files = sorted(
                        [f for f in temp_output_dir.glob("*_labels.npy")],
                        key=lambda x: x.stat().st_mtime,
                        reverse=True
                    )

                    if output_files:
                        current_signal = np.load(str(output_files[0]), allow_pickle=False)

                        # Save labels path if exists
                        if label_files:
                            current_labels_file = label_files[0]

                        # Handle 3D arrays (windowed/features)
                        if current_signal.ndim == 3:
                            n_frames, feature_len, n_channels = current_signal.shape

                            # Determine domain based on transform name
                            domain = "unknown"
                            if transform_name == "FFTTransform":
                                domain = "time-freq"
                            elif transform_name == "DCTTransform":
                                domain = "coeffs"
                            elif transform_name in ("WaveletTransform", "WindowingTransform"):
                                domain = "window-time"

                            # Limit payload size for UI responsiveness
                            MAX_FRAMES = 128
                            MAX_FEATURES = 256
                            f_lim = min(feature_len, MAX_FEATURES)
                            t_lim = min(n_frames, MAX_FRAMES)
                            cube_limited = current_signal[:t_lim, :f_lim, :]

                            viz_payload = {
                                "domain": domain,
                                "shape": [int(n_frames), int(feature_len), int(n_channels)],
                                "cube": cube_limited.tolist(),  # (frames, feature_axis, channels)
                                "frames_shown": int(t_lim),
                                "features_shown": int(f_lim),
                                "transform_name": transform_name
                            }

                            # Flatten para consumo del modelo solo si así se solicita
                            if flatten_output:
                                current_signal = current_signal.transpose(2, 0, 1).reshape(n_channels, n_frames * feature_len)

                        # Save intermediate if requested
                        if save_intermediates:
                            intermediate_file = intermediates_dir / f"step_{step_count:02d}_{transform_name}_model.npy"
                            np.save(str(intermediate_file), current_signal)

                        execution_log.append({
                            "step": step_count,
                            "type": "model_transform",
                            "name": transform_name,
                            "model_type": model_type,
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
                            "type": "model_transform",
                            "name": transform_name,
                            "model_type": model_type,
                            "status": "no_output"
                        })
                else:
                    if verbose:
                        print(f"    ❌ {transform_name} falló")
                    execution_log.append({
                        "step": step_count,
                        "type": "model_transform",
                        "name": transform_name,
                        "model_type": model_type,
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

        # Save final result to cache
        np.save(cache_file, current_signal)

        # Save labels file to cache if generated
        labels_cache_file = None
        if current_labels_file and Path(current_labels_file).exists():
            labels_cache_file = str(Path(cache_file).parent / f"{Path(cache_file).stem}_labels.npy")
            import shutil
            shutil.copy2(current_labels_file, labels_cache_file)
            if verbose:
                print(f"💾 Labels guardadas en cache: {Path(labels_cache_file).name}")

        # Save metadata
        current_config = {
            "filters": experiment.filters,
            "model_transform": model_transform_dict,
            "model_type": model_type
        }
        pipeline_hash = hashlib.md5(
            json.dumps(current_config, sort_keys=True).encode()
        ).hexdigest()

        execution_time = time.time() - start_time

        metadata = {
            "pipeline_hash": pipeline_hash,
            "experiment_id": experiment_id,
            "model_type": model_type,
            "original_file": file_path,
            "original_shape": list(original_shape),
            "final_shape": list(current_signal.shape),
            "execution_time_seconds": execution_time,
            "steps_applied": step_count,
            "execution_log": execution_log,
            "timestamp": time.time(),
            "labels_file": labels_cache_file
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        if verbose:
            # Final summary
            successful_steps = [log for log in execution_log if log.get("status") == "success"]
            failed_steps = [log for log in execution_log if log.get("status") == "failed"]

            print(f"\n{'='*60}")
            print(f"📊 RESUMEN DEL PIPELINE {model_name.upper()}")
            print(f"{'='*60}")
            print(f"⏱️  Tiempo total: {execution_time:.2f}s")
            print(f"📏 Shape: {original_shape} → {current_signal.shape}")
            print(f"✅ Pasos exitosos: {len(successful_steps)}/{len(execution_log)}")

            if failed_steps:
                print(f"❌ Pasos fallidos: {len(failed_steps)}")
                for step in failed_steps:
                    print(f"   - {step['name']} (tipo: {step['type']})")

            print(f"💾 Cache guardado: {Path(cache_file).name}")
            print(f"{'='*60}\n")

        # Build result
        result = {
            "signal": current_signal,
            "metadata": metadata,
            "cache_used": False,
            "cache_path": cache_file,
            "labels_path": labels_cache_file
        }
        if viz_payload is not None:
            result["viz"] = viz_payload
            # Exponer nombre de transform actual también fuera del payload para fácil acceso en UI
            if isinstance(viz_payload, dict) and "transform_name" in viz_payload:
                result["applied_transform_name"] = viz_payload["transform_name"]
        
        return result

    @classmethod
    def _get_model_pipeline_cache_path(cls, file_path: str, experiment_id: str, model_type: str) -> Dict[str, str]:
        """
        Constructs cache file paths for a model-specific pipeline.
        Includes model_type in the path to differentiate P300 vs Inner Speech caches.

        Returns:
            dict with keys: cache_file, metadata_file, cache_dir, intermediates_dir
        """
        import hashlib
        from pathlib import Path

        event_path = Path(file_path)

        # Si el path contiene "Aux/", navegar hasta encontrar el directorio base correcto
        # Ejemplo: Aux/inner_speech/sub-01/.../Events/archivo.npy -> usar Aux/inner_speech/... como base
        # Ejemplo: Data/inner_speech/sub-01/.../Events/archivo.npy -> usar Data/inner_speech/... como base
        parts = event_path.parts

        # Buscar "Events" en el path y usar su padre como dataset_dir
        if "Events" in parts:
            events_index = parts.index("Events")
            dataset_dir = Path(*parts[:events_index + 1])
        else:
            dataset_dir = event_path.parent

        # Create Aux/experiment_{id}/model_{type}/ structure
        aux_dir = dataset_dir / "Aux" / f"experiment_{experiment_id}" / f"model_{model_type}"
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

    # -------------------- Pipeline History System (Legacy) --------------------

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

        # Si el path contiene "Events", usar ese directorio como base
        parts = event_path.parts
        if "Events" in parts:
            events_index = parts.index("Events")
            dataset_dir = Path(*parts[:events_index + 1])
        else:
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
        verbose: bool = True,
        model_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Aplica el pipeline histórico bajo el contrato: n filtros + 1 transformación.
        Ver histórico completo solo como referencia; no se ejecutan múltiples transformaciones.
        """
        import numpy as np, time, hashlib, os, json
        from pathlib import Path

        experiment = cls._load_latest_experiment()
        experiment_id = experiment.id

        if verbose:
            print(f"\n{'='*60}")
            print(f"🧪 EXPERIMENTO #{experiment_id}")
            print(f"📂 Archivo: {Path(file_path).name}")
            print(f"{'='*60}")
            print(f"📋 Historial: {len(experiment.filters)} filtros + {len(experiment.transform)} transformadas")
            if experiment.filters:
                print("\n🔧 Filtros:")
                for i, fentry in enumerate(experiment.filters):
                    fid = fentry.get('id','?')
                    fname = next((k for k,v in fentry.items() if k!='id' and isinstance(v,dict)), '?')
                    print(f"  {i+1}. F{fid}: {fname}")
            if experiment.transform:
                print("\n🎨 Transformaciones (referencia):")
                for i, tentry in enumerate(experiment.transform):
                    tid = tentry.get('id','?')
                    tname = next((k for k,v in tentry.items() if k not in ['id','dimensionality_change'] and isinstance(v,dict)), '?')
                    print(f"  {i+1}. T{tid}: {tname}")
            print(f"{'='*60}\n")

        paths = cls._get_pipeline_cache_path(file_path, experiment_id)
        cache_file = paths['cache_file']
        metadata_file = paths['metadata_file']
        cache_dir = Path(paths['cache_dir'])
        intermediates_dir = Path(paths['intermediates_dir'])
        cache_dir.mkdir(parents=True, exist_ok=True)
        # IMPORTANTE: Crear intermediates_dir SIEMPRE porque las transformaciones lo necesitan para archivos temporales
        intermediates_dir.mkdir(parents=True, exist_ok=True)

        cache_exists = os.path.exists(cache_file) and os.path.exists(metadata_file)
        if cache_exists and not force_recalculate:
            with open(metadata_file,'r') as f:
                cached_meta = json.load(f)
            current_cfg = {"filters": experiment.filters, "transforms": experiment.transform, "model_type": model_type}
            current_hash = hashlib.md5(json.dumps(current_cfg, sort_keys=True).encode()).hexdigest()
            if cached_meta.get('pipeline_hash') == current_hash:
                if verbose:
                    print(f"✅ Cache válido: {Path(cache_file).name}")
                return {"signal": np.load(cache_file, allow_pickle=False), "metadata": cached_meta, "cache_used": True, "cache_path": cache_file, "labels_path": cached_meta.get('labels_file')}

        if verbose:
            print("🔄 Ejecutando pipeline histórico (n filtros + 1 transformación)...")

        start_time = time.time()
        current_signal = np.load(file_path, allow_pickle=False)
        original_shape = current_signal.shape
        current_labels_file = None
        execution_log: List[Dict[str,Any]] = []
        step_count = 0

        # Fase 1: filtros
        if experiment.filters:
            if verbose:
                print(f"📍 Fase 1: {len(experiment.filters)} filtros")
            for fentry in experiment.filters:
                fid = fentry.get('id')
                fname = None; fcfg=None
                for k,v in fentry.items():
                    if k!='id' and isinstance(v,dict):
                        fname=k; fcfg=v; break
                if not fname or not fcfg:
                    if verbose: print(f"⚠️ Filtro {fid} inválido, se omite")
                    continue
                try:
                    finst = cls._reconstruct_filter_instance(fname, fcfg)
                    from backend.classes.Filter.FilterSchemaFactory import FilterSchemaFactory
                    fclass = FilterSchemaFactory.available_filters[fname]
                    if verbose:
                        print(f"  → {fname} (ID:{fid}) entrada shape={current_signal.shape}")
                    tmp_in = intermediates_dir / f"temp_step_{step_count}_input.npy"; np.save(str(tmp_in), current_signal)
                    tmp_out_dir = intermediates_dir / f"temp_step_{step_count}_output"; tmp_out_dir.mkdir(exist_ok=True)
                    ok = fclass.apply(finst, file_path=str(tmp_in), directory_path_out=str(tmp_out_dir))
                    if ok:
                        outs = list(tmp_out_dir.glob('*.npy'))
                        if outs:
                            current_signal = np.load(str(outs[0]), allow_pickle=False)
                            if save_intermediates:
                                np.save(str(intermediates_dir / f"step_{step_count:02d}_{fname}_{fid}.npy"), current_signal)
                            execution_log.append({"step": step_count, "type":"filter", "id": fid, "name": fname, "shape": list(current_signal.shape), "status":"success"})
                            if verbose: print(f"    ✅ {fname} aplicado: {current_signal.shape}")
                            step_count += 1
                        else:
                            execution_log.append({"step": step_count, "type":"filter", "id": fid, "name": fname, "status":"no_output"})
                            if verbose: print(f"    ⚠️ {fname} sin salida")
                    else:
                        execution_log.append({"step": step_count, "type":"filter", "id": fid, "name": fname, "status":"failed"})
                        if verbose: print(f"    ❌ {fname} falló")
                except Exception as e:
                    execution_log.append({"step": step_count, "type":"filter", "id": fid, "name": fname or 'unknown', "status":"error", "error": str(e)})
                    if verbose: print(f"    ❌ Error en filtro {fname}: {e}")
                finally:
                    # cleanup
                    try:
                        if 'tmp_in' in locals() and tmp_in.exists(): tmp_in.unlink()
                        if 'tmp_out_dir' in locals() and tmp_out_dir.exists():
                            import shutil; shutil.rmtree(str(tmp_out_dir))
                    except Exception: pass

        # Fase 2: elegir UNA transform
        sel_name=None; sel_cfg=None; sel_source=None
        if model_type == 'p300':
            mt = cls.get_P300_transform();
            if mt:
                sel_name = next(iter(mt.keys())); sel_cfg = mt[sel_name]; sel_source='P300Classifier'
        elif model_type == 'inner':
            mt = cls.get_inner_speech_transform();
            if mt:
                sel_name = next(iter(mt.keys())); sel_cfg = mt[sel_name]; sel_source='InnerSpeechClassifier'
        if sel_name is None and experiment.transform:
            last = experiment.transform[-1]
            for k,v in last.items():
                if k not in ['id','dimensionality_change'] and isinstance(v,dict):
                    sel_name=k; sel_cfg=v; sel_source='historial:last'; break

        if sel_name:
            if verbose:
                extra = " (múltiples en historial, se aplica solo 1)" if len(experiment.transform)>1 else ""
                print(f"📍 Fase 2: transform {sel_name} fuente={sel_source}{extra}")
            try:
                tinst = cls._reconstruct_transform_instance(sel_name, sel_cfg)
                from backend.classes.FeatureExtracture.TransformSchemaFactory import TransformSchemaFactory
                tclass = TransformSchemaFactory.available_transforms[sel_name]
                tmp_in = intermediates_dir / f"temp_step_{step_count}_input.npy"; np.save(str(tmp_in), current_signal)
                tmp_out_dir = intermediates_dir / "temp_output"; tmp_out_dir.mkdir(exist_ok=True)
                # Labels temporales
                ev_name = Path(file_path).stem
                ev_class = ev_name.split('[')[0].strip() if '[' in ev_name else ev_name
                lbl_dir = intermediates_dir / "temp_labels"; lbl_dir.mkdir(exist_ok=True)
                if isinstance(current_signal, np.ndarray):
                    if current_signal.ndim == 1: n_times = current_signal.shape[0]
                    elif current_signal.ndim == 2: n_times = max(current_signal.shape)
                    elif current_signal.ndim == 3: n_times = current_signal.shape[0]
                    else: n_times = current_signal.shape[0]
                else:
                    try: n_times = len(current_signal)
                    except Exception: n_times = 0
                # IMPORTANTE: Guardar labels como STRINGS para que las transformaciones las propaguen
                # Estas se convertirán a enteros más adelante en el flujo de entrenamiento
                np.save(str(lbl_dir / tmp_in.name), np.array([ev_class]*n_times,dtype=str))
                try:
                    ok = tclass.apply(tinst, file_path_in=str(tmp_in), directory_path_out=str(tmp_out_dir), labels_directory=str(lbl_dir), labels_out_path=str(tmp_out_dir))
                except TypeError:
                    ok = tclass.apply(tinst, file_path_in=str(tmp_in), directory_path_out=str(tmp_out_dir), labels_directory=str(lbl_dir), dir_out_labels=str(tmp_out_dir))
                if ok:
                    outs = sorted([f for f in tmp_out_dir.glob('*.npy') if '_labels' not in f.name], key=lambda x: x.stat().st_mtime, reverse=True)
                    lbls = sorted(list(tmp_out_dir.glob('*_labels.npy')), key=lambda x: x.stat().st_mtime, reverse=True)
                    if outs:
                        current_signal = np.load(str(outs[0]), allow_pickle=False)

                        # IMPORTANTE: Copiar labels a intermediates ANTES de borrar tmp_out_dir
                        if lbls:
                            labels_temp_copy = intermediates_dir / f"step_{step_count:02d}_{sel_name}_labels.npy"
                            import shutil
                            shutil.copy2(str(lbls[0]), str(labels_temp_copy))
                            current_labels_file = labels_temp_copy
                            if verbose: print(f"    📋 Labels copiadas: {labels_temp_copy.name}")

                        if save_intermediates:
                            np.save(str(intermediates_dir / f"step_{step_count:02d}_{sel_name}.npy"), current_signal)
                        execution_log.append({"step": step_count, "type":"transform", "name": sel_name, "shape": list(current_signal.shape), "status":"success"})
                        if verbose: print(f"    ✅ {sel_name} aplicada: {current_signal.shape}")
                        step_count += 1
                    else:
                        execution_log.append({"step": step_count, "type":"transform", "name": sel_name, "status":"no_output"})
                        if verbose: print(f"    ⚠️ {sel_name} sin salida")
                else:
                    execution_log.append({"step": step_count, "type":"transform", "name": sel_name, "status":"failed"})
                    if verbose: print(f"    ❌ {sel_name} falló")
            except Exception as e:
                execution_log.append({"step": step_count, "type":"transform", "name": sel_name or 'unknown', "status":"error", "error": str(e)})
                if verbose: print(f"    ❌ Error transform {sel_name}: {e}")
            finally:
                try:
                    if 'tmp_in' in locals() and tmp_in.exists(): tmp_in.unlink()
                    if 'tmp_out_dir' in locals() and tmp_out_dir.exists():
                        import shutil; shutil.rmtree(str(tmp_out_dir))
                except Exception: pass

        # Guardar señal final en caché
        np.save(cache_file, current_signal)

        # Labels -> validar y guardar
        labels_cache_file = None
        if current_labels_file and Path(current_labels_file).exists():
            labels_cache_file = str(Path(cache_file).parent / f"{Path(cache_file).stem}_labels.npy")
            labels_data = np.load(str(current_labels_file), allow_pickle=True)
            if current_signal.ndim == 3:
                n_frames = current_signal.shape[0]; n_labels = len(labels_data)
                if n_labels != n_frames:
                    if verbose: print(f"  ⚠️ Labels {n_labels} != frames {n_frames}")
                    if n_labels == 1:
                        labels_data = np.array([labels_data[0]]*n_frames, dtype=labels_data.dtype)
                        if verbose: print("  🔧 Expandida 1→frames")
                    elif n_labels < n_frames:
                        labels_data = np.concatenate([labels_data, np.array([labels_data[-1]]*(n_frames-n_labels), dtype=labels_data.dtype)])
                    else:
                        labels_data = labels_data[:n_frames]
            np.save(labels_cache_file, labels_data)
            if verbose: print(f"💾 Labels cache: {Path(labels_cache_file).name} ({len(labels_data)})")

        meta_cfg = {"filters": experiment.filters, "transforms": experiment.transform, "model_type": model_type}
        pipeline_hash = hashlib.md5(json.dumps(meta_cfg, sort_keys=True).encode()).hexdigest()
        exec_time = time.time() - start_time
        metadata = {
            "pipeline_hash": pipeline_hash,
            "experiment_id": experiment_id,
            "original_file": file_path,
            "original_shape": list(original_shape),
            "final_shape": list(current_signal.shape),
            "execution_time_seconds": exec_time,
            "steps_applied": step_count,
            "execution_log": execution_log,
            "timestamp": time.time(),
            "labels_file": labels_cache_file,
            "selected_transform": sel_name,
            "selected_transform_source": sel_source
        }
        with open(metadata_file,'w') as f: json.dump(metadata, f, indent=2)

        if verbose:
            succ=[l for l in execution_log if l.get('status')=='success']
            fail=[l for l in execution_log if l.get('status') in ('failed','error')]
            noop=[l for l in execution_log if l.get('status')=='no_output']
            print(f"\n{'='*60}\n📊 RESUMEN HISTÓRICO\n{'='*60}")
            print(f"⏱️ Tiempo: {exec_time:.2f}s")
            print(f"📏 Shape: {original_shape} → {current_signal.shape}")
            print(f"✅ Éxito: {len(succ)}/{len(execution_log)}")
            if fail:
                print(f"❌ Fallos: {len(fail)}")
                for st in fail: print(f"   - {st['type']} {st['name']} ({st['status']})")
            if noop:
                print(f"⚠️ Sin salida: {len(noop)}")
                for st in noop: print(f"   - {st['type']} {st['name']}")
            print(f"Transform seleccionada: {sel_name} (fuente: {sel_source})")
            print(f"💾 Cache: {Path(cache_file).name}\n{'='*60}\n")

        return {"signal": current_signal, "metadata": metadata, "cache_used": False, "cache_path": cache_file, "labels_path": labels_cache_file}

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

        print(f"Cache limpiado: {files_deleted} archivos, {space_freed_mb:.2f} MB liberados")

        return {
            "files_deleted": files_deleted,
            "space_freed_mb": round(space_freed_mb, 2),
            "experiments_affected": list(set(experiments_affected))
        }

    # -------------------- Mini Dataset Generation for Model Testing --------------------

    @classmethod
    def generate_model_dataset(
        cls,
        dataset_path: str,
        model_type: str = "p300",
        n_train: int = 10,
        n_test: int = 5,
        selected_classes: Optional[List[str]] = None,
        force_recalculate: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Genera un mini dataset de prueba aplicando el pipeline completo de un modelo específico.

        NUEVO MÉTODO que reemplaza generate_pipeline_dataset() para flujos de P300 e Inner Speech.

        Pipeline: filters → model's transform → ready for model

        Este método es utilizado para verificar que la configuración del modelo es correcta
        antes del entrenamiento completo.

        CACHÉ: El mini-dataset se genera UNA SOLA VEZ por experimento/modelo y se reutiliza
        en todas las verificaciones, a menos que el pipeline cambie.

        Args:
            dataset_path: Ruta al dataset (ej: "Data/nieto_inner_speech")
            model_type: "p300" o "inner" para determinar qué transformación usar
            n_train: Número de ejemplos para entrenamiento
            n_test: Número de ejemplos para prueba
            selected_classes: Lista de clases a usar. Si es None, usa todas las disponibles
            force_recalculate: Si True, recalcula el pipeline aunque exista caché
            verbose: Si True, imprime progreso

        Returns:
            dict con keys:
                - train_data: Lista de arrays procesados (train)
                - train_labels: Lista de etiquetas numéricas (train)
                - test_data: Lista de arrays procesados (test)
                - test_labels: Lista de etiquetas numéricas (test)
                - class_mapping: Dict {class_name: class_id}
                - n_train: Número de ejemplos de entrenamiento
                - n_test: Número de ejemplos de prueba
                - pipeline_summary: Resumen del pipeline aplicado
                - model_type: Tipo de modelo usado

        Example:
            >>> dataset = Experiment.generate_model_dataset(
            ...     dataset_path="Data/p300_dataset",
            ...     model_type="p300",
            ...     n_train=20,
            ...     n_test=10,
            ...     selected_classes=["target", "non-target"]
            ... )
            >>> print(f"Train: {len(dataset['train_data'])} | Test: {len(dataset['test_data'])}")
        """
        import numpy as np
        from pathlib import Path
        from collections import defaultdict
        import hashlib

        # Load experiment
        experiment = cls._load_latest_experiment()
        experiment_id = experiment.id

        # Get model's transform
        if model_type == "p300":
            model_transform = cls.get_P300_transform()
        elif model_type == "inner":
            model_transform = cls.get_inner_speech_transform()
        else:
            raise ValueError(f"model_type debe ser 'p300' o 'inner', recibido: {model_type}")

        # Normalizar dataset_path
        dataset_path_normalized = dataset_path
        if not dataset_path.startswith("Data/") and not dataset_path.startswith("/"):
            dataset_path_normalized = f"Data/{dataset_path}"

        if verbose:
            print(f"\n{'='*70}")
            print(f"🔬 GENERANDO MINI DATASET DE PRUEBA - MODELO {model_type.upper()}")
            print(f"{'='*70}")
            print(f"📂 Dataset: {dataset_path_normalized}")
            print(f"📊 Train: {n_train} | Test: {n_test}")
            print(f"🎨 Transformación: {list(model_transform.keys())[0] if model_transform else 'Ninguna'}")
            print(f"{'='*70}\n")

        # Get all event files
        dataset_root = Path(dataset_path_normalized)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_root}")

        event_files = list(dataset_root.glob("*.npy"))
        if not event_files:
            raise ValueError(f"No se encontraron archivos .npy en {dataset_root}")

        # Group files by class
        files_by_class = defaultdict(list)
        for event_file in event_files:
            # Extract class from filename (before '[' if exists)
            event_name = event_file.stem
            event_class = event_name.split('[')[0].strip() if '[' in event_name else event_name
            files_by_class[event_class].append(event_file)

        # Filter classes if specified
        if selected_classes:
            files_by_class = {
                cls_name: files
                for cls_name, files in files_by_class.items()
                if cls_name in selected_classes
            }

        if not files_by_class:
            raise ValueError(f"No se encontraron clases válidas. Clases disponibles: {list(files_by_class.keys())}")

        # Create class mapping
        class_mapping = {cls_name: idx for idx, cls_name in enumerate(sorted(files_by_class.keys()))}

        if verbose:
            print(f"📋 Clases encontradas: {len(class_mapping)}")
            for cls_name, cls_id in class_mapping.items():
                print(f"  {cls_id}: {cls_name} ({len(files_by_class[cls_name])} ejemplos)")
            print()

        # Select train/test samples for each class
        train_files = []
        train_labels = []
        test_files = []
        test_labels = []

        for cls_name, files in files_by_class.items():
            cls_id = class_mapping[cls_name]

            # Shuffle and split
            import random
            random.shuffle(files)

            n_train_cls = min(n_train, len(files))
            n_test_cls = min(n_test, len(files) - n_train_cls)

            train_files.extend(files[:n_train_cls])
            train_labels.extend([cls_id] * n_train_cls)

            test_files.extend(files[n_train_cls:n_train_cls + n_test_cls])
            test_labels.extend([cls_id] * n_test_cls)

        # Apply pipeline to all files
        train_data = []
        train_labels_expanded = []  # Etiquetas expandidas según frames
        test_data = []
        test_labels_expanded = []

        if verbose:
            print(f"🔄 Aplicando pipeline a {len(train_files)} archivos de train...")

        for idx, file_path in enumerate(train_files):
            try:
                result = cls.apply_model_pipeline(
                    file_path=str(file_path),
                    model_type=model_type,
                    force_recalculate=force_recalculate,
                    save_intermediates=False,
                    verbose=False
                )
                signal = result["signal"]
                train_data.append(signal)

                # Replicar etiqueta según número de frames (eje 0)
                if signal.ndim == 3:
                    n_frames = signal.shape[0]
                    train_labels_expanded.extend([train_labels[idx]] * n_frames)
                else:
                    train_labels_expanded.append(train_labels[idx])

            except Exception as e:
                if verbose:
                    print(f"  ⚠️ Error procesando {file_path.name}: {e}")
                continue

        if verbose:
            print(f"🔄 Aplicando pipeline a {len(test_files)} archivos de test...")

        for idx, file_path in enumerate(test_files):
            try:
                result = cls.apply_model_pipeline(
                    file_path=str(file_path),
                    model_type=model_type,
                    force_recalculate=force_recalculate,
                    save_intermediates=False,
                    verbose=False
                )
                signal = result["signal"]
                test_data.append(signal)

                # Replicar etiqueta según número de frames (eje 0)
                if signal.ndim == 3:
                    n_frames = signal.shape[0]
                    test_labels_expanded.extend([test_labels[idx]] * n_frames)
                else:
                    test_labels_expanded.append(test_labels[idx])

            except Exception as e:
                if verbose:
                    print(f"  ⚠️ Error procesando {file_path.name}: {e}")
                continue

        # Usar etiquetas expandidas
        train_labels = train_labels_expanded
        test_labels = test_labels_expanded

        # Create pipeline summary
        pipeline_summary = {
            "filters": len(experiment.filters),
            "transform": list(model_transform.keys())[0] if model_transform else None,
            "model_type": model_type
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ MINI DATASET GENERADO")
            print(f"{'='*70}")
            print(f"📊 Train: {len(train_data)} ejemplos")
            print(f"📊 Test: {len(test_data)} ejemplos")
            print(f"🔧 Pipeline: {pipeline_summary['filters']} filtros + {pipeline_summary['transform'] or 'sin transform'}")
            print(f"{'='*70}\n")

        return {
            "train_data": train_data,
            "train_labels": train_labels,
            "test_data": test_data,
            "test_labels": test_labels,
            "class_mapping": class_mapping,
            "n_train": len(train_data),
            "n_test": len(test_data),
            "pipeline_summary": pipeline_summary,
            "model_type": model_type
        }

    @classmethod
    def generate_pipeline_dataset(
        cls,
        dataset_path: str,
        n_train: int = 10,
        n_test: int = 5,
        selected_classes: Optional[List[str]] = None,
        force_recalculate: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        [LEGACY - DEPRECATED] Usa generate_model_dataset() en su lugar.

        Genera un mini dataset de prueba aplicando el pipeline completo del experimento
        a eventos reales del dataset.

        Este método es utilizado para verificar que la configuración del modelo es correcta
        antes del entrenamiento completo.

        CACHÉ: El mini-dataset se genera UNA SOLA VEZ por experimento y se reutiliza
        en todas las verificaciones de modelos, a menos que el pipeline cambie.

        Args:
            dataset_path: Ruta al dataset (ej: "Data/nieto_inner_speech")
            n_train: Número de ejemplos para entrenamiento
            n_test: Número de ejemplos para prueba
            selected_classes: Lista de clases a usar. Si es None, usa todas las disponibles
            force_recalculate: Si True, recalcula el pipeline aunque exista caché
            verbose: Si True, imprime progreso

        Returns:
            dict con keys:
                - train_data: Lista de rutas a archivos .npy procesados (train)
                - train_labels: Lista de etiquetas numéricas (train)
                - test_data: Lista de rutas a archivos .npy procesados (test)
                - test_labels: Lista de etiquetas numéricas (test)
                - class_mapping: Dict {class_name: class_id}
                - n_train: Número de ejemplos de entrenamiento
                - n_test: Número de ejemplos de prueba
                - pipeline_summary: Resumen del pipeline aplicado

        Example:
            >>> dataset = Experiment.generate_pipeline_dataset(
            ...     dataset_path="Data/nieto_inner_speech",
            ...     n_train=20,
            ...     n_test=10,
            ...     selected_classes=["arriba", "abajo"]
            ... )
            >>> print(f"Train: {len(dataset['train_data'])} | Test: {len(dataset['test_data'])}")
        """
        import numpy as np
        from pathlib import Path
        from collections import defaultdict
        import hashlib

        # Load experiment
        experiment = cls._load_latest_experiment()
        experiment_id = experiment.id

        # Calcular hash del pipeline actual
        current_config = {
            "filters": experiment.filters,
            "transforms": experiment.transform,
            "dataset_path": dataset_path,
            "n_train": n_train,
            "n_test": n_test,
            "selected_classes": selected_classes
        }
        current_hash = hashlib.md5(
            json.dumps(current_config, sort_keys=True).encode()
        ).hexdigest()

        # Normalizar dataset_path: si no empieza con "Data/" o "/", agregarlo
        dataset_path_normalized = dataset_path
        if not dataset_path.startswith("Data/") and not dataset_path.startswith("/"):
            dataset_path_normalized = f"Data/{dataset_path}"

        if verbose:
            print(f"\n{'='*70}")
            print(f"🔬 GENERANDO MINI DATASET DE PRUEBA")
            print(f"{'='*70}")
            print(f"📂 Dataset: {dataset_path_normalized}")
            print(f"📊 Train: {n_train} | Test: {n_test}")

        # Load dataset module
        from backend.classes.dataset import Dataset

        # Get all available event files
        events_result = Dataset.get_events_by_class(dataset_path_normalized, class_name=None)

        if events_result["status"] != 200:
            raise FileNotFoundError(
                f"No se encontraron eventos en {dataset_path_normalized}. "
                f"Asegúrate de que el dataset esté procesado. Error: {events_result['message']}"
            )

        all_event_files = events_result["event_files"]

        if not all_event_files:
            raise ValueError(f"No se encontraron archivos de eventos en {dataset_path_normalized}")

        if verbose:
            print(f"📁 Encontrados {len(all_event_files)} eventos totales")

        # ===== CONSTRUIR RUTA DE CACHÉ USANDO LA ESTRUCTURA DE EVENTOS =====
        # Usar el primer evento para determinar dónde guardar el metadata del mini-dataset
        # Siguiendo la misma lógica que _get_pipeline_cache_path:
        # Events/Aux/experiment_{id}/mini_dataset/metadata.json
        first_event_path = Path(all_event_files[0])
        events_dir = first_event_path.parent  # Directorio Events
        mini_dataset_dir = events_dir / "Aux" / f"experiment_{experiment_id}" / "mini_dataset"
        mini_dataset_dir.mkdir(parents=True, exist_ok=True)
        mini_dataset_metadata_file = mini_dataset_dir / "metadata.json"

        # ===== VERIFICAR CACHÉ DEL MINI-DATASET =====
        if mini_dataset_metadata_file.exists() and not force_recalculate:
            try:
                with open(mini_dataset_metadata_file, 'r') as f:
                    cached_metadata = json.load(f)

                if cached_metadata.get("pipeline_hash") == current_hash:
                    # Verificar que los archivos procesados aún existen (datos + labels)
                    all_files_exist = True
                    all_files = (
                        cached_metadata["train_data"] +
                        cached_metadata["test_data"] +
                        cached_metadata.get("train_labels", []) +
                        cached_metadata.get("test_labels", [])
                    )
                    for file_path in all_files:
                        if not Path(file_path).exists():
                            all_files_exist = False
                            break

                    if all_files_exist:
                        if verbose:
                            print(f"\n{'='*70}")
                            print(f"✅ MINI DATASET EN CACHÉ (Experimento #{experiment_id})")
                            print(f"{'='*70}")
                            print(f"📂 Train: {len(cached_metadata['train_data'])} ejemplos")
                            print(f"📂 Test: {len(cached_metadata['test_data'])} ejemplos")
                            print(f"🔧 Pipeline sin cambios desde última generación")
                            print(f"💾 Caché: {mini_dataset_dir}")
                            print(f"{'='*70}\n")

                        return cached_metadata

                    if verbose:
                        print(f"⚠️ [CACHE] Archivos procesados no encontrados, regenerando...")

                else:
                    if verbose:
                        print(f"⚠️ [CACHE] Pipeline cambió, regenerando mini-dataset...")

            except Exception as e:
                if verbose:
                    print(f"⚠️ [CACHE] Error leyendo caché: {e}, regenerando...")

        # ===== GENERAR NUEVO MINI-DATASET =====
        if verbose:
            print(f"\n🔬 Generando nuevo mini-dataset...")

        # Organize events by class
        events_by_class = defaultdict(list)
        for event_file in all_event_files:
            filename = Path(event_file).name
            # Extract class from filename: "clase[inicio]{fin}.npy"
            class_name = filename.split('[')[0].strip()
            events_by_class[class_name].append(event_file)

        available_classes = list(events_by_class.keys())

        if verbose:
            print(f"🏷️  Clases encontradas: {available_classes}")
            for cls, evts in events_by_class.items():
                print(f"   - {cls}: {len(evts)} eventos")

        # Filter by selected classes if specified
        if selected_classes:
            events_by_class = {
                cls: evts for cls, evts in events_by_class.items()
                if cls in selected_classes
            }
            if not events_by_class:
                raise ValueError(
                    f"Las clases seleccionadas {selected_classes} no se encontraron en el dataset. "
                    f"Clases disponibles: {available_classes}"
                )
            if verbose:
                print(f"🎯 Clases seleccionadas: {list(events_by_class.keys())}")

        # Create class mapping: {class_name: class_id}
        class_names_sorted = sorted(events_by_class.keys())
        class_mapping = {name: idx for idx, name in enumerate(class_names_sorted)}

        if verbose:
            print(f"🔢 Mapeo de clases: {class_mapping}")

        # Calculate samples per class for balanced distribution
        n_classes = len(events_by_class)
        samples_per_class_train = n_train // n_classes
        samples_per_class_test = n_test // n_classes
        extra_train = n_train % n_classes
        extra_test = n_test % n_classes

        if verbose:
            print(f"\n📊 Distribución balanceada:")
            print(f"   Train: ~{samples_per_class_train} por clase (+{extra_train} extra)")
            print(f"   Test:  ~{samples_per_class_test} por clase (+{extra_test} extra)")

        # Collect train and test events
        train_events = []
        train_labels = []
        test_events = []
        test_labels = []

        # Use fixed seed for reproducibility (same mini-dataset across calls)
        np.random.seed(42)

        for idx, (class_name, event_files) in enumerate(sorted(events_by_class.items())):
            class_id = class_mapping[class_name]

            # Calculate samples for this class
            n_train_class = samples_per_class_train + (1 if idx < extra_train else 0)
            n_test_class = samples_per_class_test + (1 if idx < extra_test else 0)
            total_needed = n_train_class + n_test_class

            if len(event_files) < total_needed:
                if verbose:
                    print(
                        f"⚠️  Clase '{class_name}': solo {len(event_files)} eventos disponibles, "
                        f"necesarios {total_needed}. Usando todos los disponibles."
                    )
                n_train_class = min(n_train_class, len(event_files))
                n_test_class = max(0, len(event_files) - n_train_class)

            # Shuffle and split (seed already set for reproducibility)
            np.random.shuffle(event_files)

            train_files = event_files[:n_train_class]
            test_files = event_files[n_train_class:n_train_class + n_test_class]

            train_events.extend(train_files)
            train_labels.extend([class_id] * len(train_files))

            test_events.extend(test_files)
            test_labels.extend([class_id] * len(test_files))

            if verbose:
                print(f"   ✓ {class_name}: {len(train_files)} train + {len(test_files)} test")

        if verbose:
            print(f"\n🔄 Aplicando pipeline del experimento a {len(train_events) + len(test_events)} eventos...")

        # Apply pipeline to all events
        processed_train = []
        processed_test = []

        # Determinar model_type basándonos en el clasificador configurado
        model_type_for_pipeline = None
        if experiment.P300Classifier:
            model_type_for_pipeline = "p300"
        elif experiment.innerSpeachClassifier:
            model_type_for_pipeline = "inner"

        def process_events(event_list, label_list, split_name):
            processed = []
            for i, (event_path, label) in enumerate(zip(event_list, label_list)):
                try:
                    if verbose and i % 5 == 0:
                        print(f"   [{split_name}] Procesando {i+1}/{len(event_list)}...", end='\r')

                    # Apply full pipeline to this event
                    # IMPORTANTE: Usar Experiment explícitamente porque cls no se captura en nested functions
                    result = Experiment.apply_history_pipeline(
                        file_path=event_path,
                        force_recalculate=force_recalculate,
                        save_intermediates=False,
                        verbose=False,
                        model_type=model_type_for_pipeline  # ← Pasar model_type para aplicar transform correcta
                    )

                    processed.append({
                        "signal": result["signal"],
                        "label": label,
                        "original_path": event_path,
                        "cache_path": result["cache_path"],
                        "labels_path": result.get("labels_path")  # Path a labels generado por transformación
                    })

                except Exception as e:
                    if verbose:
                        print(f"\n   ⚠️ Error procesando {Path(event_path).name}: {e}")
                    continue

            if verbose:
                print(f"   [{split_name}] Completado: {len(processed)}/{len(event_list)} eventos procesados")

            return processed

        # Process train and test splits
        if verbose:
            print(f"\n🔧 Procesando split TRAIN...")
        processed_train = process_events(train_events, train_labels, "TRAIN")

        if verbose:
            print(f"\n🔧 Procesando split TEST...")
        processed_test = process_events(test_events, test_labels, "TEST")

        # Get pipeline summary
        # IMPORTANTE: Usar Experiment explícitamente
        experiment = Experiment._load_latest_experiment()
        pipeline_summary = {
            "experiment_id": experiment.id,
            "n_filters": len(experiment.filters),
            "n_transforms": len(experiment.transform),
            "filter_names": [
                next(k for k, v in f.items() if k != "id" and isinstance(v, dict))
                for f in experiment.filters
            ],
            "transform_names": [
                next(k for k, v in t.items() if k not in ["id", "dimensionality_change"] and isinstance(v, dict))
                for t in experiment.transform
            ]
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ MINI DATASET GENERADO")
            print(f"{'='*70}")
            print(f"📊 Train: {len(processed_train)} ejemplos")
            print(f"📊 Test:  {len(processed_test)} ejemplos")
            print(f"🔧 Pipeline aplicado: {pipeline_summary['n_filters']} filtros + {pipeline_summary['n_transforms']} transformadas")

            if processed_train:
                sample_shape = processed_train[0]["signal"].shape
                print(f"📐 Shape de señales procesadas: {sample_shape}")

            print(f"{'='*70}\n")

        # Preparar labels files:
        # - Si la transformación generó labels_path, usar ese
        # - Sino, crear archivo .npy con el label escalar
        labels_dir = mini_dataset_dir / "labels"
        labels_dir.mkdir(exist_ok=True)

        train_label_files = []
        test_label_files = []

        # Procesar labels de train
        for idx, proc_data in enumerate(processed_train):
            label_file = labels_dir / f"train_label_{idx:04d}.npy"

            if proc_data.get("labels_path") and Path(proc_data["labels_path"]).exists():
                # Las labels de la transformación vienen como STRINGS, convertir a enteros
                labels_from_transform = np.load(proc_data["labels_path"], allow_pickle=True)
                # Convertir todas las labels a ID numérico (todas deberían ser iguales a proc_data["label"])
                labels_array = np.full(labels_from_transform.shape, proc_data["label"], dtype=np.int64)
                np.save(label_file, labels_array)
                train_label_files.append(str(label_file))
            else:
                # Crear archivo de labels POR FRAME desde la señal procesada
                # Cargar la señal procesada para determinar el número de frames
                signal_data = np.load(proc_data["cache_path"])

                # Determinar número de frames según la dimensionalidad
                if signal_data.ndim >= 3:
                    # Formato (n_frames, feat, n_channels) o similar
                    n_frames = signal_data.shape[0]
                elif signal_data.ndim == 2:
                    # Formato (feat, n_channels) - un solo frame
                    n_frames = 1
                else:
                    # Formato plano - un solo frame
                    n_frames = 1

                # Crear array de etiquetas (una por frame, todas iguales)
                labels_array = np.full(n_frames, proc_data["label"], dtype=np.int64)
                np.save(label_file, labels_array)
                train_label_files.append(str(label_file))

        # Procesar labels de test
        for idx, proc_data in enumerate(processed_test):
            label_file = labels_dir / f"test_label_{idx:04d}.npy"

            if proc_data.get("labels_path") and Path(proc_data["labels_path"]).exists():
                # Las labels de la transformación vienen como STRINGS, convertir a enteros
                labels_from_transform = np.load(proc_data["labels_path"], allow_pickle=True)
                # Convertir todas las labels a ID numérico
                labels_array = np.full(labels_from_transform.shape, proc_data["label"], dtype=np.int64)
                np.save(label_file, labels_array)
                test_label_files.append(str(label_file))
            else:
                # Crear archivo de labels POR FRAME desde la señal procesada
                # Cargar la señal procesada para determinar el número de frames
                signal_data = np.load(proc_data["cache_path"])

                # Determinar número de frames según la dimensionalidad
                if signal_data.ndim >= 3:
                    # Formato (n_frames, feat, n_channels) o similar
                    n_frames = signal_data.shape[0]
                elif signal_data.ndim == 2:
                    # Formato (feat, n_channels) - un solo frame
                    n_frames = 1
                else:
                    # Formato plano - un solo frame
                    n_frames = 1

                # Crear array de etiquetas (una por frame, todas iguales)
                labels_array = np.full(n_frames, proc_data["label"], dtype=np.int64)
                np.save(label_file, labels_array)
                test_label_files.append(str(label_file))

        # Prepare result
        result = {
            "train_data": [p["cache_path"] for p in processed_train],
            "train_labels": train_label_files,  # Paths a archivos .npy
            "test_data": [p["cache_path"] for p in processed_test],
            "test_labels": test_label_files,    # Paths a archivos .npy
            "class_mapping": class_mapping,
            "n_classes": n_classes,
            "n_train": len(processed_train),
            "n_test": len(processed_test),
            "pipeline_summary": pipeline_summary,
            "dataset_path": dataset_path_normalized,  # Usar path normalizada
            "pipeline_hash": current_hash  # Para validación de caché
        }

        # Save metadata to cache file for future reuse
        try:
            with open(mini_dataset_metadata_file, 'w') as f:
                json.dump(result, f, indent=2)

            if verbose:
                print(f"💾 Mini-dataset metadata guardada: {mini_dataset_metadata_file.name}")
                print(f"💾 Labels guardadas en: {labels_dir}")

        except Exception as e:
            if verbose:
                print(f"⚠️ No se pudo guardar metadata del mini-dataset: {e}")

        return result
