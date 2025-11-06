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

        experiment.P300Classifier = {
            classifier_name: classifier_data
        }
        cls._save_latest_experiment(experiment)

    @classmethod
    def add_inner_speech_classifier(cls, classifier: Classifier) -> None:
        """
        Stores the Inner Speech classifier configuration.
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

        experiment.innerSpeachClassifier = {
            classifier_name: classifier_data
        }
        cls._save_latest_experiment(experiment)
