from pydantic import BaseModel, Field
import os
import json
from typing import List, Optional
from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from backend.classes.ClasificationModel.ClsificationModels import Classifier
from backend.classes.Filter.Filter import Filter
from backend.classes.Metrics import EvaluationMetrics


class Experiment(BaseModel):
    id: str = Field(..., description="Unique identifier for the experiment")
    transform: List[dict] = Field(..., description="Feature transformation applied to the data")
    P300Classifier: Optional[dict] = Field(..., description="Classification model configuration for P300 detection")
    innerSpeachClassifier: Optional[dict] = Field(..., description="Classification model configuration for Inner Speech detection")
    filters: List[dict] = Field(..., description="List of filter configurations used before transformation")
    evaluation: Optional[dict] = Field(..., description="Evaluation metrics for the experiment")

    @staticmethod
    def get_experiments_dir() -> str:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        experiments_dir = os.path.join(base, "backend", "Experiments")
        os.makedirs(experiments_dir, exist_ok=True)
        return experiments_dir

    @staticmethod
    def _get_last_experiment_path() -> str:
        return os.path.join(Experiment.get_experiments_dir(), "last_experiment.txt")

    @classmethod
    def _set_last_experiment_id(cls, experiment_id: str) -> None:
        path = cls._get_last_experiment_path()
        with open(path, "w") as f:
            f.write(experiment_id)

    @classmethod
    def _get_last_experiment_id(cls) -> str:
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
        existing_files = [f for f in os.listdir(directory) if f.startswith("experiment_") and f.endswith(".json")]

        existing_ids = []
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
            evaluation=None
        )


        path = os.path.join(directory, f"experiment_{experiment_id}.json")
        with open(path, "w") as f:
            json.dump(experiment.dict(), f, indent=4)

        cls._set_last_experiment_id(experiment_id)
        print(f"✅ Experimento {experiment_id} creado y registrado como el actual.")

    @classmethod
    def _load_latest_experiment(cls) -> "Experiment":
        experiment_id = cls._get_last_experiment_id()
        path = os.path.join(cls.get_experiments_dir(), f"experiment_{experiment_id}.json")
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def _save_latest_experiment(cls, experiment: "Experiment") -> None:
        path = os.path.join(cls.get_experiments_dir(), f"experiment_{experiment.id}.json")
        with open(path, "w") as f:
            json.dump(experiment.dict(), f, indent=4)

    @classmethod
    def add_transform_config(cls, transform: "Transform") -> None:
        experiment = cls._load_latest_experiment()
        
        transform_name = transform.__class__.__name__
        transform_data = transform.dict() if isinstance(transform, BaseModel) else vars(transform)
        
        if experiment.transform is None:
            experiment.transform = []
        
        experiment.transform.append({transform_name: transform_data})
        
        cls._save_latest_experiment(experiment)


    @classmethod
    def add_P300_classifier(cls, classifier) -> None:
        experiment = cls._load_latest_experiment()
        classifier_name = classifier.__class__.__name__
        classifier_data = classifier.dict() if isinstance(classifier, BaseModel) else vars(classifier)
        experiment.P300Classifier = {
            classifier_name: classifier_data
        }
        cls._save_latest_experiment(experiment)

    @classmethod
    def add_inner_speech_classifier(cls, classifier) -> None:
        experiment = cls._load_latest_experiment()
        # firt, we get the clasification model name
        classifier_name = classifier.__class__.__name__
        # We get the object atributes
        classifier_data = classifier.dict() if isinstance(classifier, BaseModel) else vars(classifier)
        # We concat the name on the atributes
        experiment.innerSpeachClassifier = {
            classifier_name: classifier_data
        }
        cls._save_latest_experiment(experiment)


    @classmethod
    def add_filter_config(cls, filter_instance) -> None:
        experiment = cls._load_latest_experiment()
        filter_name = filter_instance.__class__.__name__
    
        filter_data = filter_instance.dict() if isinstance(filter_instance, BaseModel) else vars(filter_instance)
        if experiment.filters is None: 
            experiment.filters = []

      
        experiment.filters.append({filter_name:filter_data})
 
        cls._save_latest_experiment(experiment)