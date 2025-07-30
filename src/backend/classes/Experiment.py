from pydantic import BaseModel, Field
import os
import json
from typing import Type
from backend.classes.FeatureExtracture.FeatureExtracture import Transform
from backend.classes.ClasificationModel.ClsificationModels import Classifier
from backend.classes.Filter.Filter import Filter
from backend.classes.Metrics import EvaluationMetrics

class Experiment(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier for the experiment"
    )
    transform: Transform = Field(
        ...,
        description="Feature transformation applied to the data"
    )
    classifier: Classifier = Field(
        ...,
        description="Classification model configuration"
    )
    filter: Filter = Field(
        ...,
        description="Filter configuration used before transformation"
    )
    evaluation: EvaluationMetrics = Field(
        ...,
        description="Evaluation metrics for the experiment"
    )
    @classmethod
    def create_blank_json(cls, directory: str) -> str:
        """
        Creates a blank JSON file with default values for all attributes.
        The file is saved in the given directory with an autoincremented id.
        Returns the name (id) of the experiment created.
        """
        
        os.makedirs(directory, exist_ok=True)

        # Buscar archivos existentes tipo experiment_*.json
        existing_files = [
            f for f in os.listdir(directory)
            if f.startswith("experiment_") and f.endswith(".json")
        ]

        # Determinar el siguiente id disponible
        existing_ids = []
        for f in existing_files:
            try:
                num = int(f.replace("experiment_", "").replace(".json", ""))
                existing_ids.append(num)
            except ValueError:
                continue

        next_id = max(existing_ids, default=0) + 1
        experiment_id = str(next_id)

        # Crear instancia con valores por defecto
        new_experiment = cls(
            id=experiment_id,
            transform=Transform(),  # Se espera que Transform tenga valores por defecto
            classifier=Classifier(),  # Igual
            filter=Filter(),  # Igual
            evaluation=EvaluationMetrics()
        )

        # Guardar como JSON
        file_path = os.path.join(directory, f"experiment_{experiment_id}.json")
        with open(file_path, "w") as f:
            json.dump(new_experiment.dict(), f, indent=4)

        return experiment_id