"""
Helper centralizado para gestión de almacenamiento de modelos entrenados.

Estructura de directorios:
    backend/models/
        {experiment_id}/
            {model_type}/  # p300 o inner
                {model_name}_{timestamp}.pkl
                experiment_snapshot.json
                training_info.json
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
import os


class ModelStorage:
    """Gestiona el almacenamiento de modelos entrenados con sus metadatos."""

    # Directorio base relativo desde src/
    BASE_DIR = Path("backend/models")

    @classmethod
    def get_base_dir(cls) -> Path:
        """
        Retorna el directorio base absoluto para modelos.

        Returns:
            Path absoluto al directorio backend/models/
        """
        # Obtener directorio actual de trabajo
        cwd = Path.cwd()

        # Si estamos en src/, usar directamente BASE_DIR
        if cwd.name == "src":
            return cwd / cls.BASE_DIR

        # Si estamos en raíz del proyecto, agregar src/
        if (cwd / "src").exists():
            return cwd / "src" / cls.BASE_DIR

        # Fallback: intentar encontrar src/ en el árbol
        current = cwd
        for _ in range(5):  # Máximo 5 niveles arriba
            if (current / "src").exists():
                return current / "src" / cls.BASE_DIR
            current = current.parent

        # Si no encontramos, usar ruta relativa directa
        return Path(cls.BASE_DIR)

    @classmethod
    def generate_model_path(
        cls,
        experiment_id: str,
        model_type: str,  # "p300" o "inner"
        model_name: str,
        timestamp: Optional[str] = None
    ) -> Path:
        """
        Genera ruta completa para guardar un modelo.

        Args:
            experiment_id: ID del experimento (e.g., "exp_20251112_143022")
            model_type: Tipo de modelo ("p300" o "inner")
            model_name: Nombre del modelo (e.g., "SVNN", "SVM", "LSTM")
            timestamp: Timestamp opcional (se genera automáticamente si no se proporciona)

        Returns:
            Path completo: backend/models/{experiment_id}/{model_type}/{model_name}_{timestamp}.pkl

        Example:
            >>> ModelStorage.generate_model_path("exp_001", "p300", "SVNN")
            PosixPath('backend/models/exp_001/p300/svnn_20251112_143022.pkl')
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{model_name.lower()}_{timestamp}.pkl"
        base_dir = cls.get_base_dir()

        return base_dir / experiment_id / model_type / filename

    @classmethod
    def save_experiment_snapshot(
        cls,
        experiment_id: str,
        model_type: str,
        experiment_data: Dict[str, Any]
    ) -> Path:
        """
        Guarda snapshot del experimento (filtros, transforms, configs).

        Args:
            experiment_id: ID del experimento
            model_type: Tipo de modelo ("p300" o "inner")
            experiment_data: Diccionario con configuración completa del experimento

        Returns:
            Path al archivo experiment_snapshot.json guardado

        Example:
            experiment_data = {
                "id": "exp_001",
                "dataset": "BNCI2014-001",
                "filters": [...],
                "transform": [...],
                "classifier_config": {...}
            }
            ModelStorage.save_experiment_snapshot("exp_001", "p300", experiment_data)
        """
        base_dir = cls.get_base_dir()
        snapshot_dir = base_dir / experiment_id / model_type
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = snapshot_dir / "experiment_snapshot.json"

        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)

        print(f"📋 [ModelStorage] Snapshot guardado en: {snapshot_path}")
        return snapshot_path

    @classmethod
    def save_training_info(
        cls,
        experiment_id: str,
        model_type: str,
        training_info: Dict[str, Any]
    ) -> Path:
        """
        Guarda información del entrenamiento (métricas, hiperparámetros, tiempos).

        Args:
            experiment_id: ID del experimento
            model_type: Tipo de modelo ("p300" o "inner")
            training_info: Diccionario con información del entrenamiento

        Returns:
            Path al archivo training_info.json guardado

        Example:
            training_info = {
                "model_name": "SVNN",
                "timestamp": "20251112_143022",
                "metrics": {...},
                "hyperparams": {...},
                "training_time": 123.45
            }
            ModelStorage.save_training_info("exp_001", "p300", training_info)
        """
        base_dir = cls.get_base_dir()
        info_dir = base_dir / experiment_id / model_type
        info_dir.mkdir(parents=True, exist_ok=True)

        info_path = info_dir / "training_info.json"

        # Si ya existe, cargar y agregar nueva entrada
        existing_info = []
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                existing_info = json.load(f)

        # Agregar nueva info
        existing_info.append(training_info)

        # Guardar actualizado
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(existing_info, f, indent=2, ensure_ascii=False)

        print(f"📊 [ModelStorage] Info de entrenamiento guardada en: {info_path}")
        return info_path

    @classmethod
    def load_experiment_snapshot(cls, experiment_id: str, model_type: str) -> Dict[str, Any]:
        """
        Carga el snapshot del experimento.

        Args:
            experiment_id: ID del experimento
            model_type: Tipo de modelo ("p300" o "inner")

        Returns:
            Diccionario con configuración del experimento
        """
        base_dir = cls.get_base_dir()
        snapshot_path = base_dir / experiment_id / model_type / "experiment_snapshot.json"

        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot no encontrado: {snapshot_path}")

        with open(snapshot_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def list_models(cls, experiment_id: Optional[str] = None, model_type: Optional[str] = None) -> list:
        """
        Lista todos los modelos guardados.

        Args:
            experiment_id: Filtrar por experimento (opcional)
            model_type: Filtrar por tipo ("p300" o "inner", opcional)

        Returns:
            Lista de diccionarios con información de cada modelo
        """
        base_dir = cls.get_base_dir()

        if not base_dir.exists():
            return []

        models = []

        # Iterar sobre experimentos
        for exp_dir in base_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            if experiment_id and exp_dir.name != experiment_id:
                continue

            # Iterar sobre tipos (p300, inner)
            for type_dir in exp_dir.iterdir():
                if not type_dir.is_dir():
                    continue

                if model_type and type_dir.name != model_type:
                    continue

                # Buscar archivos .pkl
                for pkl_file in type_dir.glob("*.pkl"):
                    models.append({
                        "experiment_id": exp_dir.name,
                        "model_type": type_dir.name,
                        "model_file": pkl_file.name,
                        "full_path": str(pkl_file),
                        "timestamp": pkl_file.stat().st_mtime
                    })

        # Ordenar por timestamp descendente
        models.sort(key=lambda x: x["timestamp"], reverse=True)

        return models


# ========== Funciones de conveniencia ==========

def get_current_experiment_id() -> str:
    """
    Obtiene el ID del experimento actual desde Experiment.

    Returns:
        ID del experimento actual
    """
    try:
        from backend.classes.Experiment import Experiment
        experiment = Experiment._load_latest_experiment()
        return experiment.id
    except Exception as e:
        print(f"⚠️ [ModelStorage] No se pudo cargar ID del experimento: {e}")
        # Generar ID temporal
        return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_model_with_metadata(
    model_instance,
    model_name: str,
    model_type: str,
    metrics: Dict[str, Any],
    experiment_snapshot: Dict[str, Any],
    hyperparams: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Guarda un modelo junto con todos sus metadatos.

    Args:
        model_instance: Instancia del modelo (con método save())
        model_name: Nombre del modelo (e.g., "SVNN")
        model_type: Tipo ("p300" o "inner")
        metrics: Métricas de evaluación
        experiment_snapshot: Snapshot del experimento completo
        hyperparams: Hiperparámetros del entrenamiento (opcional)

    Returns:
        Path al modelo guardado
    """
    experiment_id = get_current_experiment_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generar path del modelo
    model_path = ModelStorage.generate_model_path(
        experiment_id=experiment_id,
        model_type=model_type,
        model_name=model_name,
        timestamp=timestamp
    )

    # Crear directorio
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar modelo
    model_instance.save(str(model_path))
    print(f"💾 [ModelStorage] Modelo guardado en: {model_path}")

    # Guardar snapshot del experimento
    ModelStorage.save_experiment_snapshot(
        experiment_id=experiment_id,
        model_type=model_type,
        experiment_data=experiment_snapshot
    )

    # Guardar info de entrenamiento
    training_info = {
        "model_name": model_name,
        "timestamp": timestamp,
        "model_file": model_path.name,
        "metrics": metrics,
        "hyperparams": hyperparams or {}
    }

    ModelStorage.save_training_info(
        experiment_id=experiment_id,
        model_type=model_type,
        training_info=training_info
    )

    return model_path
