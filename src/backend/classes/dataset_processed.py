"""
Utilidad para crear subsets de datasets con pipeline completo aplicado.
"""
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple
import random
import json
import numpy as np
import re
from datetime import datetime


def create_processed_subset_dataset(
    dataset_name: str,
    percentage: float,
    train_split: float,
    seed: int = 42,
    model_type: str = "inner",
    selected_classes: Optional[List[str]] = None,
    *,
    write_frames: bool = True
) -> dict:
    """Crea un subset del dataset aplicando TODO EL PIPELINE del experimento.

    Este método replica el flujo de generate_model_dataset pero permite especificar
    el porcentaje de datos a usar. Aplica: filtros → transforms del modelo → listo para entrenar.

    Estructura de salida en:
        Aux/<dataset_name>/generated_datasets/<timestamp>/

    Contiene:
        - train_X.npy / test_X.npy: Arrays procesados con pipeline
        - train_y.npy / test_y.npy: Etiquetas expandidas por frame
        - metadata.json: Información del subset y pipeline
        - class_mapping.json: Mapeo clase → índice numérico

    Args:
        dataset_name: Nombre del dataset (coincide con carpeta bajo Aux/)
        percentage: 1..100 porcentaje de eventos a usar
        train_split: fracción para entrenamiento (resto test)
        seed: semilla RNG
        model_type: "p300" o "inner" para determinar transforms del experimento
        selected_classes: Lista de clases a incluir (None = todas)

    Returns:
        dict con status, paths y resumen
    """
    from backend.classes.Experiment import Experiment

    if percentage <= 0 or percentage > 100:
        return {"status": 400, "message": "percentage fuera de rango (1-100)"}
    if train_split <= 0 or train_split >= 1:
        return {"status": 400, "message": "train_split debe estar entre 0 y 1"}

    # Buscar eventos en Aux/<dataset>/Events
    aux_root = Path("Aux") / dataset_name
    if not aux_root.exists():
        return {"status": 404, "message": f"No existe Aux/{dataset_name}"}

    # Recolectar todos los eventos .npy desde Events/
    all_events = sorted(aux_root.rglob("Events/*.npy"))
    if not all_events:
        return {"status": 404, "message": "No se encontraron eventos .npy en Events/"}

    # Agrupar por clase (del nombre del archivo)
    def _extract_class(path: Path) -> str:
        name = path.stem
        match = re.match(r"^([^\[]+)", name)
        return match.group(1) if match else "unknown"

    files_by_class = defaultdict(list)
    for event_file in all_events:
        cls = _extract_class(event_file)
        files_by_class[cls].append(event_file)

    # Filtrar clases si se especifica
    if selected_classes:
        files_by_class = {
            cls: files for cls, files in files_by_class.items()
            if cls in selected_classes
        }

    if not files_by_class:
        return {"status": 404, "message": "No se encontraron clases válidas"}

    # Crear mapeo de clases
    class_mapping = {cls: idx for idx, cls in enumerate(sorted(files_by_class.keys()))}

    # Samplear porcentaje de cada clase
    random.seed(seed)
    sampled_by_class = {}
    for cls, files in files_by_class.items():
        n_samples = max(1, int(len(files) * (percentage / 100.0)))
        sampled_by_class[cls] = random.sample(files, min(n_samples, len(files)))

    # Split train/test por clase
    train_files = []
    train_labels = []
    test_files = []
    test_labels = []

    for cls, files in sampled_by_class.items():
        cls_id = class_mapping[cls]
        random.shuffle(files)
        n_train_cls = int(len(files) * train_split)

        train_files.extend(files[:n_train_cls])
        train_labels.extend([cls_id] * n_train_cls)

        test_files.extend(files[n_train_cls:])
        test_labels.extend([cls_id] * (len(files) - n_train_cls))

    # Aplicar pipeline completo del experimento a cada archivo
    print(f"\n🔬 Aplicando pipeline del experimento (modelo {model_type})...")
    print(f"📊 Train: {len(train_files)} eventos | Test: {len(test_files)} eventos")

    train_data = []   # lista de arrays 3D (n_frames, feat, n_channels)
    train_labels_expanded = []  # lista de labels por frame (int)
    test_data = []
    test_labels_expanded = []

    # Helper: normalizar 2D a (feat, ch) con canales al final
    def _normalize_2d(arr2d: np.ndarray) -> np.ndarray:
        # Heurística: canales suelen ser pocos (<=128) y feat/tiempo grande (>=256)
        r, c = arr2d.shape
        # Si la primera dimensión parece canales (pequeña) y la segunda es larga → transponer
        if r <= 128 and c >= r * 2:
            arr2d = arr2d.T  # (time, ch)
        # Si la segunda dimensión parece canales (pequeña), ya está como (feat, ch)
        return arr2d

    # Helper: alinear lista de arrays 3D (frames, feat, ch) a misma feat y ch usando min recorte
    def _align_cubes(cubes: list[np.ndarray]) -> list[np.ndarray]:
        if not cubes:
            return cubes
        # Asegurar consistencia frames variable pero feat y ch iguales
        min_feat = min(c.shape[1] for c in cubes)
        min_ch = min(c.shape[2] for c in cubes)
        aligned = []
        for c in cubes:
            cc = c[:, :min_feat, :min_ch]
            aligned.append(cc)
        return aligned

    # Procesar train
    for idx, file_path in enumerate(train_files):
        try:
            # IMPORTANTE: force_recalculate=True para aplicar transforms actualizadas
            result = Experiment.apply_model_pipeline(
                file_path=str(file_path),
                model_type=model_type,
                force_recalculate=True,  # ← Forzar recálculo para aplicar pipeline actualizado
                save_intermediates=False,
                verbose=False,
                flatten_output=False  # ← Mantener 3D si la transform genera cubo
            )
            signal = result["signal"]

            # Validación estricta: no forzar dimensiones
            if signal.ndim != 3:
                raise ValueError(
                    f"Se esperaba señal 3D (frames, feat, ch); recibido ndim={signal.ndim} con shape {np.shape(signal)}"
                )

            train_data.append(signal)

            # Expandir etiqueta por frames si es 3D
            if signal.ndim == 3:
                n_frames = signal.shape[0]
                train_labels_expanded.extend([train_labels[idx]] * n_frames)
            else:
                train_labels_expanded.append(train_labels[idx])

        except Exception as e:
            print(f"  ⚠️ Error en train {file_path.name}: {e}")
            continue

    # Procesar test
    for idx, file_path in enumerate(test_files):
        try:
            # IMPORTANTE: force_recalculate=True para aplicar transforms actualizadas
            result = Experiment.apply_model_pipeline(
                file_path=str(file_path),
                model_type=model_type,
                force_recalculate=True,  # ← Forzar recálculo para aplicar pipeline actualizado
                save_intermediates=False,
                verbose=False,
                flatten_output=False  # ← Mantener 3D si la transform genera cubo
            )
            signal = result["signal"]

            # Validación estricta: no forzar dimensiones
            if signal.ndim != 3:
                raise ValueError(
                    f"Se esperaba señal 3D (frames, feat, ch); recibido ndim={signal.ndim} con shape {np.shape(signal)}"
                )

            test_data.append(signal)

            # Expandir etiqueta por frames si es 3D
            if signal.ndim == 3:
                n_frames = signal.shape[0]
                test_labels_expanded.extend([test_labels[idx]] * n_frames)
            else:
                test_labels_expanded.append(test_labels[idx])

        except Exception as e:
            print(f"  ⚠️ Error en test {file_path.name}: {e}")
            continue

    if not train_data or not test_data:
        return {"status": 500, "message": "Error aplicando pipeline: no se procesó ningún evento"}

    # Alinear shapes dentro de cada split (recorte interno) y luego garantizar CONSISTENCIA GLOBAL
    train_data = _align_cubes(train_data)
    test_data = _align_cubes(test_data)

    # Verificación global: todas las muestras (train+test) deben compartir feat/ch finales.
    # Si difieren todavía entre splits se recorta nuevamente de forma conjunta.
    if train_data and test_data:
        feat_train = {c.shape[1] for c in train_data}
        ch_train = {c.shape[2] for c in train_data}
        feat_test = {c.shape[1] for c in test_data}
        ch_test = {c.shape[2] for c in test_data}
        if (len(feat_train) != 1 or len(ch_train) != 1 or
                len(feat_test) != 1 or len(ch_test) != 1 or
                next(iter(feat_train)) != next(iter(feat_test)) or
                next(iter(ch_train)) != next(iter(ch_test))):
            # Recalcular mínimo global y recortar ambos splits para asegurar igualdad.
            min_feat_global = min(min(feat_train), min(feat_test))
            min_ch_global = min(min(ch_train), min(ch_test))
            def _crop_global(cubes: list[np.ndarray], mf: int, mc: int) -> list[np.ndarray]:
                out_list = []
                for cube in cubes:
                    out_list.append(cube[:, :mf, :mc])
                return out_list
            print(f"⚠️ [Subset] Inconsistencia feat/ch entre train y test. Recortando globalmente a (feat={min_feat_global}, ch={min_ch_global}).")
            train_data = _crop_global(train_data, min_feat_global, min_ch_global)
            test_data = _crop_global(test_data, min_feat_global, min_ch_global)
            global_aligned = True
        else:
            global_aligned = False
    else:
        global_aligned = False

    # Crear directorio de salida
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = aux_root / "generated_datasets" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # 🔧 CAPTURAR CONFIGURACIÓN COMPLETA DEL EXPERIMENTO para validación posterior
    exp_snapshot = {}
    try:
        exp = Experiment._load_latest_experiment()
        # Guardar filtros
        exp_snapshot["filters"] = exp.filters if exp.filters else []
        # Guardar transforms según model_type
        if model_type == "p300" and exp.P300Classifier:
            for model_name, model_cfg in exp.P300Classifier.items():
                exp_snapshot["transform"] = model_cfg.get("transform", {})
                exp_snapshot["classifier_config"] = model_cfg.get("config", {})
                break
        elif model_type == "inner" and exp.innerSpeachClassifier:
            for model_name, model_cfg in exp.innerSpeachClassifier.items():
                exp_snapshot["transform"] = model_cfg.get("transform", {})
                exp_snapshot["classifier_config"] = model_cfg.get("config", {})
                break
        else:
            exp_snapshot["transform"] = {}
            exp_snapshot["classifier_config"] = {}
    except Exception as e:
        print(f"⚠️ No se pudo capturar configuración del experimento: {e}")
        exp_snapshot = {"filters": [], "transform": {}, "classifier_config": {}}

    # Concatenar y guardar arrays (formato 3D garantizado)
    X_train = np.concatenate(train_data, axis=0)
    y_train = np.array(train_labels_expanded, dtype=np.int64)
    X_test = np.concatenate(test_data, axis=0)
    y_test = np.array(test_labels_expanded, dtype=np.int64)

    train_x_path = str(out_dir / "train_X.npy")
    train_y_path = str(out_dir / "train_y.npy")
    test_x_path = str(out_dir / "test_X.npy")
    test_y_path = str(out_dir / "test_y.npy")

    np.save(train_x_path, X_train)
    np.save(train_y_path, y_train)
    np.save(test_x_path, X_test)
    np.save(test_y_path, y_test)

    print(f"✅ Arrays guardados (3D): train {X_train.shape}, test {X_test.shape}")

    train_x_files = None
    train_y_files = None
    test_x_files = None
    test_y_files = None

    if write_frames:
        # Adicional: guardar frames por separado (una muestra por archivo)
        frames_train_dir = out_dir / "train" / "data"
        frames_train_lbl_dir = out_dir / "train" / "labels"
        frames_test_dir = out_dir / "test" / "data"
        frames_test_lbl_dir = out_dir / "test" / "labels"
        frames_train_dir.mkdir(parents=True, exist_ok=True)
        frames_train_lbl_dir.mkdir(parents=True, exist_ok=True)
        frames_test_dir.mkdir(parents=True, exist_ok=True)
        frames_test_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Helpers para escribir una muestra por archivo, conservando 3D (1, feat, ch)
        def _dump_frames(X: np.ndarray, y: np.ndarray, data_dir: Path, lbl_dir: Path, prefix: str) -> list[tuple[str, str]]:
            paths: list[tuple[str, str]] = []
            n = X.shape[0]
            for i in range(n):
                x_i = X[i:i+1]  # (1, feat, ch)
                y_i = np.array([int(y[i])], dtype=np.int64)
                xf = data_dir / f"{prefix}_{i:06d}.npy"
                yf = lbl_dir / f"{prefix}_{i:06d}.npy"
                np.save(str(xf), x_i)
                np.save(str(yf), y_i)
                paths.append((str(xf), str(yf)))
            return paths

        train_pairs = _dump_frames(X_train, y_train, frames_train_dir, frames_train_lbl_dir, "frame")
        test_pairs = _dump_frames(X_test, y_test, frames_test_dir, frames_test_lbl_dir, "frame")

        # Manifests de listas de archivos por split
        train_x_files = str(out_dir / "train_x_files.json")
        train_y_files = str(out_dir / "train_y_files.json")
        test_x_files = str(out_dir / "test_x_files.json")
        test_y_files = str(out_dir / "test_y_files.json")

        with open(train_x_files, "w", encoding="utf-8") as f:
            json.dump([p[0] for p in train_pairs], f, ensure_ascii=False, indent=2)
        with open(train_y_files, "w", encoding="utf-8") as f:
            json.dump([p[1] for p in train_pairs], f, ensure_ascii=False, indent=2)
        with open(test_x_files, "w", encoding="utf-8") as f:
            json.dump([p[0] for p in test_pairs], f, ensure_ascii=False, indent=2)
        with open(test_y_files, "w", encoding="utf-8") as f:
            json.dump([p[1] for p in test_pairs], f, ensure_ascii=False, indent=2)

    # Guardar metadata CON snapshot del experimento para validación posterior
    experiment = Experiment._load_latest_experiment()
    meta = {
        "dataset_name": dataset_name,
        "timestamp": ts,
        "percentage": percentage,
        "train_split": train_split,
        "seed": seed,
        "model_type": model_type,
        "experiment_id": experiment.id,
        "experiment_snapshot": exp_snapshot,  # ← Copia completa de filtros/transforms
        "n_train_events": len(train_files),
        "n_test_events": len(test_files),
        "n_train_frames": len(train_labels_expanded),
        "n_test_frames": len(test_labels_expanded),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "feature_dim": int(X_train.shape[1]),
        "channel_dim": int(X_train.shape[2]),
        "global_aligned": bool(global_aligned),
        "pipeline_applied": True,
        "filters_count": len(experiment.filters),
        "classes": list(class_mapping.keys()),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Guardar class_mapping
    with open(out_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)

    return {
        "status": 200,
        "message": "Subset creado con pipeline aplicado",
        "subset_dir": str(out_dir),
        "train_X": train_x_path,
        "train_y": train_y_path,
        "test_X": test_x_path,
        "test_y": test_y_path,
    "train_x_files": train_x_files,
    "train_y_files": train_y_files,
    "test_x_files": test_x_files,
    "test_y_files": test_y_files,
        "metadata": str(out_dir / "metadata.json"),
        "class_mapping": str(out_dir / "class_mapping.json"),
        "n_train": len(train_labels_expanded),
        "n_test": len(test_labels_expanded),
        "n_train_events": len(train_files),
        "n_test_events": len(test_files),
        "n_classes": len(class_mapping),
        "classes": list(class_mapping.keys()),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
    }


def list_available_subsets(dataset_name: str) -> List[dict]:
    """Lista todos los subsets generados para un dataset.
    
    Returns:
        Lista de dicts con info de cada subset: {
            "dir": str,
            "timestamp": str,
            "metadata": dict,
            "compatible": bool,
            "compatibility_info": str
        }
    """
    from backend.classes.Experiment import Experiment
    
    aux_root = Path("Aux") / dataset_name
    subsets_dir = aux_root / "generated_datasets"
    
    if not subsets_dir.exists():
        return []
    
    subsets = []
    for subset_dir in sorted(subsets_dir.iterdir()):
        if not subset_dir.is_dir():
            continue
            
        metadata_file = subset_dir / "metadata.json"
        if not metadata_file.exists():
            continue
            
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Validar compatibilidad con el experimento actual
            compatible, info = validate_subset_compatibility(metadata)
            
            subsets.append({
                "dir": str(subset_dir),
                "timestamp": metadata.get("timestamp", "unknown"),
                "metadata": metadata,
                "compatible": compatible,
                "compatibility_info": info
            })
        except Exception as e:
            print(f"⚠️ Error leyendo metadata de {subset_dir}: {e}")
            continue
    
    return subsets


def validate_subset_compatibility(subset_metadata: dict) -> Tuple[bool, str]:
    """Valida si un subset es compatible con el experimento actual.
    
    Compara:
    - Filtros aplicados
    - Transformadas configuradas
    - Tipo de modelo (p300/inner)
    
    Returns:
        (compatible: bool, info: str con detalles)
    """
    from backend.classes.Experiment import Experiment
    
    try:
        current_exp = Experiment._load_latest_experiment()
    except Exception as e:
        return False, f"No se pudo cargar experimento actual: {e}"
    
    # Extraer snapshot guardado en el subset
    exp_snapshot = subset_metadata.get("experiment_snapshot", {})
    if not exp_snapshot:
        return False, "⚠️ Subset sin snapshot de experimento (generado con versión antigua)"
    
    model_type = subset_metadata.get("model_type", "unknown")
    
    # Obtener configuración actual del experimento
    current_filters = current_exp.filters if current_exp.filters else []
    current_transform = {}
    current_classifier = {}
    
    if model_type == "p300" and current_exp.P300Classifier:
        for model_name, model_cfg in current_exp.P300Classifier.items():
            current_transform = model_cfg.get("transform", {})
            current_classifier = model_cfg.get("config", {})
            break
    elif model_type == "inner" and current_exp.innerSpeachClassifier:
        for model_name, model_cfg in current_exp.innerSpeachClassifier.items():
            current_transform = model_cfg.get("transform", {})
            current_classifier = model_cfg.get("config", {})
            break
    
    # Comparar filtros
    subset_filters = exp_snapshot.get("filters", [])
    if json.dumps(current_filters, sort_keys=True) != json.dumps(subset_filters, sort_keys=True):
        return False, "❌ Filtros diferentes al experimento actual"
    
    # Comparar transformadas (estructura y parámetros)
    subset_transform = exp_snapshot.get("transform", {})
    if json.dumps(current_transform, sort_keys=True) != json.dumps(subset_transform, sort_keys=True):
        return False, "❌ Transformadas diferentes al experimento actual"

    # Chequeo de shape final esperado vs shapes del subset
    try:
        final_output_shape = None
        # Buscar última transform con 'output_shape' en snapshot actual del experimento
        if isinstance(current_transform, dict):
            # Algunas implementaciones guardan lista de transforms bajo 'pipeline' o similar
            pipeline = current_transform.get("pipeline") or current_transform.get("transforms")
            if isinstance(pipeline, list) and pipeline:
                for t in reversed(pipeline):
                    if isinstance(t, dict) and t.get("output_shape"):
                        final_output_shape = t.get("output_shape")
                        break
            # Alternativamente si current_transform ya es la última transform
            if final_output_shape is None and current_transform.get("output_shape"):
                final_output_shape = current_transform.get("output_shape")

        train_shape = subset_metadata.get("train_shape")
        test_shape = subset_metadata.get("test_shape")

        if final_output_shape and train_shape and test_shape:
            # Normalizar a tuplas
            fos = tuple(final_output_shape)
            tr = tuple(train_shape)
            te = tuple(test_shape)

            # Aceptar fos de longitud 2 (feat, ch) si el subset añade frames al inicio
            def shapes_compatible(fos_tuple, subset_tuple):
                if len(fos_tuple) == len(subset_tuple):
                    return fos_tuple == subset_tuple
                # Caso: subset 3D (frames, feat, ch) y fos 2D (feat, ch)
                if len(subset_tuple) == 3 and len(fos_tuple) == 2:
                    return fos_tuple[0] == subset_tuple[1] and fos_tuple[1] == subset_tuple[2]
                return False

            if not shapes_compatible(fos, tr) or not shapes_compatible(fos, te):
                return False, f"❌ Shape final transform {fos} no coincide con subset train={tr} test={te}"
    except Exception as _shape_err:
        # No bloquear por errores de introspección de shape
        pass

    # Validación adicional: consistencia train/test en dimensiones de features y canales
    try:
        train_shape = subset_metadata.get("train_shape")
        test_shape = subset_metadata.get("test_shape")
        if train_shape and test_shape:
            if len(train_shape) < 2 or len(test_shape) < 2:
                return False, "❌ Subset con shapes insuficientes"
            # Comparar últimas dos dimensiones como (feat, ch)
            if train_shape[-2:] != test_shape[-2:]:
                return False, f"❌ Features/canales difieren train={train_shape[-2:]} test={test_shape[-2:]}"
    except Exception:
        pass
    
    return True, "✅ Compatible con experimento actual"


def get_subset_by_dir(subset_dir: str) -> Optional[dict]:
    """Carga la información completa de un subset dado su directorio.
    
    Returns:
        Dict con paths y metadata, o None si no existe
    """
    subset_path = Path(subset_dir)
    if not subset_path.exists():
        return None
    
    metadata_file = subset_path / "metadata.json"
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        return {
            "status": 200,
            "subset_dir": str(subset_path),
            "train_X": str(subset_path / "train_X.npy"),
            "train_y": str(subset_path / "train_y.npy"),
            "test_X": str(subset_path / "test_X.npy"),
            "test_y": str(subset_path / "test_y.npy"),
            "metadata": str(metadata_file),
            "class_mapping": str(subset_path / "class_mapping.json"),
            "metadata_content": metadata
        }
    except Exception as e:
        print(f"⚠️ Error cargando subset: {e}")
        return None
