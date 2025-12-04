"""
Motor de Simulación para P300 + Inner Speech
=============================================

Clase SimulationEngine que maneja el procesamiento de ventanas deslizantes,
aplicación de pipelines y predicciones con modelos P300 e Inner Speech.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from backend.helpers.simulation_utils import (
    apply_pipeline_from_snapshot,
    predict_with_model,
    extract_window_with_padding
)
from shared.fileUtils import get_dataset_metadata


class SimulationEngine:
    """
    Motor de simulación que procesa señal EEG con modelos P300 e Inner Speech.

    Workflow:
    1. Ventaneo deslizante de la señal raw
    2. Para cada ventana: aplicar pipeline P300 → predecir
    3. Si P300 = 1: aplicar pipeline Inner Speech → predecir clase
    4. Comparar con labels reales para evaluación
    """

    def __init__(
        self,
        raw_signal: np.ndarray,
        labels: np.ndarray,
        sfreq: float,
        p300_model_config: Dict[str, Any],
        inner_model_config: Dict[str, Any],
        hop_percent: float = 50.0,
        dataset_name: str = None
    ):
        """
        Inicializa el motor de simulación.

        Args:
            raw_signal: Señal EEG completa (n_channels, n_samples)
            labels: Labels por muestra (1, n_samples) o (n_samples,)
            sfreq: Frecuencia de muestreo en Hz (ej: 1024)
            p300_model_config: Dict retornado por load_model_for_inference()
            inner_model_config: Dict retornado por load_model_for_inference()
            hop_percent: Porcentaje de hop (25-75), default 50%
            dataset_name: Nombre del dataset (ej: "arabic_inner_speech") para mapear labels
        """
        self.raw_signal = raw_signal
        self.labels = labels.flatten() if labels.ndim > 1 else labels
        self.sfreq = sfreq
        self.p300_config = p300_model_config
        self.inner_config = inner_model_config

        # Cargar metadata del dataset para mapeo de labels
        self.dataset_classes = []
        self.label_to_class_map = {}
        if dataset_name:
            try:
                metadata = get_dataset_metadata(dataset_name)
                self.dataset_classes = metadata.get('classes', [])
                # Crear mapeo: índice numérico -> nombre de clase
                self.label_to_class_map = {i: cls for i, cls in enumerate(self.dataset_classes)}
                # También mapear strings directamente
                for cls in self.dataset_classes:
                    self.label_to_class_map[cls] = cls
                    self.label_to_class_map[str(cls).lower()] = cls
                print(f"[SimulationEngine] Mapeo de clases cargado: {self.dataset_classes}")
            except Exception as e:
                print(f"[SimulationEngine] ⚠️ No se pudo cargar metadata del dataset: {e}")
                print(f"[SimulationEngine] Las labels se usarán tal cual")

        # Calcular parámetros de ventaneo
        self.window_size_p300 = p300_model_config['window_size_samples']
        self.hop_size = int(self.window_size_p300 * (hop_percent / 100))

        print(f"\n{'='*70}")
        print(f"[SimulationEngine] INICIALIZACIÓN")
        print(f"{'='*70}")
        print(f"  Señal: {raw_signal.shape[0]} canales × {raw_signal.shape[1]} samples")
        print(f"  Frecuencia: {sfreq} Hz")
        print(f"  Modelo P300: {p300_model_config['model_name']}")
        print(f"    - Window size: {self.window_size_p300} samples ({self.window_size_p300/sfreq:.2f}s)")
        print(f"  Modelo Inner: {inner_model_config['model_name']}")
        print(f"    - Window size: {inner_model_config['window_size_samples']} samples ({inner_model_config['window_size_samples']/sfreq:.2f}s)")
        print(f"  Hop: {hop_percent}% = {self.hop_size} samples ({self.hop_size/sfreq:.2f}s)")

        # Generar índices de ventanas
        self.window_indices = self._generate_window_indices()
        print(f"  Total de ventanas: {len(self.window_indices)}")
        print(f"{'='*70}\n")

        # Cache de resultados
        self.results: List[Dict] = []

    def _generate_window_indices(self) -> List[Tuple[int, int]]:
        """
        Genera lista de índices (start, end) para ventanas deslizantes.

        Returns:
            Lista de tuplas (start, end)
        """
        indices = []
        signal_length = self.raw_signal.shape[1]

        start = 0
        while start + self.window_size_p300 <= signal_length:
            end = start + self.window_size_p300
            indices.append((start, end))
            start += self.hop_size

        return indices

    def process_window(self, window_idx: int) -> Dict[str, Any]:
        """
        Procesa una ventana específica con ambos modelos.

        Args:
            window_idx: Índice de la ventana a procesar

        Returns:
            {
                "window_idx": int,
                "start_sample": int,
                "end_sample": int,
                "time_sec": float,
                "p300_prediction": int (0 o 1),
                "p300_confidence": float,
                "p300_probabilities": list,
                "inner_prediction": int or None,
                "inner_confidence": float or None,
                "inner_probabilities": list or None,
                "label_real": str,
                "is_correct": bool
            }
        """
        start, end = self.window_indices[window_idx]
        time_sec = start / self.sfreq

        # ========== PASO 1: MODELO P300 ==========
        # Extraer ventana
        ventana_p300 = self.raw_signal[:, start:end]

        # Aplicar pipeline P300
        try:
            transformed_p300 = apply_pipeline_from_snapshot(
                raw_window=ventana_p300,
                snapshot_pipeline=self.p300_config['pipeline_config'],
                model_type="p300",
                experiment_id=self.p300_config['model_metadata'].get('experiment_id')
            )
        except Exception as e:
            print(f"[SimulationEngine] ERROR aplicando pipeline P300 en ventana {window_idx}: {e}")
            return self._error_result(window_idx, start, end, time_sec, str(e))

        # Predecir con modelo P300
        try:
            # P300 model classes (binary: 0=no_p300, 1=p300)
            p300_classes = self.p300_config['model_metadata'].get('classes', ['no_p300', 'p300'])

            p300_result = predict_with_model(
                model_instance=self.p300_config['model_instance'],
                model_name=self.p300_config['model_name'],
                transformed_data=transformed_p300,
                frame_context=self.p300_config.get('frame_context'),
                class_names=p300_classes
            )
        except Exception as e:
            print(f"[SimulationEngine] ERROR prediciendo con P300 en ventana {window_idx}: {e}")
            return self._error_result(window_idx, start, end, time_sec, str(e))

        # ========== PASO 2: MODELO INNER SPEECH (si P300 = 1) ==========
        inner_result = None

        if p300_result['prediction'] == 1:
            # Extraer ventana para Inner Speech (start-1 con padding)
            window_size_inner = self.inner_config['window_size_samples']
            start_inner = max(0, start - 1)

            ventana_inner = extract_window_with_padding(
                raw_signal=self.raw_signal,
                start=start_inner,
                window_size=window_size_inner
            )

            # Aplicar pipeline Inner Speech
            try:
                transformed_inner = apply_pipeline_from_snapshot(
                    raw_window=ventana_inner,
                    snapshot_pipeline=self.inner_config['pipeline_config'],
                    model_type="inner",
                    experiment_id=self.inner_config['model_metadata'].get('experiment_id')
                )
            except Exception as e:
                print(f"[SimulationEngine] ERROR aplicando pipeline Inner en ventana {window_idx}: {e}")
                # Continuar con P300 solamente
                transformed_inner = None

            # Predecir clase si el pipeline fue exitoso
            if transformed_inner is not None:
                try:
                    # Inner Speech model classes (ej: ["rest", "arriba", "abajo", ...])
                    inner_classes = self.inner_config['model_metadata'].get('classes', [])

                    inner_result = predict_with_model(
                        model_instance=self.inner_config['model_instance'],
                        model_name=self.inner_config['model_name'],
                        transformed_data=transformed_inner,
                        frame_context=self.inner_config.get('frame_context'),
                        class_names=inner_classes
                    )
                except Exception as e:
                    print(f"[SimulationEngine] ERROR prediciendo con Inner en ventana {window_idx}: {e}")
                    inner_result = None

        # ========== PASO 3: EVALUACIÓN ==========
        # Obtener label real (mayoría en la ventana)
        label_real = self._get_majority_label(start, end)

        # Evaluar correctitud
        is_correct = self._evaluate_prediction(
            p300_pred=p300_result['prediction'],
            inner_pred=inner_result['prediction'] if inner_result else None,
            label_real=label_real
        )

        # ========== RESULTADO ==========
        result = {
            "window_idx": window_idx,
            "start_sample": start,
            "end_sample": end,
            "time_sec": time_sec,
            "p300_prediction": p300_result['prediction'],
            "p300_confidence": p300_result['confidence'],
            "p300_probabilities": p300_result['probabilities'],
            "inner_prediction": inner_result['prediction'] if inner_result else None,
            "inner_confidence": inner_result['confidence'] if inner_result else None,
            "inner_probabilities": inner_result['probabilities'] if inner_result else None,
            "label_real": label_real,
            "is_correct": is_correct
        }

        return result

    def process_all_windows(self, verbose: bool = True) -> List[Dict]:
        """
        Procesa todas las ventanas de la sesión.

        Args:
            verbose: Mostrar progreso

        Returns:
            Lista de resultados (uno por ventana)
        """
        self.results = []
        total = len(self.window_indices)

        print(f"\n[SimulationEngine] Procesando {total} ventanas...")

        for idx in range(total):
            # Mostrar progreso
            if verbose and (idx % 100 == 0 or idx == total - 1):
                progress = (idx + 1) / total * 100
                print(f"  Progreso: {idx+1}/{total} ({progress:.1f}%)")

            result = self.process_window(idx)
            self.results.append(result)

        print(f"[SimulationEngine] ✅ Procesamiento completado\n")
        return self.results

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Calcula métricas globales de performance.

        Returns:
            {
                "total_windows": int,
                "correct": int,
                "accuracy": float,
                "p300_detected": int,
                "p300_detection_rate": float,
                "by_class": {
                    "rest": {"total": int, "correct": int, "accuracy": float},
                    ...
                }
            }
        """
        if not self.results:
            return {}

        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])

        # Detecciones P300
        p300_detected = sum(1 for r in self.results if r['p300_prediction'] == 1)

        # ===== FIX: Métricas por clase (separar clases del modelo de clases desconocidas) =====
        unique_labels = set(r['label_real'] for r in self.results)
        by_class = {}

        # Obtener clases que el modelo Inner conoce
        inner_classes = self.inner_config['model_metadata'].get('classes', [])
        inner_classes_lower = [str(c).lower() for c in inner_classes]

        for label in unique_labels:
            class_results = [r for r in self.results if r['label_real'] == label]
            class_correct = sum(1 for r in class_results if r['is_correct'])

            # Verificar si esta clase está en el modelo Inner Speech
            label_lower = str(label).lower()
            is_inner_class = label_lower in inner_classes_lower

            by_class[label] = {
                "total": len(class_results),
                "correct": class_correct,
                "accuracy": class_correct / len(class_results) if class_results else 0.0,
                "is_inner_class": is_inner_class  # Marca si el modelo Inner puede predecir esta clase
            }

        return {
            "total_windows": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "p300_detected": p300_detected,
            "p300_detection_rate": p300_detected / total if total > 0 else 0.0,
            "by_class": by_class,
            "inner_classes": inner_classes  # Lista de clases que Inner Speech conoce
        }

    # ========================================================================
    # Métodos Helper Privados
    # ========================================================================

    def _get_majority_label(self, start: int, end: int) -> str:
        """
        Obtiene la label más común en el rango de samples y la mapea al nombre de clase correcto.

        Args:
            start: Sample inicial
            end: Sample final

        Returns:
            Label más frecuente mapeada a nombre de clase (str)
        """
        window_labels = self.labels[start:end]

        # Contar ocurrencias
        unique, counts = np.unique(window_labels, return_counts=True)

        # Retornar el más frecuente
        majority_idx = np.argmax(counts)
        raw_label = unique[majority_idx]

        # Mapear label al nombre de clase correcto
        if self.label_to_class_map:
            # Intentar mapear como número
            if isinstance(raw_label, (int, np.integer)):
                mapped_label = self.label_to_class_map.get(int(raw_label))
                if mapped_label:
                    return str(mapped_label)

            # Intentar mapear como string
            mapped_label = self.label_to_class_map.get(str(raw_label))
            if mapped_label:
                return str(mapped_label)

            # Si dice "unlabeled" o valores desconocidos, mapear a "rest"
            if str(raw_label).lower() in ['unlabeled', 'unknown', 'none', '']:
                return "rest"

        # Si no hay mapeo disponible, retornar tal cual
        return str(raw_label)

    def _evaluate_prediction(
        self,
        p300_pred: int,
        inner_pred: Optional[int],
        label_real: str
    ) -> bool:
        """
        Evalúa si la predicción completa es correcta.

        Lógica:
        - Si label_real == 'rest': correcto si p300_pred == 0
        - Si label_real != 'rest': correcto si p300_pred == 1 Y inner_pred == label_real

        Args:
            p300_pred: Predicción del modelo P300 (0 o 1)
            inner_pred: Predicción del modelo Inner (índice de clase) o None
            label_real: Label real (str)

        Returns:
            True si la predicción es correcta
        """
        # Caso 1: Baseline/rest
        if label_real == 'rest':
            # Correcto si NO detectó P300
            return p300_pred == 0

        # Caso 2: Clase activa (arriba, abajo, etc.)
        else:
            # ===== FIX: Inner Speech NO entrena con "rest" =====
            # Si label_real no está en las clases del modelo, NO es evaluable
            classes = self.inner_config['model_metadata'].get('classes', [])

            # Normalizar a lowercase para comparación
            classes_lower = [str(c).lower() for c in classes]
            label_real_lower = str(label_real).lower()

            # Si el label real no está en las clases del modelo (ej: "rest", "none", "unlabeled")
            if label_real_lower not in classes_lower:
                # Esta ventana NO es evaluable para Inner Speech
                # Porque el modelo nunca fue entrenado con esta clase
                # Solo verificar que P300 detectó correctamente (debería predecir 0 para "rest")
                # Si P300 predijo 0, es correcto (no intentó clasificar con Inner)
                # Si P300 predijo 1, es incorrecto (falsa alarma de P300)
                return p300_pred == 0

            # Clase activa que SÍ está en el modelo
            # Debe detectar P300 Y clasificar correctamente
            if p300_pred != 1:
                return False  # No detectó P300 cuando debería

            if inner_pred is None:
                return False  # P300 detectado pero Inner no predijo

            # Mapear índice de clase a nombre
            if inner_pred < len(classes):
                predicted_class = classes[inner_pred]
                return str(predicted_class).lower() == label_real_lower
            else:
                # Índice fuera de rango
                print(f"[_evaluate_prediction] WARNING: inner_pred={inner_pred} >= len(classes)={len(classes)}")
                return False

    def _error_result(
        self,
        window_idx: int,
        start: int,
        end: int,
        time_sec: float,
        error_msg: str
    ) -> Dict:
        """
        Crea resultado de error cuando falla el procesamiento.
        """
        label_real = self._get_majority_label(start, end)

        return {
            "window_idx": window_idx,
            "start_sample": start,
            "end_sample": end,
            "time_sec": time_sec,
            "p300_prediction": -1,  # Indicador de error
            "p300_confidence": 0.0,
            "p300_probabilities": [],
            "inner_prediction": None,
            "inner_confidence": None,
            "inner_probabilities": None,
            "label_real": label_real,
            "is_correct": False,
            "error": error_msg
        }

    def get_window_result(self, window_idx: int) -> Optional[Dict]:
        """
        Obtiene resultado de una ventana específica (si ya fue procesada).

        Args:
            window_idx: Índice de la ventana

        Returns:
            Dict con resultado o None si no fue procesada
        """
        if window_idx < len(self.results):
            return self.results[window_idx]
        return None
