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
        hop_percent: float = 50.0
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
        """
        self.raw_signal = raw_signal
        self.labels = labels.flatten() if labels.ndim > 1 else labels
        self.sfreq = sfreq
        self.p300_config = p300_model_config
        self.inner_config = inner_model_config

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

        # Métricas por clase
        unique_labels = set(r['label_real'] for r in self.results)
        by_class = {}

        for label in unique_labels:
            class_results = [r for r in self.results if r['label_real'] == label]
            class_correct = sum(1 for r in class_results if r['is_correct'])

            by_class[label] = {
                "total": len(class_results),
                "correct": class_correct,
                "accuracy": class_correct / len(class_results) if class_results else 0.0
            }

        return {
            "total_windows": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "p300_detected": p300_detected,
            "p300_detection_rate": p300_detected / total if total > 0 else 0.0,
            "by_class": by_class
        }

    # ========================================================================
    # Métodos Helper Privados
    # ========================================================================

    def _get_majority_label(self, start: int, end: int) -> str:
        """
        Obtiene la label más común en el rango de samples.

        Args:
            start: Sample inicial
            end: Sample final

        Returns:
            Label más frecuente (str)
        """
        window_labels = self.labels[start:end]

        # Contar ocurrencias
        unique, counts = np.unique(window_labels, return_counts=True)

        # Retornar el más frecuente
        majority_idx = np.argmax(counts)
        return str(unique[majority_idx])

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
            # Debe detectar P300 Y clasificar correctamente
            if p300_pred != 1:
                return False  # No detectó P300 cuando debería

            if inner_pred is None:
                return False  # P300 detectado pero Inner no predijo

            # Mapear índice de clase a nombre
            # Nota: Esto depende del orden de clases en el modelo Inner
            classes = self.inner_config['model_metadata'].get('classes', [])

            # NO filtrar 'rest' porque el modelo fue entrenado con ella
            # y los índices de predicción corresponden a la lista completa
            if inner_pred < len(classes):
                predicted_class = classes[inner_pred]
                return predicted_class == label_real
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
