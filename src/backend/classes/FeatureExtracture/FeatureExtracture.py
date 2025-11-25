from pydantic import BaseModel, Field
from typing import Optional, Tuple, Dict
import numpy as np

# ---------------------------- BASE ----------------------------

class Transform(BaseModel):
    sp: float  # puntos por segundo
    id: str   # identificador único (dentro del experimento)
    model_type: Optional[str] = Field(
        None,
        description="Tipo de modelo: 'p300' (binario 0/1) o 'inner' (multiclase desde 1). Si None, no se re-etiqueta."
    )
    all_classes: Optional[list] = Field(
        None,
        description="Lista de todas las clases posibles del dataset (para mapeo consistente en multiclase)"
    )

    def get_sp(self) -> float:
        return self.sp

    def get_id(self) -> str:
        return self.id

    def relabel_for_model(self, labels_array: np.ndarray, all_classes: Optional[list] = None) -> Tuple[np.ndarray, Dict[int, str], Optional[np.ndarray]]:
        """
        Re-etiqueta ventanas a formato numérico según model_type.

        Args:
            labels_array: Array de etiquetas string (e.g., ["Target", "rest", "word_1", ...])
            all_classes: Lista opcional de todas las clases posibles del dataset (para mapeo consistente en multiclase)

        Returns:
            Tuple[np.ndarray, Dict[int, str], Optional[np.ndarray]]:
                - Array numérico de etiquetas (int)
                - Diccionario de mapeo {id: label_string}
                - Máscara booleana de muestras válidas (None si no se filtró nada)

        Reglas:
            - None: Mapeo directo de clases únicas a IDs (0, 1, 2...)
            - "p300": Binario → 0 (NonTarget/rest), 1 (Target)
            - "inner": Multiclase → IDs desde 0, EXCLUYENDO rest/none/unlabeled
        """
        # Asegurar que sea array numpy de strings
        if not isinstance(labels_array, np.ndarray):
            labels_array = np.array(labels_array, dtype=str)

        # Caso: Sin model_type configurado
        if self.model_type is None:
            # Mapeo simple: clases únicas → IDs desde 0
            unique_classes = sorted(set(labels_array))
            class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
            id_to_class = {idx: cls for cls, idx in class_to_id.items()}

            numeric_labels = np.array([class_to_id[label] for label in labels_array], dtype=int)
            return numeric_labels, id_to_class, None  # No se filtra nada

        # Caso: P300 - Etiquetado binario (0/1)
        if self.model_type.lower() == "p300":
            # Arquitectura de 2 etapas:
            #   0 = NonTarget/rest (sin intención)
            #   1 = Target (con intención: abajo, arriba, derecha, izquierda, etc.)
            # Inicializar todas como Target (1) por defecto
            numeric_labels = np.ones(len(labels_array), dtype=int)

            # Marcar como NonTarget (0) solo "rest" y variantes
            # (case-insensitive, también acepta None/none)
            labels_lower = np.char.lower(labels_array.astype(str))
            nontarget_mask = (labels_lower == "rest") | (labels_lower == "none")
            numeric_labels[nontarget_mask] = 0

            id_to_class = {0: "NonTarget", 1: "Target"}

            n_nontarget = np.sum(nontarget_mask)
            n_target = len(labels_array) - n_nontarget
            print(f"[Transform.relabel] P300 binario: {n_target} Target (1), {n_nontarget} NonTarget (0)")
            return numeric_labels, id_to_class, None  # No se filtra nada, se mapean a 0/1

        # Caso: Inner Speech - Multiclase (desde 1, sin 0)
        elif self.model_type.lower() == "inner":
            # Usar all_classes si está disponible para mapeo consistente entre eventos
            # De lo contrario, usar las clases únicas del array actual
            if all_classes is not None:
                unique_classes = sorted(set(all_classes))
            else:
                unique_classes = sorted(set(labels_array))

            # ===== FILTRAR CLASES NO DESEADAS PARA INNER SPEECH =====
            # Inner Speech solo clasifica PENSAMIENTOS (palabras mentales)
            # Excluir: rest, none, unlabeled (son estados baseline, no pensamientos)
            excluded_classes = ["rest", "none", "unlabeled"]
            unique_classes = [cls for cls in unique_classes
                            if cls.lower() not in excluded_classes]

            if len(unique_classes) == 0:
                raise ValueError("[Inner Speech] No quedan clases válidas después de filtrar 'rest'")

            print(f"[Transform.relabel] Inner Speech: {len(unique_classes)} clases (excluyendo rest/none/unlabeled)")

            # Mapeo 0-indexed para compatibilidad con sparse_categorical_crossentropy
            class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
            id_to_class = {idx: cls for cls, idx in class_to_id.items()}

            # Identificar muestras válidas (excluir rest/none/unlabeled)
            labels_lower = np.char.lower(labels_array.astype(str))
            valid_mask = ~np.isin(labels_lower, excluded_classes)
            filtered_labels = labels_array[valid_mask]

            if len(filtered_labels) == 0:
                raise ValueError("[Inner Speech] Todas las muestras son 'rest' - no hay datos de pensamientos")

            # Crear numeric_labels solo con las muestras válidas
            numeric_labels = np.array([class_to_id[label] for label in filtered_labels], dtype=int)

            # Solo contar las clases presentes en filtered_labels
            present_classes = set(filtered_labels)
            class_counts = {f"{cls} ({class_to_id[cls]})": np.sum(filtered_labels == cls) for cls in present_classes}
            print(f"[Transform.relabel] Inner Speech multiclase: {len(unique_classes)} clases totales (IDs desde 0) → presentes: {class_counts}")
            print(f"[Transform.relabel] ⚠️  Se filtrarán {len(labels_array) - len(filtered_labels)} muestras de rest/none/unlabeled")

            # Retornar numeric_labels (ya filtrado), mapeo, y máscara para filtrar datos X
            return numeric_labels, id_to_class, valid_mask

        else:
            raise ValueError(
                f"model_type desconocido: '{self.model_type}'. "
                f"Use 'p300' (binario 0/1) o 'inner' (multiclase desde 0)."
            )









# if __name__ == "__main__":
#     from pprint import pprint

#     print("\n🎯 Esquemas de Transformadas:")
#     pprint(TransformSchemaFactory.get_all_transform_schemas())
