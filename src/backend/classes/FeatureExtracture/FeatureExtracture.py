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

    def get_sp(self) -> float:
        return self.sp

    def get_id(self) -> str:
        return self.id

    def relabel_for_model(self, labels_array: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Re-etiqueta ventanas a formato numérico según model_type.

        Args:
            labels_array: Array de etiquetas string (e.g., ["Target", "rest", "word_1", ...])

        Returns:
            Tuple[np.ndarray, Dict[int, str]]:
                - Array numérico de etiquetas (int)
                - Diccionario de mapeo {id: label_string}

        Reglas:
            - None: Mapeo directo de clases únicas a IDs (0, 1, 2...)
            - "p300": Binario → 0 (NonTarget/rest), 1 (Target)
            - "inner": Multiclase → 1, 2, 3... (sin usar 0, clases ordenadas alfabéticamente)
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
            return numeric_labels, id_to_class

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
            return numeric_labels, id_to_class

        # Caso: Inner Speech - Multiclase (desde 1, sin 0)
        elif self.model_type.lower() == "inner":
            # Mapeo: cada clase única → ID desde 1 (alfabéticamente)
            unique_classes = sorted(set(labels_array))
            class_to_id = {cls: idx + 1 for idx, cls in enumerate(unique_classes)}  # +1 para empezar desde 1
            id_to_class = {idx: cls for cls, idx in class_to_id.items()}

            numeric_labels = np.array([class_to_id[label] for label in labels_array], dtype=int)

            class_counts = {f"{cls} ({class_to_id[cls]})": np.sum(labels_array == cls) for cls in unique_classes}
            print(f"[Transform.relabel] Inner Speech multiclase: {len(unique_classes)} clases (IDs desde 1) → {class_counts}")
            return numeric_labels, id_to_class

        else:
            raise ValueError(
                f"model_type desconocido: '{self.model_type}'. "
                f"Use 'p300' (binario 0/1) o 'inner' (multiclase desde 1)."
            )









# if __name__ == "__main__":
#     from pprint import pprint

#     print("\n🎯 Esquemas de Transformadas:")
#     pprint(TransformSchemaFactory.get_all_transform_schemas())
