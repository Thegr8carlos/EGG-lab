from __future__ import annotations
from typing import List, Literal, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, Field, field_validator
import pickle
from pathlib import Path
from datetime import datetime

# ===== Tipos =====
NDArray = np.ndarray
ActName = Literal["relu", "tanh", "sigmoid", "gelu", "softmax", "linear"]
PoolName = Literal["last", "mean", "max", "attn"]
import time

from backend.classes.Metrics import EvaluationMetrics
from backend.classes.ClasificationModel.utils.RecurrentModelDataUtils import RecurrentModelDataUtils
from backend.classes.ClasificationModel.utils.TrainResult import TrainResult

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# =====================================================================================
# 1) Activación y capas densas
# =====================================================================================
class ActivationFunction(BaseModel):
    kind: ActName = Field("relu", description="Tipo de activación.")

class DenseLayer(BaseModel):
    units: int = Field(..., ge=1, description="Número de neuronas.")
    activation: ActivationFunction = Field(default_factory=lambda: ActivationFunction(kind="relu"))

    @field_validator("units")
    @classmethod
    def _v_units(cls, v: int) -> int:
        if v < 1:
            raise ValueError("DenseLayer.units debe ser >= 1.")
        return v

# =====================================================================================
# 2) Definición de una capa LSTM (lógica)
# =====================================================================================
class LSTMLayer(BaseModel):
    hidden_size: int = Field(..., ge=1, description="Dimensión oculta por dirección.")
    bidirectional: bool = Field(False, description="Usar bidireccional en esta capa.")
    num_layers: int = Field(1, ge=1, le=10, description="Pilas internas de LSTM en esta capa.")
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout entre capas internas (solo si num_layers>1).")
    recurrent_dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout en conexiones recurrentes.")
    return_sequences: bool = Field(True, description="Si True, devolver secuencia (T,D); si False, solo el último paso.")
    use_bias: bool = Field(True, description="Usar términos bias en LSTM.")
    kernel_initializer: str = Field("glorot_uniform", description="Inicializador de pesos del kernel.")
    recurrent_initializer: str = Field("orthogonal", description="Inicializador de pesos recurrentes.")
    unit_forget_bias: bool = Field(True, description="Inicializar bias del forget gate a 1 (recomendado).")
    # TensorFlow no soporta proj_size directamente, pero lo dejamos para compatibilidad de schema
    proj_size: int = Field(0, ge=0, description="Tamaño de proyección (0=sin proyección). Nota: TensorFlow no lo soporta nativamente.")

    def output_dim(self, input_dim: int) -> Tuple[int, bool]:
        """
        Calcula (feature_dim_salida, es_secuencia?)
        D = (proj_size si >0, si no hidden_size) * (2 si bidi else 1)
        Nota: proj_size no se usa en TensorFlow, solo hidden_size
        """
        base = self.proj_size if self.proj_size > 0 else self.hidden_size
        d = base * (2 if self.bidirectional else 1)
        return d, bool(self.return_sequences)

# =====================================================================================
# 3) Encoder secuencial: lista de capas LSTM
# =====================================================================================
class SequenceEncoder(BaseModel):
    input_feature_dim: int = Field(..., ge=1, description="Dim por paso temporal en la entrada.")
    layers: List[LSTMLayer] = Field(..., description="Pila de capas LSTM (una o más).")
    use_packed_sequences: bool = Field(True, description="Usar pack_padded_sequence para longitudes variables.")
    pad_value: float = Field(0.0, description="Valor de padding temporal.")
    enforce_sorted: bool = Field(False, description="Exige batch ordenado por longitud para packing.")

    @field_validator("layers")
    @classmethod
    def _v_layers(cls, layers: List[LSTMLayer]) -> List[LSTMLayer]:
        if len(layers) == 0:
            raise ValueError("SequenceEncoder.layers no puede estar vacío.")
        return layers

    def infer_output_signature(self) -> Tuple[int, bool]:
        """
        Propaga dims capa a capa para saber si sale (T,D) o (D,) y cuál es D.
        """
        d_in = int(self.input_feature_dim)
        is_seq = True
        for layer in self.layers:
            d_out, rs = layer.output_dim(d_in)
            d_in = d_out
            is_seq = rs
        return d_in, is_seq

# =====================================================================================
# 4) Pooling temporal
# =====================================================================================
class TemporalPooling(BaseModel):
    kind: PoolName = Field("last", description="Reducción temporal: 'last'|'mean'|'max'|'attn'.")
    attn_hidden: Optional[int] = Field(64, ge=1, description="Hidden para atención (si kind='attn').")

# =====================================================================================
# 5) Modelo LSTM completo: Encoder + Pooling + FC + Softmax
# =====================================================================================
class LSTMNet(BaseModel):
    # Encoder
    encoder: SequenceEncoder = Field(..., description="Bloque secuencial (LSTM apiladas).")

    # Pooling tras el encoder (si el encoder ya devuelve vector, pooling es no-op)
    pooling: TemporalPooling = Field(default_factory=lambda: TemporalPooling(kind="last"))

    # Fully connected intermedias con activación común
    fc_layers: List[DenseLayer] = Field(default_factory=list, description="Capas densas intermedias.")
    fc_activation_common: ActivationFunction = Field(
        default_factory=lambda: ActivationFunction(kind="relu"),
        description="Activación común impuesta a todas las fc_layers."
    )

    # Capa de clasificación
    classification: DenseLayer = Field(..., description="Salida 'softmax' con units=n_clases.")

    @field_validator("fc_layers")
    @classmethod
    def _v_fc(cls, layers, info):
        values = info.data
        common: ActivationFunction = values.get("fc_activation_common", ActivationFunction(kind="relu"))
        for i, lyr in enumerate(layers):
            lyr.activation = common
            layers[i] = lyr
        return layers

    @field_validator("classification")
    @classmethod
    def _v_softmax(cls, lyr: DenseLayer) -> DenseLayer:
        if lyr.activation.kind != "softmax":
            raise ValueError("La capa de clasificación DEBE usar activación 'softmax'.")
        return lyr

    # ------- Dimensión “estática” (útil para debug) -------
    def feature_dim_after_encoder(self) -> int:
        d, is_seq = self.encoder.infer_output_signature()
        return int(d)

    def flatten_dim(self) -> int:
        return self.feature_dim_after_encoder()

    # Objeto de modelo entrenado (keras.Model) para query posterior (se llena en fit)
    _tf_model: Optional[object] = None  # usar 'object' para no forzar import keras al parsear schema

    # =================================================================================
    # Persistencia: save/load
    # =================================================================================
    def save(self, path: str):
        """
        Guarda la instancia completa (configuración + modelo entrenado) a disco.
        
        Args:
            path: Ruta completa donde guardar el archivo .pkl
                  Ejemplo: "src/backend/models/p300/lstm_20251109_143022.pkl"
        
        Note:
            Usa pickle para serializar toda la instancia incluyendo _tf_model.
            El directorio padre se crea automáticamente si no existe.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[LSTM] Modelo guardado en: {path}")
    
    @classmethod
    def load(cls, path: str) -> "LSTMNet":
        """
        Carga una instancia completa desde disco.
        
        Args:
            path: Ruta al archivo .pkl guardado previamente
        
        Returns:
            Instancia de LSTMNet con modelo entrenado listo para query()
        
        Example:
            lstm_model = LSTMNet.load("src/backend/models/p300/lstm_20251109_143022.pkl")
            predictions = LSTMNet.query(lstm_model, sequences)
        """
        with open(path, 'rb') as f:
            instance = pickle.load(f)
        print(f"[LSTM] Modelo cargado desde: {path}")
        return instance
    
    @staticmethod
    def _generate_model_path(label: str, base_dir: str = "src/backend/models") -> str:
        """
        Genera ruta única para guardar modelo.
        
        Args:
            label: Etiqueta del experimento (e.g., "p300", "inner")
            base_dir: Directorio base para modelos
        
        Returns:
            Ruta completa: "{base_dir}/{label}/lstm_{timestamp}.pkl"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lstm_{timestamp}.pkl"
        return str(Path(base_dir) / label / filename)

    # =================================================================================
    # Helpers de metadatos
    # =================================================================================
    @staticmethod
    def extract_metadata_from_experiment(experiment_dict: dict, transform_indices: Optional[List[int]] = None) -> List[dict]:
        """
        Extrae metadatos de dimensionality_change desde un diccionario de Experiment.

        Args:
            experiment_dict: Diccionario con estructura de Experiment (output de experiment.dict())
            transform_indices: Índices de las transformadas a extraer. Si None, extrae todas.

        Returns:
            Lista de diccionarios con metadatos de dimensionality_change

        Ejemplo:
            experiment = Experiment._load_latest_experiment()
            metadata = LSTMNet.extract_metadata_from_experiment(experiment.dict(), [0, 1])
        """
        transforms = experiment_dict.get("transform", [])
        if not transforms:
            return []

        if transform_indices is None:
            transform_indices = list(range(len(transforms)))

        metadata_list = []
        for idx in transform_indices:
            if idx < 0 or idx >= len(transforms):
                raise IndexError(f"Índice de transform fuera de rango: {idx}")

            transform_entry = transforms[idx]
            dim_change = transform_entry.get("dimensionality_change", {})

            # Crear diccionario de metadatos con estructura estándar
            metadata = {
                "output_axes_semantics": dim_change.get("output_axes_semantics", {}),
                "output_shape": dim_change.get("output_shape"),
                "input_shape": dim_change.get("input_shape"),
                "standardized_to": dim_change.get("standardized_to"),
                "transposed_from_input": dim_change.get("transposed_from_input"),
                "orig_was_1d": dim_change.get("orig_was_1d"),
            }
            metadata_list.append(metadata)

        return metadata_list

    @staticmethod
    def _interpret_metadata(metadata: dict) -> Tuple[str, bool]:
        """
        Interpreta metadatos de dimensionality_change para determinar forma y orientación.

        Args:
            metadata: Diccionario con:
                - output_axes_semantics: {"axis0": "time", "axis1": "channels"} o similar
                - output_shape: (T, F) o (F, T)
                - transposed_from_input: bool indicando si se transpuso

        Returns:
            Tupla (axis_order, needs_transpose):
                - axis_order: "time_first" si (T,F) o "features_first" si (F,T)
                - needs_transpose: True si necesita transposición para llegar a (T,F)
        """
        semantics = metadata.get("output_axes_semantics") or metadata.get("axes_semantics") or {}
        output_shape = metadata.get("output_shape") or metadata.get("shape")
        transposed_from_input = metadata.get("transposed_from_input")

        if not semantics and not output_shape:
            raise ValueError("metadata debe contener 'output_axes_semantics' o 'output_shape'")

        # Extraer significado de ejes
        axis_meanings = {}
        for axis_key, meaning in semantics.items():
            axis_idx = int(axis_key.replace("axis", "")) if "axis" in axis_key else -1
            if axis_idx >= 0:
                axis_meanings[axis_idx] = str(meaning).lower()

        # Determinar orientación
        if 0 in axis_meanings and 1 in axis_meanings:
            axis0_meaning = axis_meanings[0]
            axis1_meaning = axis_meanings[1]

            if "time" in axis0_meaning or "sample" in axis0_meaning or "frame" in axis0_meaning:
                # Eje 0 es tiempo -> (T, F) formato correcto
                return "time_first", False
            elif "channel" in axis1_meaning or "feature" in axis1_meaning or "freq" in axis1_meaning:
                # Eje 1 es features, eje 0 debe ser tiempo -> (T, F) formato correcto
                return "time_first", False
            elif "channel" in axis0_meaning or "feature" in axis0_meaning or "freq" in axis0_meaning:
                # Eje 0 es features -> (F, T) necesita transposición
                return "features_first", True
            elif "time" in axis1_meaning or "sample" in axis1_meaning or "frame" in axis1_meaning:
                # Eje 1 es tiempo -> (F, T) necesita transposición
                return "features_first", True

        # Fallback: usar output_shape si existe
        if output_shape and len(output_shape) == 2:
            T_or_F, F_or_T = output_shape
            # Heurística: si primera dim >> segunda, probablemente es (T, F)
            if T_or_F > F_or_T:
                return "time_first", False
            else:
                return "features_first", True

        # Fallback final
        return "time_first", False

    # =================================================================================
    # Dataset helpers: clasificación por SECUENCIA (una etiqueta por archivo)
    # =================================================================================
    @staticmethod
    def _load_sequence(path: str, metadata: Optional[dict] = None) -> NDArray:
        """
        Carga secuencia desde archivo .npy usando RecurrentModelDataUtils.

        Wrapper que agrega interpretación de metadatos específica de LSTM.

        Args:
            path: Ruta al archivo .npy
            metadata: Diccionario opcional con metadatos de dimensionality_change

        Returns:
            Array 2D de forma (T, F)
        """
        # Cargar usando utilidades compartidas
        X = RecurrentModelDataUtils.load_sequence(path, metadata, model_name="LSTM")

        # Si hay metadatos 2D, aplicar interpretación específica de LSTM
        if X.ndim == 2 and metadata and (metadata.get("output_axes_semantics") or metadata.get("output_shape")):
            _, needs_transpose = LSTMNet._interpret_metadata(metadata)
            if needs_transpose:
                X = X.T

        return X

    @staticmethod
    def _load_label_scalar(path: str) -> int:
        """
        Carga etiqueta escalar usando RecurrentModelDataUtils.

        Args:
            path: Ruta al archivo .npy con etiquetas

        Returns:
            Etiqueta más frecuente (moda) como entero
        """
        return RecurrentModelDataUtils.load_label_scalar(path)

    @classmethod
    def _prepare_sequences_and_labels(
        cls,
        x_paths: Sequence[str],
        y_paths: Sequence[str],
        pad_value: float = 0.0,
        metadata_list: Optional[Sequence[dict]] = None,
    ):
        """
        Prepara secuencias y etiquetas usando RecurrentModelDataUtils.

        Devuelve:
          - sequences: lista de arrays (Ti, F)
          - lengths: (N,) int64
          - y: (N,) int64

        Args:
            x_paths: Rutas a archivos .npy con secuencias
            y_paths: Rutas a archivos .npy con etiquetas
            pad_value: Valor de padding para secuencias variables
            metadata_list: Lista opcional de diccionarios con metadatos de dimensionality_change.
        """
        return RecurrentModelDataUtils.prepare_sequences_and_labels(
            x_paths=x_paths,
            y_paths=y_paths,
            pad_value=pad_value,
            metadata_list=metadata_list,
            load_sequence_func=cls._load_sequence  # Usa la versión LSTM con interpretación de metadatos
        )

    # =================================================================================
    # Entrenamiento: TensorFlow/Keras con soporte de metadatos
    # =================================================================================
    @classmethod
    def train(
        cls,
        instance: "LSTMNet",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
        metadata_train: Optional[List[dict]] = None,
        metadata_test: Optional[List[dict]] = None,
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 1e-3,
        model_label: Optional[str] = None,
    ):
        """
        Entrena un modelo LSTM usando TensorFlow/Keras (wrapper legacy).
        
        Este método mantiene compatibilidad con código existente retornando solo EvaluationMetrics.
        Internamente delega a fit() para evitar duplicación de código.

        Args:
            instance: Instancia de LSTMNet con arquitectura configurada
            xTest: Lista de rutas a archivos .npy de test
            yTest: Lista de rutas a archivos .npy con etiquetas de test
            xTrain: Lista de rutas a archivos .npy de entrenamiento
            yTrain: Lista de rutas a archivos .npy con etiquetas de entrenamiento
            metadata_train: Lista de diccionarios con metadatos de dimensionality_change para train.
                           Cada diccionario debe contener:
                           - 'output_axes_semantics': dict con semántica de ejes
                           - 'output_shape': tuple con forma de salida
                           Ejemplo:
                           [{"output_axes_semantics": {"axis0": "time", "axis1": "channels"},
                             "output_shape": (1000, 64)}]
            metadata_test: Lista de diccionarios con metadatos para test
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño de batch
            lr: Learning rate
            model_label: Etiqueta opcional para auto-guardar (e.g., "p300", "inner").
                        Si se proporciona, guarda automáticamente en src/backend/models/{label}/
            
        Returns:
            EvaluationMetrics: Solo las métricas de evaluación (para compatibilidad backward)
            
        Note:
            Si necesitas acceso al modelo entrenado y más información, usa fit() en su lugar.
        """
        # Delegar a fit() y extraer solo métricas (evita duplicación de código)
        result = cls.fit(
            instance=instance,
            xTest=xTest,
            yTest=yTest,
            xTrain=xTrain,
            yTrain=yTrain,
            metadata_train=metadata_train,
            metadata_test=metadata_test,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            return_history=False,  # No necesitamos historia completa para legacy API
            model_label=model_label
        )
        return result.metrics

    # Nuevo método: devuelve TrainResult con el modelo entrenado y métricas
    @classmethod
    def fit(
        cls,
        instance: "LSTMNet",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
        metadata_train: Optional[List[dict]] = None,
        metadata_test: Optional[List[dict]] = None,
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 1e-3,
        return_history: bool = True,
        model_label: Optional[str] = None,
    ) -> TrainResult:
        """Entrena y devuelve paquete TrainResult (modelo + métricas + historia).

        No rompe `train()`: es una API paralela opt-in.

        Args:
            model_label: Etiqueta opcional para auto-guardar (e.g., "p300", "inner").
                        Si se proporciona, guarda automáticamente en src/backend/models/{label}/
        """
        try:
            print(f"[LSTM.fit] Preparando datos con enfoque memory-efficient...")
            print(f"  Train: {len(xTrain)} archivos")
            print(f"  Test:  {len(xTest)} archivos")

            # OPTIMIZACIÓN MEMORIA: Cargar solo una pequeña muestra para inferir dimensiones
            # En lugar de cargar TODOS los archivos, cargamos solo los primeros 20
            sample_size = min(20, len(xTrain), len(xTest))

            print(f"[LSTM.fit] Cargando muestra ({sample_size} archivos) para inferir dimensiones...")
            seq_tr_sample, len_tr_sample, y_tr_sample = cls._prepare_sequences_and_labels(
                xTrain[:sample_size], yTrain[:sample_size],
                pad_value=instance.encoder.pad_value,
                metadata_list=metadata_train[:sample_size] if metadata_train else None
            )
            seq_te_sample, len_te_sample, y_te_sample = cls._prepare_sequences_and_labels(
                xTest[:sample_size], yTest[:sample_size],
                pad_value=instance.encoder.pad_value,
                metadata_list=metadata_test[:sample_size] if metadata_test else None
            )

            # Inferir dimensiones globales
            max_len_tr = int(len_tr_sample.max())
            max_len_te = int(len_te_sample.max())
            max_len_global = max(max_len_tr, max_len_te)
            in_F = seq_tr_sample[0].shape[1]

            # Inferir número de clases (necesitamos cargar TODOS los labels)
            print(f"[LSTM.fit] Cargando etiquetas completas para inferir número de clases...")
            _, _, y_tr = cls._prepare_sequences_and_labels(
                xTrain, yTrain,
                pad_value=instance.encoder.pad_value,
                metadata_list=metadata_train
            )
            _, _, y_te = cls._prepare_sequences_and_labels(
                xTest, yTest,
                pad_value=instance.encoder.pad_value,
                metadata_list=metadata_test
            )

            print(f"[LSTM.fit] Dimensiones inferidas:")
            print(f"  max_len_global: {max_len_global}")
            print(f"  input_features: {in_F}")
            print(f"  train_samples:  {len(y_tr)}")
            print(f"  test_samples:   {len(y_te)}")
        except Exception as e:
            raise RuntimeError(f"Error preparando dataset LSTM: {e}") from e

        d_enc, is_seq = instance.encoder.infer_output_signature()
        n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1
        if d_enc < 1 or in_F < 1 or n_classes < 2:
            raise ValueError("Dimensionalidades/num clases inválidas.")

        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            # Limpiar modelo previo (cada fit() reconstruye desde cero)
            instance._tf_model = None

            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU config error: {e}")

            pad_value = instance.encoder.pad_value

            # =========================================================================
            # GENERADOR DE DATOS MEMORY-EFFICIENT
            # =========================================================================
            def create_data_generator(x_paths, y_labels, metadata, batch_size, shuffle=True):
                """
                Generador que carga batches bajo demanda para reducir uso de memoria.

                Args:
                    x_paths: Lista de rutas a archivos .npy
                    y_labels: Array numpy con etiquetas (ya cargado, es pequeño)
                    metadata: Lista de metadatos (puede ser None)
                    batch_size: Tamaño de batch
                    shuffle: Si True, mezcla los índices en cada época
                """
                n_samples = len(x_paths)
                indices = np.arange(n_samples)

                while True:  # Generador infinito para Keras
                    if shuffle:
                        np.random.shuffle(indices)

                    for start_idx in range(0, n_samples, batch_size):
                        end_idx = min(start_idx + batch_size, n_samples)
                        batch_indices = indices[start_idx:end_idx]

                        # Cargar batch bajo demanda
                        batch_x = []
                        batch_y = []

                        for idx in batch_indices:
                            # Cargar y procesar secuencia
                            seq = cls._load_sequence(
                                x_paths[idx],
                                metadata=metadata[idx] if metadata and idx < len(metadata) else None
                            )
                            batch_x.append(seq)
                            batch_y.append(y_labels[idx])

                        # Padding del batch
                        batch_size_actual = len(batch_x)
                        X_batch = np.full((batch_size_actual, max_len_global, in_F), pad_value, dtype=np.float32)

                        for i, seq in enumerate(batch_x):
                            T = min(seq.shape[0], max_len_global)
                            X_batch[i, :T, :] = seq[:T, :]

                        y_batch = np.array(batch_y, dtype=np.int64)

                        yield X_batch, y_batch

            # Calcular steps por época
            steps_per_epoch_train = int(np.ceil(len(xTrain) / batch_size))
            steps_per_epoch_val = int(np.ceil(len(xTest) / batch_size))

            print(f"[LSTM.fit] Configuración de generadores:")
            print(f"  steps_per_epoch (train): {steps_per_epoch_train}")
            print(f"  steps_per_epoch (val):   {steps_per_epoch_val}")
            print(f"  batch_size:              {batch_size}")

            # Crear generadores
            train_generator = create_data_generator(xTrain, y_tr, metadata_train, batch_size, shuffle=True)
            val_generator = create_data_generator(xTest, y_te, metadata_test, batch_size, shuffle=False)

            # Para evaluación final, necesitamos cargar test completo (es más pequeño)
            print(f"[LSTM.fit] Cargando test set completo para evaluación final...")
            seq_te_full, _, _ = cls._prepare_sequences_and_labels(
                xTest, yTest,
                pad_value=instance.encoder.pad_value,
                metadata_list=metadata_test
            )

            # Padding test set
            X_te_padded = np.full((len(seq_te_full), max_len_global, in_F), pad_value, dtype=np.float32)
            for i, seq in enumerate(seq_te_full):
                T = min(seq.shape[0], max_len_global)
                X_te_padded[i, :T, :] = seq[:T, :]

            del seq_te_full  # Liberar memoria
            import gc
            gc.collect()

            def build(spec: LSTMNet, input_shape: Tuple[int, int]) -> keras.Model:
                inputs = layers.Input(shape=input_shape, name='input')
                x = layers.Masking(mask_value=spec.encoder.pad_value)(inputs)
                for i, lstm_layer in enumerate(spec.encoder.layers):
                    return_sequences = lstm_layer.return_sequences or (i < len(spec.encoder.layers) - 1)
                    lstm = layers.LSTM(
                        units=lstm_layer.hidden_size,
                        return_sequences=return_sequences,
                        dropout=lstm_layer.dropout if lstm_layer.num_layers > 1 else 0.0,
                        recurrent_dropout=lstm_layer.recurrent_dropout,
                        kernel_initializer=lstm_layer.kernel_initializer,
                        recurrent_initializer=lstm_layer.recurrent_initializer,
                        use_bias=lstm_layer.use_bias,
                        unit_forget_bias=lstm_layer.unit_forget_bias,
                        name=f'lstm_{i}'
                    )
                    if lstm_layer.bidirectional:
                        x = layers.Bidirectional(lstm, name=f'bidirectional_lstm_{i}')(x)
                    else:
                        x = lstm(x)
                    for j in range(1, lstm_layer.num_layers):
                        inner_return_seq = return_sequences if j == lstm_layer.num_layers - 1 else True
                        lstm_inner = layers.LSTM(
                            units=lstm_layer.hidden_size,
                            return_sequences=inner_return_seq,
                            dropout=lstm_layer.dropout,
                            recurrent_dropout=lstm_layer.recurrent_dropout,
                            kernel_initializer=lstm_layer.kernel_initializer,
                            recurrent_initializer=lstm_layer.recurrent_initializer,
                            use_bias=lstm_layer.use_bias,
                            unit_forget_bias=lstm_layer.unit_forget_bias,
                            name=f'lstm_{i}_inner_{j}'
                        )
                        if lstm_layer.bidirectional:
                            x = layers.Bidirectional(lstm_inner, name=f'bidirectional_lstm_{i}_inner_{j}')(x)
                        else:
                            x = lstm_inner(x)
                if is_seq or instance.encoder.layers[-1].return_sequences:
                    pk = instance.pooling.kind
                    if pk == "last":
                        x = layers.Lambda(lambda t: t[:, -1, :], name='last_pooling')(x)
                    elif pk == "mean":
                        x = layers.GlobalAveragePooling1D(name='mean_pooling')(x)
                    elif pk == "max":
                        x = layers.GlobalMaxPooling1D(name='max_pooling')(x)
                    elif pk == "attn":
                        attn_hidden = instance.pooling.attn_hidden or 64
                        attn_scores = layers.Dense(attn_hidden, activation='tanh', name='attn_hidden')(x)
                        attn_scores = layers.Dense(1, name='attn_scores')(attn_scores)
                        attn_weights = layers.Softmax(axis=1, name='attn_weights')(attn_scores)
                        x = layers.Multiply(name='attn_multiply')([x, attn_weights])
                        x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name='attn_sum')(x)
                for i, fc in enumerate(instance.fc_layers):
                    x = layers.Dense(fc.units, name=f'fc_{i}')(x)
                    act = instance.fc_activation_common.kind
                    if act == "relu":
                        x = layers.ReLU(name=f'fc_{i}_relu')(x)
                    elif act == "tanh":
                        x = layers.Activation('tanh', name=f'fc_{i}_tanh')(x)
                    elif act == "gelu":
                        x = layers.Activation('gelu', name=f'fc_{i}_gelu')(x)
                    elif act == "sigmoid":
                        x = layers.Activation('sigmoid', name=f'fc_{i}_sigmoid')(x)
                outputs = layers.Dense(instance.classification.units, activation='softmax', name='classification')(x)
                return keras.Model(inputs=inputs, outputs=outputs, name='LSTMNet')

            # IMPORTANTE: Configurar GPU ANTES de construir el modelo
            import tensorflow as tf
            import gc

            # Limpiar sesión anterior
            tf.keras.backend.clear_session()
            gc.collect()

            # Configurar estrategia de entrenamiento con fallback GPU -> CPU
            def train_with_fallback():
                """Intenta entrenar en GPU, si falla cae a CPU automáticamente"""
                nonlocal batch_size

                # Fase 1: Intentar con GPU optimizada
                print("🚀 [LSTM] Intentando entrenamiento en GPU con optimizaciones...")

                # Configurar memory growth PRIMERO (antes de cualquier operación GPU)
                try:
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"   ✓ Memory growth habilitado en {len(gpus)} GPU(s)")
                except Exception as e:
                    print(f"   ⚠️ No se pudo configurar memory growth: {e}")

                # Habilitar Mixed Precision (usa menos memoria)
                try:
                    from tensorflow.keras import mixed_precision
                    policy = mixed_precision.Policy('mixed_float16')
                    mixed_precision.set_global_policy(policy)
                    print("   ✓ Mixed Precision (FP16) habilitado")
                except Exception as e:
                    print(f"   ⚠️ No se pudo habilitar Mixed Precision: {e}")

                # Reducir batch_size inicial para GPU
                gpu_batch_size = max(4, batch_size // 2)

                max_gpu_retries = 2
                model = None

                for attempt in range(max_gpu_retries):
                    try:
                        print(f"   Intento GPU {attempt + 1}/{max_gpu_retries} (batch_size={gpu_batch_size})")

                        # Recalcular steps si cambió batch_size
                        steps_train = int(np.ceil(len(xTrain) / gpu_batch_size))
                        steps_val = int(np.ceil(len(xTest) / gpu_batch_size))

                        # Recrear generadores con nuevo batch_size
                        train_gen = create_data_generator(xTrain, y_tr, metadata_train, gpu_batch_size, shuffle=True)
                        val_gen = create_data_generator(xTest, y_te, metadata_test, gpu_batch_size, shuffle=False)

                        # Construir modelo DESPUÉS de configurar GPU
                        model = build(instance, input_shape=(max_len_global, in_F))
                        model.compile(
                            optimizer=keras.optimizers.Adam(learning_rate=lr),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )

                        # Entrenar en GPU usando generador
                        t0 = time.perf_counter()
                        history = model.fit(
                            train_gen,
                            steps_per_epoch=steps_train,
                            epochs=epochs,
                            verbose=1,
                            validation_data=val_gen,
                            validation_steps=steps_val
                        )
                        train_time = time.perf_counter() - t0

                        print(f"✅ [LSTM] Entrenamiento en GPU completado exitosamente!")
                        print(f"   Tiempo: {train_time:.2f}s | Batch size: {gpu_batch_size}")
                        return history, train_time, gpu_batch_size, model, 'GPU'

                    except Exception as e:
                        error_str = str(e)
                        is_memory_error = any(keyword in error_str for keyword in [
                            "OOM", "RESOURCE_EXHAUSTED", "out of memory",
                            "Failed copying input tensor", "Dst tensor is not initialized",
                            "SameWorkerRecvDone", "unable to allocate"
                        ])

                        if is_memory_error and attempt < max_gpu_retries - 1:
                            # Reducir batch_size y reintentar
                            gpu_batch_size = max(2, gpu_batch_size // 2)
                            print(f"   ⚠️ OOM en GPU, reduciendo batch_size a {gpu_batch_size}")

                            tf.keras.backend.clear_session()
                            gc.collect()
                            continue
                        else:
                            # GPU falló, pasar a CPU
                            print(f"   ❌ GPU no disponible: {str(e)[:100]}")
                            break

                # Fase 2: Fallback a CPU
                print("🔄 [LSTM] Cambiando a entrenamiento en CPU...")
                print("   (Esto será más lento pero garantiza que complete)")

                # Limpiar configuración GPU
                tf.keras.backend.clear_session()
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy('float32')
                gc.collect()

                # Usar batch_size más pequeño también en CPU para evitar OOM de RAM
                cpu_batch_size = max(4, batch_size // 4)
                print(f"   Usando batch_size reducido en CPU: {cpu_batch_size}")

                # Recalcular steps para CPU
                steps_train_cpu = int(np.ceil(len(xTrain) / cpu_batch_size))
                steps_val_cpu = int(np.ceil(len(xTest) / cpu_batch_size))

                # Recrear generadores con batch_size reducido
                train_gen_cpu = create_data_generator(xTrain, y_tr, metadata_train, cpu_batch_size, shuffle=True)
                val_gen_cpu = create_data_generator(xTest, y_te, metadata_test, cpu_batch_size, shuffle=False)

                # Forzar uso de CPU
                with tf.device('/CPU:0'):
                    # Reconstruir modelo en CPU
                    model = build(instance, input_shape=(max_len_global, in_F))
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=lr),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )

                    # Entrenar en CPU con generador
                    t0 = time.perf_counter()
                    history = model.fit(
                        train_gen_cpu,
                        steps_per_epoch=steps_train_cpu,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_gen_cpu,
                        validation_steps=steps_val_cpu
                    )
                    train_time = time.perf_counter() - t0

                    print(f"✅ [LSTM] Entrenamiento en CPU completado exitosamente!")
                    print(f"   Tiempo: {train_time:.2f}s | Batch size: {cpu_batch_size}")
                    return history, train_time, cpu_batch_size, model, 'CPU'

            # Ejecutar entrenamiento con fallback
            history, train_time, final_batch_size, model, device_used = train_with_fallback()
            batch_size = final_batch_size  # Actualizar para predicción

            t_pred0 = time.perf_counter()
            # Predicción - USAR EL MISMO DISPOSITIVO que entrenamiento
            if device_used == 'CPU':
                with tf.device('/CPU:0'):
                    logits = model.predict(X_te_padded, batch_size=batch_size, verbose=0)
            else:
                logits = model.predict(X_te_padded, batch_size=batch_size, verbose=0)
            y_pred = np.argmax(logits, axis=1)
            y_true = y_te
            eval_seconds = time.perf_counter() - t_pred0

            acc  = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec  = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            f1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            cm   = confusion_matrix(y_true, y_pred).tolist()
            try:
                auc = float(roc_auc_score(y_true, logits, multi_class="ovr", average="weighted"))
            except Exception:
                auc = 0.0
            metrics = EvaluationMetrics(
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1_score=f1,
                confusion_matrix=cm,
                auc_roc=auc,
                loss=history.history.get('loss', []),
                evaluation_time=f"{eval_seconds:.4f}s",
            )
            instance._tf_model = model
            print(f"[LSTM.fit] Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")
            
            # Auto-guardar si se proporciona label
            if model_label:
                save_path = cls._generate_model_path(model_label)
                instance.save(save_path)
            
            return TrainResult(
                metrics=metrics,
                model=model,
                model_name="LSTM",
                training_seconds=float(train_time),
                history=history.history if return_history else None,
                hyperparams={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "n_classes": n_classes,
                }
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            dummy = EvaluationMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                confusion_matrix=[],
                auc_roc=0.0,
                loss=[],
                evaluation_time="",
            )
            return TrainResult(
                metrics=dummy,
                model=None,
                model_name="LSTM",
                training_seconds=0.0,
                history=None,
                hyperparams={"error": str(e)}
            )

    @classmethod
    def query(
        cls,
        instance: "LSTMNet",
        sequences: List[NDArray],
        pad_value: float = 0.0,
        return_logits: bool = False
    ):
        """Inferencia sobre lista de secuencias (cada secuencia (T,F))."""
        if instance._tf_model is None:
            raise RuntimeError("Modelo LSTM no entrenado: usa fit() antes de query().")
        if not sequences:
            return []
        max_len = max(seq.shape[0] for seq in sequences)
        F = sequences[0].shape[1]
        batch = np.full((len(sequences), max_len, F), pad_value, dtype=np.float32)
        for i, seq in enumerate(sequences):
            T = seq.shape[0]
            batch[i, :T, :] = seq
        probs = instance._tf_model.predict(batch, verbose=0)
        preds = np.argmax(probs, axis=1).tolist()
        if return_logits:
            return preds, probs.tolist()
        return preds



# Alias for backward compatibility
LSTM = LSTMNet


# ======================= Ejemplo de uso =======================
"""
# Ejemplo 1: Construcción básica de modelo LSTM

from backend.classes.ClasificationModel.LSTM import LSTMNet, LSTMLayer, SequenceEncoder, TemporalPooling, DenseLayer, ActivationFunction

# Configurar capas LSTM
lstm1 = LSTMLayer(
    hidden_size=128,
    bidirectional=True,
    dropout=0.2,
    recurrent_dropout=0.1,
    num_layers=1,
    return_sequences=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    unit_forget_bias=True
)

lstm2 = LSTMLayer(
    hidden_size=64,
    bidirectional=True,
    dropout=0.2,
    recurrent_dropout=0.1,
    num_layers=1,
    return_sequences=False
)

# Configurar encoder
encoder = SequenceEncoder(
    input_feature_dim=64,  # número de canales/features
    layers=[lstm1, lstm2],
    use_packed_sequences=True,  # no se usa en TF, solo para compatibilidad
    pad_value=0.0,
    enforce_sorted=False
)

# Configurar pooling
pooling = TemporalPooling(
    kind="last",  # opciones: "last", "mean", "max", "attn"
    attn_hidden=64
)

# Configurar capas densas finales
fc_layers = [
    DenseLayer(units=128),
    DenseLayer(units=64)
]

# Capa de clasificación (n_classes=5)
classification = DenseLayer(
    units=5,
    activation=ActivationFunction(kind="softmax")
)

# Construir modelo completo
lstm_net = LSTMNet(
    encoder=encoder,
    pooling=pooling,
    fc_layers=fc_layers,
    fc_activation_common=ActivationFunction(kind="relu"),
    classification=classification
)

# Ejemplo 2: Entrenamiento con metadatos (RECOMENDADO)
from backend.classes.Experiment import Experiment

# Extraer metadatos desde Experiment
experiment = Experiment._load_latest_experiment()
metadata_train = LSTMNet.extract_metadata_from_experiment(experiment.dict(), transform_indices=[0, 1])
metadata_test = LSTMNet.extract_metadata_from_experiment(experiment.dict(), transform_indices=[0])

# Entrenar modelo
metrics = LSTMNet.train(
    lstm_net,
    xTest=["path/to/test1.npy"],
    yTest=["path/to/test1_label.npy"],
    xTrain=["path/to/train1.npy", "path/to/train2.npy"],
    yTrain=["path/to/train1_label.npy", "path/to/train2_label.npy"],
    metadata_train=metadata_train,
    metadata_test=metadata_test,
    epochs=10,
    batch_size=32,
    lr=1e-3
)

# Ejemplo 3: Metadatos manuales
metadata_train = [
    {
        "output_axes_semantics": {"axis0": "time", "axis1": "channels"},
        "output_shape": (1000, 64),
        "transposed_from_input": False
    },
    {
        "output_axes_semantics": {"axis0": "time", "axis1": "channels"},
        "output_shape": (1500, 64),
        "transposed_from_input": False
    }
]

metadata_test = [
    {
        "output_axes_semantics": {"axis0": "time", "axis1": "channels"},
        "output_shape": (1200, 64)
    }
]

metrics = LSTMNet.train(
    lstm_net,
    xTest=["path/to/test1.npy"],
    yTest=["path/to/test1_label.npy"],
    xTrain=["path/to/train1.npy", "path/to/train2.npy"],
    yTrain=["path/to/train1_label.npy", "path/to/train2_label.npy"],
    metadata_train=metadata_train,
    metadata_test=metadata_test,
    epochs=10,
    batch_size=32,
    lr=1e-3
)

# Ejemplo 4: Sin metadatos (fallback heurístico)
metrics = LSTMNet.train(
    lstm_net,
    xTest=["path/to/test1.npy"],
    yTest=["path/to/test1_label.npy"],
    xTrain=["path/to/train1.npy", "path/to/train2.npy"],
    yTrain=["path/to/train1_label.npy", "path/to/train2_label.npy"],
    epochs=10,
    batch_size=32,
    lr=1e-3
)

# Ejemplo 5: Configuración con diferentes tipos de pooling

# Pooling con atención
pooling_attn = TemporalPooling(kind="attn", attn_hidden=128)

# Pooling promedio
pooling_mean = TemporalPooling(kind="mean")

# Pooling máximo
pooling_max = TemporalPooling(kind="max")

# Último paso (más común)
pooling_last = TemporalPooling(kind="last")

# Ejemplo 6: Configuración avanzada de LSTM
lstm_advanced = LSTMLayer(
    hidden_size=256,
    bidirectional=True,
    dropout=0.3,
    recurrent_dropout=0.2,
    num_layers=2,  # Apilar 2 LSTMs internas
    return_sequences=True,
    kernel_initializer="he_normal",
    recurrent_initializer="orthogonal",
    use_bias=True,
    unit_forget_bias=True
)
"""