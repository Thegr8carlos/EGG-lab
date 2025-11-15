from __future__ import annotations
from typing import List, Literal, Optional, Sequence, Tuple, Union
import math
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
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

# ===== Tipos =====
NDArray = np.ndarray
ActName = Literal["relu", "tanh", "sigmoid", "gelu", "softmax", "linear"]
PoolName = Literal["last", "mean", "max", "attn"]

# =====================================================================================
# 1) Activación y capas densas (comparten el patrón con tu CNN)
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
# 2) Definición de una capa GRU
# =====================================================================================
class GRULayer(BaseModel):
    hidden_size: int = Field(..., ge=1, description="Dimensión oculta por dirección.")
    bidirectional: bool = Field(False, description="Usar bidireccional en esta capa.")
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout entre capas (solo efectivo si num_layers>1).")
    recurrent_dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout en conexiones recurrentes.")
    num_layers: int = Field(1, ge=1, le=10, description="Pilas internas de GRU en esta capa lógica.")
    return_sequences: bool = Field(True, description="Si True, devolver secuencias (T,H); si False, solo el último paso.")
    kernel_initializer: str = Field("glorot_uniform", description="Inicializador de pesos del kernel.")
    recurrent_initializer: str = Field("orthogonal", description="Inicializador de pesos recurrentes.")
    use_bias: bool = Field(True, description="Si True, usa bias en las capas GRU.")
    reset_after: bool = Field(True, description="Aplicar reset gate después de multiplicación matricial (GRU moderno).")

    def output_dim(self, input_dim: int) -> Tuple[int, bool]:
        """
        Devuelve (feature_dim_salida, devuelve_secuencia?)
        - Si return_sequences=True -> (D, True)
        - Si return_sequences=False -> (D, False)
        Donde D = hidden_size * (2 si bidir else 1).
        """
        d = self.hidden_size * (2 if self.bidirectional else 1)
        return d, bool(self.return_sequences)

# =====================================================================================
# 3) Encoder secuencial: lista de capas GRU + opciones de empaquetado
# =====================================================================================
class SequenceEncoder(BaseModel):
    input_feature_dim: int = Field(..., ge=1, description="Dimensionalidad por paso temporal en la entrada.")
    layers: List[GRULayer] = Field(..., description="Pila de capas GRU (una o más).")
    use_packed_sequences: bool = Field(True, description="Usar pack_padded_sequence (PyTorch) para longitudes variables.")
    pad_value: float = Field(0.0, description="Valor de padding en tiempo.")

    @field_validator("layers")
    @classmethod
    def _v_layers(cls, layers: List[GRULayer]) -> List[GRULayer]:
        if len(layers) == 0:
            raise ValueError("SequenceEncoder.layers no puede estar vacío.")
        return layers

    def infer_output_signature(self) -> Tuple[int, bool]:
        """
        Propaga dim a través de las capas para predecir si sale (T,D) o (D,)
        y cuál es D.
        """
        d_in = int(self.input_feature_dim)
        is_seq = True  # la entrada es secuencia (T,d)
        for layer in self.layers:
            d_out, rs = layer.output_dim(d_in)
            d_in = d_out
            is_seq = rs
        return d_in, is_seq  # (feature_dim_final, devuelve_secuencia?)

# =====================================================================================
# 4) Pooling temporal: last/mean/max/attention (simple)
# =====================================================================================
class TemporalPooling(BaseModel):
    kind: PoolName = Field("last", description="Reducción temporal a vector: 'last'|'mean'|'max'|'attn'.")
    attn_hidden: Optional[int] = Field(64, ge=1, description="Hidden para atención (si kind='attn').")

# =====================================================================================
# 5) Clasificador completo tipo GRU: Encoder + Pooling + FC + Softmax
# =====================================================================================
class GRUNet(BaseModel):
    # Encoder
    encoder: SequenceEncoder = Field(..., description="Bloque secuencial (GRU apiladas).")

    # Pooling luego del encoder (si el encoder ya deja vector, esto se ignora)
    pooling: TemporalPooling = Field(default_factory=lambda: TemporalPooling(kind="last"))

    # Fully connected (intermedias) con activación común
    fc_layers: List[DenseLayer] = Field(default_factory=list, description="Capas densas intermedias.")
    fc_activation_common: ActivationFunction = Field(
        default_factory=lambda: ActivationFunction(kind="relu"),
        description="Activación común que se impone a todas las fc_layers."
    )

    # Clasificación
    classification: DenseLayer = Field(..., description="Capa final con activación 'softmax' y units=n_clases.")

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

    # ------- Dimensionamiento “estático” (sin PyTorch) -------
    def feature_dim_after_encoder(self) -> int:
        d, is_seq = self.encoder.infer_output_signature()
        if is_seq:
            # si el encoder devuelve (T,D), el pooling lo colapsa a (D,)
            return int(d)
        else:
            return int(d)

    def flatten_dim(self) -> int:
        # En RNN, el "flatten" es simplemente el vector tras pooling (o salida final).
        return self.feature_dim_after_encoder()

    # Objeto de modelo entrenado (keras.Model) para query posterior (se llena en fit)
    _tf_model: Optional[object] = None

    # =================================================================================
    # Persistencia: save/load
    # =================================================================================
    def save(self, path: str):
        """
        Guarda la instancia completa (configuración + modelo entrenado) a disco.
        
        Args:
            path: Ruta completa donde guardar el archivo .pkl
                  Ejemplo: "src/backend/models/p300/gru_20251109_143022.pkl"
        
        Note:
            Usa pickle para serializar toda la instancia incluyendo _tf_model.
            El directorio padre se crea automáticamente si no existe.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[GRU] Modelo guardado en: {path}")
    
    @classmethod
    def load(cls, path: str) -> "GRUNet":
        """
        Carga una instancia completa desde disco.
        
        Args:
            path: Ruta al archivo .pkl guardado previamente
        
        Returns:
            Instancia de GRUNet con modelo entrenado listo para query()
        
        Example:
            gru_model = GRUNet.load("src/backend/models/p300/gru_20251109_143022.pkl")
            predictions = GRUNet.query(gru_model, sequences)
        """
        with open(path, 'rb') as f:
            instance = pickle.load(f)
        print(f"[GRU] Modelo cargado desde: {path}")
        return instance
    
    @staticmethod
    def _generate_model_path(label: str, base_dir: str = "src/backend/models") -> str:
        """
        Genera ruta única para guardar modelo.
        
        Args:
            label: Etiqueta del experimento (e.g., "p300", "inner")
            base_dir: Directorio base para modelos
        
        Returns:
            Ruta completa: "{base_dir}/{label}/gru_{timestamp}.pkl"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gru_{timestamp}.pkl"
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
            metadata = GRUNet.extract_metadata_from_experiment(experiment.dict(), [0, 1])
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
    # Entrenamiento: TensorFlow/Keras con soporte de metadatos
    # =================================================================================
    @classmethod
    def train(
        cls,
        instance: "GRUNet",
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
        Entrena un modelo GRU usando TensorFlow/Keras (wrapper legacy).
        
        Este método mantiene compatibilidad con código existente retornando solo EvaluationMetrics.
        Internamente delega a fit() para evitar duplicación de código.

        Args:
            instance: Instancia de GRUNet con arquitectura configurada
            xTest: Lista de rutas a archivos .npy de test
            yTest: Lista de rutas a archivos .npy con etiquetas de test
            xTrain: Lista de rutas a archivos .npy de entrenamiento
            yTrain: Lista de rutas a archivos .npy con etiquetas de entrenamiento
            metadata_train: Lista de diccionarios con metadatos de dimensionality_change para train
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
            return_history=False,
            model_label=model_label
        )
        return result.metrics

    @classmethod
    def fit(
        cls,
        instance: "GRUNet",
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
        # 1) Prepara dataset con metadatos
        try:
            

            seq_tr, len_tr, y_tr = RecurrentModelDataUtils.prepare_sequences_and_labels(
                xTrain, yTrain,
                pad_value=instance.encoder.pad_value,
                metadata_list=metadata_train
            )
            seq_te, len_te, y_te = RecurrentModelDataUtils.prepare_sequences_and_labels(
                xTest, yTest,
                pad_value=instance.encoder.pad_value,
                metadata_list=metadata_test
            )
        except Exception as e:
            raise RuntimeError(f"Error preparando dataset GRU: {e}") from e

        # 2) Checa compatibilidad de dims
        d_enc, is_seq = instance.encoder.infer_output_signature()
        in_F = instance.encoder.input_feature_dim
        if d_enc < 1 or in_F < 1:
            raise ValueError("Dimensionalidades inválidas en encoder.")
        n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1

        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, models

            # Limpiar modelo previo (cada fit() reconstruye desde cero)
            instance._tf_model = None

            # Configurar GPU si está disponible
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU config error: {e}")

            # Padding de secuencias a longitud máxima
            max_len_tr = int(len_tr.max())
            max_len_te = int(len_te.max())
            # Usar el máximo global para que train y test tengan la misma dimensión temporal
            max_len_global = max(max_len_tr, max_len_te)
            pad_value = instance.encoder.pad_value

            def pad_sequences(seqs: List[NDArray], max_len: int, pad_val: float) -> NDArray:
                """Pad secuencias a max_len. Retorna (N, T, F)"""
                N = len(seqs)
                F = seqs[0].shape[1]
                padded = np.full((N, max_len, F), pad_val, dtype=np.float32)
                for i, seq in enumerate(seqs):
                    T = seq.shape[0]
                    padded[i, :T, :] = seq
                return padded

            X_tr_padded = pad_sequences(seq_tr, max_len_global, pad_value)  # (N_tr, T_global, F)
            X_te_padded = pad_sequences(seq_te, max_len_global, pad_value)  # (N_te, T_global, F)

            # ------- Construcción del modelo con TensorFlow/Keras -------
            def build_gru_model(spec: GRUNet, input_shape: Tuple[int, int]) -> keras.Model:
                """
                Construye modelo GRU usando Keras.

                Args:
                    spec: Especificación GRUNet
                    input_shape: (max_seq_len, feature_dim)
                """
                model_layers = []

                # Input con masking
                inputs = layers.Input(shape=input_shape, name='input')
                x = layers.Masking(mask_value=spec.encoder.pad_value)(inputs)

                # Encoder (GRU layers)
                for i, gru_layer in enumerate(spec.encoder.layers):
                    return_sequences = gru_layer.return_sequences or (i < len(spec.encoder.layers) - 1)

                    # Configurar GRU con todos los parámetros
                    gru = layers.GRU(
                        units=gru_layer.hidden_size,
                        return_sequences=return_sequences,
                        dropout=gru_layer.dropout if gru_layer.num_layers > 1 else 0.0,
                        recurrent_dropout=gru_layer.recurrent_dropout,
                        kernel_initializer=gru_layer.kernel_initializer,
                        recurrent_initializer=gru_layer.recurrent_initializer,
                        use_bias=gru_layer.use_bias,
                        reset_after=gru_layer.reset_after,
                        name=f'gru_{i}'
                    )

                    if gru_layer.bidirectional:
                        x = layers.Bidirectional(gru, name=f'bidirectional_gru_{i}')(x)
                    else:
                        x = gru(x)

                    # Si hay múltiples num_layers, apilar más GRUs
                    for j in range(1, gru_layer.num_layers):
                        inner_return_seq = return_sequences if j == gru_layer.num_layers - 1 else True
                        gru_inner = layers.GRU(
                            units=gru_layer.hidden_size,
                            return_sequences=inner_return_seq,
                            dropout=gru_layer.dropout,
                            recurrent_dropout=gru_layer.recurrent_dropout,
                            kernel_initializer=gru_layer.kernel_initializer,
                            recurrent_initializer=gru_layer.recurrent_initializer,
                            use_bias=gru_layer.use_bias,
                            reset_after=gru_layer.reset_after,
                            name=f'gru_{i}_inner_{j}'
                        )
                        if gru_layer.bidirectional:
                            x = layers.Bidirectional(gru_inner, name=f'bidirectional_gru_{i}_inner_{j}')(x)
                        else:
                            x = gru_inner(x)

                # Pooling temporal (si la última capa devuelve secuencias)
                if is_seq or spec.encoder.layers[-1].return_sequences:
                    pool_kind = spec.pooling.kind
                    if pool_kind == "last":
                        # Tomar último paso (ya manejado por return_sequences=False)
                        x = layers.Lambda(lambda t: t[:, -1, :], name='last_pooling')(x)
                    elif pool_kind == "mean":
                        x = layers.GlobalAveragePooling1D(name='mean_pooling')(x)
                    elif pool_kind == "max":
                        x = layers.GlobalMaxPooling1D(name='max_pooling')(x)
                    elif pool_kind == "attn":
                        # Attention pooling simple
                        attn_hidden = spec.pooling.attn_hidden or 64
                        # Calcular scores de atención
                        attn_scores = layers.Dense(attn_hidden, activation='tanh', name='attn_hidden')(x)
                        attn_scores = layers.Dense(1, name='attn_scores')(attn_scores)
                        attn_weights = layers.Softmax(axis=1, name='attn_weights')(attn_scores)
                        # Weighted sum
                        x = layers.Multiply(name='attn_multiply')([x, attn_weights])
                        x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name='attn_sum')(x)

                # FC layers
                for i, fc in enumerate(spec.fc_layers):
                    x = layers.Dense(fc.units, name=f'fc_{i}')(x)
                    act = spec.fc_activation_common.kind
                    if act == "relu":
                        x = layers.ReLU(name=f'fc_{i}_relu')(x)
                    elif act == "tanh":
                        x = layers.Activation('tanh', name=f'fc_{i}_tanh')(x)
                    elif act == "gelu":
                        x = layers.Activation('gelu', name=f'fc_{i}_gelu')(x)
                    elif act == "sigmoid":
                        x = layers.Activation('sigmoid', name=f'fc_{i}_sigmoid')(x)

                # Classification layer
                outputs = layers.Dense(spec.classification.units, activation='softmax', name='classification')(x)

                return keras.Model(inputs=inputs, outputs=outputs, name='GRUNet')

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
                print("🚀 [GRU] Intentando entrenamiento en GPU con optimizaciones...")

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

                        # Construir modelo DESPUÉS de configurar GPU
                        model = build_gru_model(instance, input_shape=(X_tr_padded.shape[1], in_F))
                        model.compile(
                            optimizer=keras.optimizers.Adam(learning_rate=lr),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )

                        # Entrenar en GPU
                        t0 = time.perf_counter()
                        history = model.fit(
                            X_tr_padded, y_tr,
                            batch_size=gpu_batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(X_te_padded, y_te)
                        )
                        train_time = time.perf_counter() - t0

                        print(f"✅ [GRU] Entrenamiento en GPU completado exitosamente!")
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
                print("🔄 [GRU] Cambiando a entrenamiento en CPU...")
                print("   (Esto será más lento pero garantiza que complete)")

                # Limpiar configuración GPU
                tf.keras.backend.clear_session()
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy('float32')
                gc.collect()

                # Usar batch_size más pequeño también en CPU para evitar OOM de RAM
                cpu_batch_size = max(4, batch_size // 4)
                print(f"   Usando batch_size reducido en CPU: {cpu_batch_size}")

                # Forzar uso de CPU
                with tf.device('/CPU:0'):
                    # Reconstruir modelo en CPU
                    model = build_gru_model(instance, input_shape=(X_tr_padded.shape[1], in_F))
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=lr),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )

                    # Entrenar en CPU con batch_size reducido
                    t0 = time.perf_counter()
                    history = model.fit(
                        X_tr_padded, y_tr,
                        batch_size=cpu_batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_te_padded, y_te)
                    )
                    train_time = time.perf_counter() - t0

                    print(f"✅ [GRU] Entrenamiento en CPU completado exitosamente!")
                    print(f"   Tiempo: {train_time:.2f}s | Batch size: {cpu_batch_size}")
                    return history, train_time, cpu_batch_size, model, 'CPU'

            # Ejecutar entrenamiento con fallback
            history, train_time, final_batch_size, model, device_used = train_with_fallback()
            batch_size = final_batch_size  # Actualizar para predicción

            train_losses: List[float] = history.history['loss']

            # Evaluación - USAR EL MISMO DISPOSITIVO que entrenamiento
            t0 = time.perf_counter()
            if device_used == 'CPU':
                with tf.device('/CPU:0'):
                    logits = model.predict(X_te_padded, batch_size=batch_size, verbose=0)
            else:
                logits = model.predict(X_te_padded, batch_size=batch_size, verbose=0)
            y_pred = np.argmax(logits, axis=1)
            y_true = y_te
            eval_seconds = time.perf_counter() - t0

            # Métricas básicas
            acc  = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec  = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            f1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            cm   = confusion_matrix(y_true, y_pred).tolist()

            # AUC-ROC
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
                loss=train_losses,
                evaluation_time=f"{eval_seconds:.4f}s",
            )

            # Guardar modelo en la instancia
            instance._tf_model = model

            # Auto-guardar si se proporciona model_label
            if model_label:
                save_path = cls._generate_model_path(model_label)
                instance.save(save_path)
                print(f"[GRU] Modelo guardado automáticamente en: {save_path}")

            # Construir historial para TrainResult
            hist_dict = history.history if return_history else {}

            # Construir hiperparámetros
            hyperparams = {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "n_classes": n_classes,
                "max_len_train": max_len_tr,
                "max_len_test": max_len_te,
                "model_architecture": model.summary(print_fn=lambda x: None) or "GRU",
            }

            print(f"[GRU] Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")

            return TrainResult(
                metrics=metrics,
                model=instance,
                model_name="GRU",
                training_seconds=train_time,
                history=hist_dict,
                hyperparams=hyperparams,
            )

        except Exception as e:
            print("[GRU] Entrenamiento real no ejecutado (fallback).")
            print(f"Razón: {e}")
            import traceback
            traceback.print_exc()
            metrics = EvaluationMetrics(
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
                metrics=metrics,
                model=instance,
                model_name="GRU",
                training_seconds=0.0,
                history={},
                hyperparams={"error": str(e)},
            )

    # =================================================================================
    # Inferencia con modelo entrenado
    # =================================================================================
    @classmethod
    def query(
        cls,
        instance: "GRUNet",
        data: List[NDArray],
        metadata_list: Optional[List[dict]] = None,
        batch_size: int = 64
    ) -> Tuple[NDArray, NDArray]:
        """
        Realiza inferencia con el modelo GRU entrenado.

        Args:
            instance: Instancia de GRUNet con modelo entrenado (_tf_model debe existir)
            data: Lista de secuencias como NDArray (cada una con shape (T, F))
            metadata_list: Lista opcional de diccionarios con metadatos (misma estructura que train)
            batch_size: Tamaño de batch para predicción

        Returns:
            Tuple[NDArray, NDArray]: (predictions, probabilities)
                - predictions: Array 1D con clases predichas (índices)
                - probabilities: Array 2D con probabilidades por clase (N, n_classes)

        Raises:
            ValueError: Si el modelo no ha sido entrenado (_tf_model es None)
        """
        if instance._tf_model is None:
            raise ValueError(
                "El modelo no ha sido entrenado. Llama a train() o fit() primero, "
                "o carga un modelo existente con load()."
            )

        try:
            import tensorflow as tf
            
            # Preparar secuencias (sin labels ya que es inferencia)
            if metadata_list:
                # Aplicar mismo procesamiento que en entrenamiento
                processed_seqs = []
                for seq, metadata in zip(data, metadata_list):
                    axis_order, needs_transpose = cls._interpret_metadata(metadata)
                    if needs_transpose:
                        seq = seq.T  # (F,T) → (T,F)
                    processed_seqs.append(seq)
            else:
                processed_seqs = data

            # Padding a longitud máxima
            max_len = max(seq.shape[0] for seq in processed_seqs)
            pad_value = instance.encoder.pad_value
            F = processed_seqs[0].shape[1]
            N = len(processed_seqs)

            X_padded = np.full((N, max_len, F), pad_value, dtype=np.float32)
            for i, seq in enumerate(processed_seqs):
                T = seq.shape[0]
                X_padded[i, :T, :] = seq

            # Predicción
            probabilities = instance._tf_model.predict(X_padded, batch_size=batch_size, verbose=0)
            predictions = np.argmax(probabilities, axis=1)

            return predictions, probabilities

        except Exception as e:
            raise RuntimeError(f"Error en inferencia GRU: {e}") from e



# Alias for backward compatibility
GRU = GRUNet


# ======================= Ejemplo de uso =======================
"""
# Ejemplo 1: Construcción básica de modelo GRU

from backend.classes.ClasificationModel.GRU import GRUNet, GRULayer, SequenceEncoder, TemporalPooling, DenseLayer, ActivationFunction

# Configurar capas GRU
gru1 = GRULayer(
    hidden_size=128,
    bidirectional=True,
    dropout=0.2,
    recurrent_dropout=0.1,
    num_layers=1,
    return_sequences=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal"
)

gru2 = GRULayer(
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
    layers=[gru1, gru2],
    use_packed_sequences=True,  # no se usa en TF, solo para compatibilidad
    pad_value=0.0
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
gru_net = GRUNet(
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
metadata_train = GRUNet.extract_metadata_from_experiment(experiment.dict(), transform_indices=[0, 1])
metadata_test = GRUNet.extract_metadata_from_experiment(experiment.dict(), transform_indices=[0])

# Entrenar modelo
metrics = GRUNet.train(
    gru_net,
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

metrics = GRUNet.train(
    gru_net,
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
metrics = GRUNet.train(
    gru_net,
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

# Ejemplo 6: Configuración avanzada de GRU
gru_advanced = GRULayer(
    hidden_size=256,
    bidirectional=True,
    dropout=0.3,
    recurrent_dropout=0.2,
    num_layers=2,  # Apilar 2 GRUs internas
    return_sequences=True,
    kernel_initializer="he_normal",
    recurrent_initializer="orthogonal",
    use_bias=True,
    reset_after=True
)
"""
