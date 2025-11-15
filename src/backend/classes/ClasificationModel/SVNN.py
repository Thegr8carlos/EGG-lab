from __future__ import annotations
from typing import List, Optional, Literal, Sequence, Tuple
import os
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

# Importa tu base
from backend.classes.ClasificationModel.ClsificationModels import Classifier
# (Opcional) from backend.classes.Metrics import EvaluationMetrics
import time

from backend.classes.Metrics import EvaluationMetrics
from backend.classes.ClasificationModel.utils.TrainResult import TrainResult

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# ========= Tipos / helpers =========
NDArray = np.ndarray
ActName = Literal["relu", "tanh", "sigmoid", "gelu", "softmax", "linear"]
Reduce3D = Literal["flatten", "mean_time_flat", "mean_all"]
ScaleMode = Literal["none", "standard"]  # standard: z-score por muestra

# ========= Bloques declarativos =========
class ActivationFunction(BaseModel):
    kind: ActName = Field("relu", description="Tipo de activación.")

class DenseLayer(BaseModel):
    units: int = Field(..., ge=1, description="Número de neuronas.")
    activation: ActivationFunction = Field(default_factory=lambda: ActivationFunction(kind="relu"))
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout posterior a la capa.")
    batchnorm: bool = Field(False, description="Aplicar BatchNorm antes de la activación.")
    kernel_initializer: str = Field("glorot_uniform", description="Inicializador de pesos.")
    bias_initializer: str = Field("zeros", description="Inicializador de bias.")
    kernel_regularizer: Optional[str] = Field(None, description="Regularizador L1/L2 (e.g., 'l2', 'l1').")
    regularizer_value: float = Field(0.01, gt=0.0, description="Valor del regularizador si se usa.")

    @field_validator("units")
    @classmethod
    def _v_units(cls, v: int) -> int:
        if v < 1:
            raise ValueError("DenseLayer.units debe ser >= 1.")
        return v

class InputAdapter(BaseModel):
    """
    Cómo convertir un .npy arbitario en un vector 1D de features por archivo.
    """
    reduce_3d: Reduce3D = Field(
        "flatten",
        description=(
            "Cuando X.ndim==3: "
            "'flatten' -> aplana todo; "
            "'mean_time_flat' -> promedio sobre eje 0 (p.ej., frames) y luego flatten; "
            "'mean_all' -> escalar único por canal/frecuencia (promedia todo)."
        )
    )
    scale: ScaleMode = Field("standard", description="'none' o 'standard' (z-score por muestra).")
    allow_mixed_dims: bool = Field(
        True,  # Cambiado a True para soportar muestras de diferentes longitudes con auto-padding
        description="Si True, permite que cada archivo tenga forma distinta (se aplica auto-padding con ceros)."
    )

    def transform_one(self, x: NDArray) -> NDArray:
        """
        Devuelve un vector 1D (features,).
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            vec = x.astype(np.float32, copy=False)
        elif x.ndim == 2:
            # si viene (F, T) o (T, F): da igual para MLP -> flatten
            vec = x.astype(np.float32, copy=False).reshape(-1)
        elif x.ndim == 3:
            X = x.astype(np.float32, copy=False)
            if self.reduce_3d == "flatten":
                vec = X.reshape(-1)
            elif self.reduce_3d == "mean_time_flat":
                # interpreta eje 0 como "tiempo" (o frames) y promedio
                Xr = X.mean(axis=0)  # (H,W) o (Freq,Canales)
                vec = Xr.reshape(-1)
            else:  # mean_all
                vec = np.array([X.mean()], dtype=np.float32)
        else:
            raise ValueError(f"Entrada .npy con ndim={x.ndim} no soportada (esperado 1D/2D/3D).")

        if self.scale == "standard":
            m = float(vec.mean())
            s = float(vec.std() + 1e-8)
            vec = (vec - m) / s
        return vec.astype(np.float32, copy=False)

    def transform_batch(self, paths: Sequence[str]) -> NDArray:
        """
        Carga una lista de rutas .npy y devuelve una matriz (N, D).
        Si las longitudes varían y allow_mixed_dims=True, se rellena a la máxima con ceros.
        """
        vecs: List[NDArray] = []
        dims: List[int] = []
        for p in paths:
            if (not os.path.exists(p)) or (not p.endswith(".npy")):
                raise FileNotFoundError(f"Archivo inválido: {p} (debe existir y ser .npy)")
            x = np.load(p, allow_pickle=True)
            v = self.transform_one(x)
            vecs.append(v)
            dims.append(int(v.shape[0]))

        if len(set(dims)) == 1:
            return np.stack(vecs, axis=0)

        if not self.allow_mixed_dims:
            raise ValueError(
                "Las muestras tienen distinta longitud de features. "
                "Activa allow_mixed_dims=True para auto-padding."
            )

        D = max(dims)
        out = np.zeros((len(vecs), D), dtype=np.float32)
        for i, v in enumerate(vecs):
            out[i, : v.shape[0]] = v
        return out

# ========= SVNN (MLP) =========
class SVNN(Classifier):
    # Preset rápido:
    hidden_size: int = Field(64, ge=1, description="Ancho base (si no se especifican layers).")

    # Optimización / entrenamiento
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Tasa de aprendizaje")
    epochs: int = Field(100, ge=1, le=1000, description="Épocas de entrenamiento")
    batch_size: int = Field(16, ge=1, le=512, description="Tamaño de batch")

    # Arquitectura abierta
    layers: List[DenseLayer] = Field(
        default_factory=list,
        description="Capas densas. Si se deja vacío, se usan 2 capas con units=hidden_size."
    )
    fc_activation_common: ActivationFunction = Field(
        default_factory=lambda: ActivationFunction(kind="relu"),
        description="Activación común para layers (se impone al validar)."
    )
    classification_units: int = Field(
        2, ge=2,
        description="Clases de salida (si no quieres inferirlo de y)."
    )

    # Adaptador de entrada
    input_adapter: InputAdapter = Field(default_factory=InputAdapter)

    # Modelo entrenado (keras.Model) para query posterior (se llena en fit)
    _keras_model: Optional[object] = None

    @field_validator("layers")
    @classmethod
    def _v_layers(cls, layers, info):
        """
        Si no hay layers, genera dos con units=hidden_size.
        Impone activación común a todas (si no es linear/softmax).
        """
        values = info.data
        if not layers or len(layers) == 0:
            hs = int(values.get("hidden_size", 64))
            layers = [DenseLayer(units=hs), DenseLayer(units=hs)]
        common: ActivationFunction = values.get("fc_activation_common", ActivationFunction(kind="relu"))
        # no toques la activación si el usuario ya puso softmax o linear en una capa oculta
        for i, lyr in enumerate(layers):
            if lyr.activation.kind in ("softmax",):
                continue
            layers[i].activation = common
        return layers

    # ----------------------------- Persistencia: save/load -----------------------------
    def save(self, path: str):
        """
        Guarda la instancia completa (configuración + modelo entrenado) a disco.
        
        Args:
            path: Ruta completa donde guardar el archivo .pkl
                  Ejemplo: "src/backend/models/p300/svnn_20251109_143022.pkl"
        
        Note:
            Usa pickle para serializar toda la instancia incluyendo _keras_model.
            El directorio padre se crea automáticamente si no existe.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[SVNN] Modelo guardado en: {path}")
    
    @classmethod
    def load(cls, path: str) -> "SVNN":
        """
        Carga una instancia completa desde disco.
        
        Args:
            path: Ruta al archivo .pkl guardado previamente
        
        Returns:
            Instancia de SVNN con modelo entrenado listo para query()
        
        Example:
            svnn_model = SVNN.load("src/backend/models/p300/svnn_20251109_143022.pkl")
            predictions = SVNN.query(svnn_model, x_paths)
        """
        with open(path, 'rb') as f:
            instance = pickle.load(f)
        print(f"[SVNN] Modelo cargado desde: {path}")
        return instance
    
    @staticmethod
    def _generate_model_path(label: str, base_dir: str = "src/backend/models") -> str:
        """
        Genera ruta única para guardar modelo.
        
        Args:
            label: Etiqueta del experimento (e.g., "p300", "inner")
            base_dir: Directorio base para modelos
        
        Returns:
            Ruta completa: "{base_dir}/{label}/svnn_{timestamp}.pkl"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"svnn_{timestamp}.pkl"
        return str(Path(base_dir) / label / filename)

    # ----------------------------- Helpers de metadatos -----------------------------
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
            metadata = SVNN.extract_metadata_from_experiment(experiment.dict(), [0, 1])
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

    # ----------------------------- I/O Helpers -----------------------------
    @staticmethod
    def _load_labels_scalar(paths: Sequence[str]) -> NDArray:
        """
        Carga etiquetas para clasificación por archivo.
        Acepta:
        - escalar ((), (1,)) -> se usa directo
        - vector (n_frames,) -> se usa la MODA (etiqueta más frecuente)
        Devuelve (N,) int64
        """
        from collections import Counter
        ys: List[int] = []

        for p in paths:
            if (not os.path.exists(p)) or (not p.endswith(".npy")):
                raise FileNotFoundError(f"Archivo de etiqueta inválido: {p}")

            y = np.load(p, allow_pickle=True)
            y = np.array(y).reshape(-1)

            if y.size == 0:
                raise ValueError(f"Etiqueta inválida en {p}: array vacío.")

            if y.size == 1:
                ys.append(int(y[0]))
                continue

            # Vector por frame -> moda
            cleaned = []
            for val in y:
                try:
                    cleaned.append(int(val))
                except (ValueError, TypeError):
                    cleaned.append(str(val))

            most_common_label = Counter(cleaned).most_common(1)[0][0]
            try:
                ys.append(int(most_common_label))
            except (ValueError, TypeError):
                raise ValueError(
                    f"La moda de las etiquetas en {p} no es un entero válido: {most_common_label}"
                )

        return np.array(ys, dtype=np.int64)


    @classmethod
    def _prepare_xy_with_padding(
        cls,
        instance: "SVNN",
        x_paths: Sequence[str],
        y_paths: Sequence[str],
        target_dim: int,
    ) -> Tuple[NDArray, NDArray]:
        """
        Prepara datos con padding a una dimensión específica.

        Args:
            instance: Instancia de SVNN
            x_paths: Rutas a archivos .npy con features
            y_paths: Rutas a archivos .npy con etiquetas
            target_dim: Dimensión objetivo para padding

        Returns:
            X: (N, target_dim) - Features con padding
            y: (N,) - Labels
        """
        print(f"[SVNN._prepare_xy_with_padding] Preparando {len(x_paths)} muestras con target_dim={target_dim}")

        if len(x_paths) != len(y_paths):
            raise ValueError("x_paths y y_paths deben tener la misma longitud.")

        # Cargar y transformar cada muestra
        vecs: List[NDArray] = []
        for p in x_paths:
            if not os.path.exists(p) or not p.endswith(".npy"):
                raise FileNotFoundError(f"Archivo inválido: {p}")
            x = np.load(p, allow_pickle=True)
            v = instance.input_adapter.transform_one(x)
            vecs.append(v)

        # Crear matriz con padding a target_dim
        X = np.zeros((len(vecs), target_dim), dtype=np.float32)
        for i, v in enumerate(vecs):
            # Si v es más largo que target_dim, truncar
            # Si v es más corto, rellenar con ceros
            if v.shape[0] > target_dim:
                X[i, :] = v[:target_dim]
            else:
                X[i, : v.shape[0]] = v

        # Cargar labels
        y = cls._load_labels_scalar(y_paths)

        print(f"[SVNN._prepare_xy_with_padding] Resultado: X.shape={X.shape}, y.shape={y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"N muestras no coincide con N etiquetas: X={X.shape[0]} vs y={y.shape[0]}")

        return X, y

    @classmethod
    def _prepare_xy(
        cls,
        instance: "SVNN",
        x_paths: Sequence[str],
        y_paths: Sequence[str],
        metadata_list: Optional[Sequence[dict]] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        - X: cada .npy -> vector 1D según input_adapter.
        - y: un escalar por archivo (.npy).

        Args:
            instance: Instancia de SVNN con configuración
            x_paths: Rutas a archivos .npy con features
            y_paths: Rutas a archivos .npy con etiquetas
            metadata_list: Lista opcional de metadatos (actualmente no se usa, el InputAdapter maneja todo)

        Note:
            El InputAdapter ya maneja la transformación de datos arbitrarios a vectores 1D,
            por lo que los metadatos son opcionales y solo informativos.
        """
        # Modo clásico: cada ruta representa UNA muestra con sus features y su etiqueta.
        print(f"[SVNN._prepare_xy] DEBUG - Recibido:")
        print(f"  len(x_paths) = {len(x_paths)}")
        print(f"  len(y_paths) = {len(y_paths)}")

        if len(x_paths) != len(y_paths):
            print(f"[SVNN._prepare_xy] ERROR - Longitudes no coinciden!")
            print(f"  x_paths ({len(x_paths)}): {x_paths[:3] if len(x_paths) > 0 else []}")
            print(f"  y_paths ({len(y_paths)}): {y_paths[:3] if len(y_paths) > 0 else []}")
            raise ValueError("x_paths y y_paths deben tener la misma longitud.")

        X = instance.input_adapter.transform_batch(x_paths)  # (N,D)
        y = cls._load_labels_scalar(y_paths)                 # (N,)

        print(f"[SVNN._prepare_xy] DEBUG - Después de cargar:")
        print(f"  X.shape = {X.shape}")
        print(f"  y.shape = {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"N muestras no coincide con N etiquetas: X={X.shape[0]} vs y={y.shape[0]}")
        return X, y

    # ----------------------------- Entrenamiento -----------------------------
    @classmethod
    def train(
        cls,
        instance: "SVNN",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
        metadata_train: Optional[Sequence[dict]] = None,
        metadata_test: Optional[Sequence[dict]] = None,
        model_label: Optional[str] = None,
    ):
        """
        Entrenamiento con TensorFlow/Keras (wrapper legacy).
        
        Este método mantiene compatibilidad con código existente retornando solo EvaluationMetrics.
        Internamente delega a fit() para evitar duplicación de código.

        Args:
            instance: Instancia de SVNN con configuración
            xTest: Rutas a archivos .npy de test
            yTest: Rutas a etiquetas de test
            xTrain: Rutas a archivos .npy de entrenamiento
            yTrain: Rutas a etiquetas de entrenamiento
            metadata_train: Metadatos opcionales para datos de entrenamiento
            metadata_test: Metadatos opcionales para datos de test
            model_label: Etiqueta opcional para auto-guardar (e.g., "p300", "inner").
                        Si se proporciona, guarda automáticamente en src/backend/models/{label}/

        Returns:
            EvaluationMetrics: Solo las métricas de evaluación (para compatibilidad backward)
            
        Note:
            Si necesitas acceso al modelo entrenado y más información, usa fit() en su lugar.

        Ejemplo:
            # Con metadatos
            experiment = Experiment._load_latest_experiment()
            metadata = SVNN.extract_metadata_from_experiment(experiment.dict())
            metrics = SVNN.train(model, xTest, yTest, xTrain, yTrain,
                               metadata_train=metadata, metadata_test=metadata)
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
            return_history=False,  # No necesitamos historia completa para legacy API
            model_label=model_label
        )
        return result.metrics

    @classmethod
    def fit(
        cls,
        instance: "SVNN",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
        metadata_train: Optional[Sequence[dict]] = None,
        metadata_test: Optional[Sequence[dict]] = None,
        return_history: bool = True,
        model_label: Optional[str] = None,
        *,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> TrainResult:
        """Entrena y devuelve paquete TrainResult (modelo + métricas + historia).

        No rompe `train()`: es una API paralela opt-in.
        
        Args:
            model_label: Etiqueta opcional para auto-guardar (e.g., "p300", "inner").
                        Si se proporciona, guarda automáticamente en src/backend/models/{label}/
        """
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers

        # 0) Limpiar modelo previo (cada fit() reconstruye desde cero basándose en los datos actuales)
        instance._keras_model = None
        # Limpiar backend de Keras para evitar reutilización de pesos previos
        keras.backend.clear_session()

        # 1) Preparar datos con padding global unificado
        # IMPORTANTE: Si allow_mixed_dims=True, debemos calcular el max global
        # entre train y test para que ambos tengan la misma dimensión final

        if instance.input_adapter.allow_mixed_dims:
            # Calcular dimensión máxima global (train + test)
            print("[SVNN.fit] allow_mixed_dims=True, calculando dimensión máxima global...")

            # Cargar y transformar UNA muestra de cada set para obtener dimensiones
            sample_dims_train = []
            sample_dims_test = []

            for p in xTrain[:min(10, len(xTrain))]:  # Muestrear primeros 10
                x = np.load(p, allow_pickle=True)
                v = instance.input_adapter.transform_one(x)
                sample_dims_train.append(v.shape[0])

            for p in xTest[:min(10, len(xTest))]:
                x = np.load(p, allow_pickle=True)
                v = instance.input_adapter.transform_one(x)
                sample_dims_test.append(v.shape[0])

            max_train = max(sample_dims_train) if sample_dims_train else 0
            max_test = max(sample_dims_test) if sample_dims_test else 0
            global_max_dim = max(max_train, max_test)

            print(f"   Max dim train (sample): {max_train}")
            print(f"   Max dim test (sample):  {max_test}")
            print(f"   Global max dim:         {global_max_dim}")

            # Preparar datos con padding a dimensión global
            print(f"[SVNN.fit] Preparando train set con padding...")
            Xtr, ytr = cls._prepare_xy_with_padding(instance, xTrain, yTrain, target_dim=global_max_dim)
            print(f"[SVNN.fit] Preparando test set con padding...")
            Xte, yte = cls._prepare_xy_with_padding(instance, xTest, yTest, target_dim=global_max_dim)
        else:
            # Modo normal: asume que todas las muestras ya tienen misma dimensión
            Xtr, ytr = cls._prepare_xy(instance, xTrain, yTrain, metadata_train)
            Xte, yte = cls._prepare_xy(instance, xTest, yTest, metadata_test)

        # 1.1) Validación estricta: las dimensiones de features deben coincidir
        if Xtr.shape[1] != Xte.shape[1]:
            d_tr, d_te = int(Xtr.shape[1]), int(Xte.shape[1])
            raise ValueError(
                f"Dimensión de features distinta entre train/test: {d_tr} vs {d_te}. "
                "Asegura que ambos provengan del mismo pipeline/transform. "
                f"Considera activar allow_mixed_dims=True en input_adapter."
            )

        in_dim = int(Xtr.shape[1])
        # Inferir número de clases de manera robusta
        try:
            n_classes = int(np.unique(np.concatenate([ytr.reshape(-1), yte.reshape(-1)])).size)
        except Exception:
            n_classes = int(max(int(ytr.max()), int(yte.max()))) + 1
        out_dim = max(n_classes, int(instance.classification_units))

        # 2) Función helper para construir el modelo
        def build_svnn_model():
            """Construye el modelo SVNN secuencial"""
            model_layers = []
            d_in = in_dim

            # Capas ocultas según instance.layers
            for i, lyr in enumerate(instance.layers):
                # Configurar regularizador si se especifica
                reg = None
                if lyr.kernel_regularizer:
                    if lyr.kernel_regularizer.lower() == "l2":
                        reg = regularizers.l2(lyr.regularizer_value)
                    elif lyr.kernel_regularizer.lower() == "l1":
                        reg = regularizers.l1(lyr.regularizer_value)
                    elif lyr.kernel_regularizer.lower() == "l1_l2":
                        reg = regularizers.l1_l2(l1=lyr.regularizer_value, l2=lyr.regularizer_value)

                # Capa Dense
                model_layers.append(
                    layers.Dense(
                        units=lyr.units,
                        use_bias=True,
                        kernel_initializer=lyr.kernel_initializer,
                        bias_initializer=lyr.bias_initializer,
                        kernel_regularizer=reg,
                        name=f"dense_{i}"
                    )
                )
                d_in = lyr.units

                # BatchNormalization si se solicita
                if lyr.batchnorm:
                    model_layers.append(layers.BatchNormalization(name=f"bn_{i}"))

                # Activación
                a = lyr.activation.kind
                if a == "relu":
                    model_layers.append(layers.ReLU(name=f"relu_{i}"))
                elif a == "tanh":
                    model_layers.append(layers.Activation("tanh", name=f"tanh_{i}"))
                elif a == "gelu":
                    model_layers.append(layers.Activation("gelu", name=f"gelu_{i}"))
                elif a == "sigmoid":
                    model_layers.append(layers.Activation("sigmoid", name=f"sigmoid_{i}"))
                elif a == "linear":
                    pass  # Sin activación
                elif a == "softmax":
                    # No aplicar softmax en capas intermedias
                    pass

                # Dropout si se solicita
                if lyr.dropout > 0:
                    model_layers.append(layers.Dropout(rate=float(lyr.dropout), name=f"dropout_{i}"))

            # Capa de salida
            model_layers.append(
                layers.Dense(
                    units=out_dim,
                    activation="softmax",
                    kernel_initializer="glorot_uniform",
                    name="output"
                )
            )

            # Construir modelo secuencial
            return keras.Sequential(model_layers, name="SVNN_MLP")

        # 3) Resolver hiperparámetros (UI override si se pasan)
        ep_val = int(epochs) if epochs is not None else int(instance.epochs)
        bs_val = int(batch_size) if batch_size is not None else int(instance.batch_size)
        lr_val = float(learning_rate) if learning_rate is not None else float(instance.learning_rate)

        # IMPORTANTE: Configurar GPU ANTES de construir el modelo
        import tensorflow as tf
        import gc

        # Limpiar sesión anterior
        tf.keras.backend.clear_session()
        gc.collect()

        # 4) Entrenamiento con fallback GPU -> CPU
        def train_with_fallback():
            """Intenta entrenar en GPU, si falla cae a CPU automáticamente"""
            nonlocal bs_val

            # Fase 1: Intentar con GPU optimizada
            print("🚀 [SVNN] Intentando entrenamiento en GPU con optimizaciones...")

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
            gpu_batch_size = max(4, bs_val // 2)

            max_gpu_retries = 2
            model = None

            for attempt in range(max_gpu_retries):
                try:
                    print(f"   Intento GPU {attempt + 1}/{max_gpu_retries} (batch_size={gpu_batch_size})")

                    # Construir modelo DESPUÉS de configurar GPU
                    model = build_svnn_model()
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=lr_val),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"]
                    )

                    # Entrenar en GPU
                    t0 = time.perf_counter()
                    history = model.fit(
                        Xtr, ytr,
                        batch_size=gpu_batch_size,
                        epochs=ep_val,
                        validation_data=(Xte, yte),
                        verbose=1
                    )
                    train_time = time.perf_counter() - t0

                    print(f"✅ [SVNN] Entrenamiento en GPU completado exitosamente!")
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
            print("🔄 [SVNN] Cambiando a entrenamiento en CPU...")
            print("   (Esto será más lento pero garantiza que complete)")

            # Limpiar configuración GPU
            tf.keras.backend.clear_session()
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('float32')
            gc.collect()

            # Usar batch_size más pequeño también en CPU para evitar OOM de RAM
            cpu_batch_size = max(4, bs_val // 4)
            print(f"   Usando batch_size reducido en CPU: {cpu_batch_size}")

            # Forzar uso de CPU
            with tf.device('/CPU:0'):
                # Reconstruir modelo en CPU
                model = build_svnn_model()
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr_val),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"]
                )

                # Entrenar en CPU con batch_size reducido
                t0 = time.perf_counter()
                history = model.fit(
                    Xtr, ytr,
                    batch_size=cpu_batch_size,
                    epochs=ep_val,
                    validation_data=(Xte, yte),
                    verbose=1
                )
                train_time = time.perf_counter() - t0

                print(f"✅ [SVNN] Entrenamiento en CPU completado exitosamente!")
                print(f"   Tiempo: {train_time:.2f}s | Batch size: {cpu_batch_size}")
                return history, train_time, cpu_batch_size, model, 'CPU'

        # Ejecutar entrenamiento con fallback
        history, train_time, final_batch_size, model, device_used = train_with_fallback()
        bs_val = final_batch_size  # Actualizar para predicción

        # 5) Evaluación + métricas completas
        t_eval = time.perf_counter()

        # Predicciones y probabilidades - USAR EL MISMO DISPOSITIVO que entrenamiento
        if device_used == 'CPU':
            with tf.device('/CPU:0'):
                y_proba = model.predict(Xte, batch_size=max(64, int(instance.batch_size)), verbose=0)
        else:
            y_proba = model.predict(Xte, batch_size=max(64, int(instance.batch_size)), verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        y_true = yte

        eval_seconds = time.perf_counter() - t_eval

        # Métricas básicas
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()

        # AUC-ROC (binario/multiclase usando softmax y OVR)
        try:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            auc = 0.0

        metrics = EvaluationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            confusion_matrix=cm,
            auc_roc=auc,
            loss=history.history.get("loss", []),
            evaluation_time=f"{eval_seconds:.4f}s",
        )

        # Guardar modelo en instancia para query()
        instance._keras_model = model

        # Logging compacto de depuración
        try:
            uniq_tr, uniq_te = np.unique(ytr).tolist(), np.unique(yte).tolist()
            print(f"[SVNN.fit] Xtr={Xtr.shape} Xte={Xte.shape} classes(tr)={uniq_tr} classes(te)={uniq_te}")
        except Exception:
            pass
        print(f"[SVNN.fit] Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")
        
        # Auto-guardar si se proporciona label
        if model_label:
            save_path = cls._generate_model_path(model_label)
            instance.save(save_path)
        
        return TrainResult(
            metrics=metrics,
            model=model,
            model_name="SVNN",
            training_seconds=float(train_time),
            history=history.history if return_history else None,
            hyperparams={
                "learning_rate": lr_val,
                "epochs": ep_val,
                "batch_size": bs_val,
                "n_classes": n_classes,
                "input_dim": in_dim,
            }
        )

    @classmethod
    def query(
        cls,
        instance: "SVNN",
        x_paths: List[str],
        return_logits: bool = False
    ):
        """Inferencia sobre lista de archivos .npy.
        
        Args:
            instance: Instancia de SVNN con modelo entrenado
            x_paths: Lista de rutas a archivos .npy con features
            return_logits: Si True, retorna (predictions, probabilities)
            
        Returns:
            predictions: Lista de etiquetas predichas (int)
            Si return_logits=True: (predictions, probabilities)
        """
        if instance._keras_model is None:
            raise RuntimeError("Modelo SVNN no entrenado: usa fit() antes de query().")
        if not x_paths:
            return []
        
        # Transformar datos usando el input_adapter
        X = instance.input_adapter.transform_batch(x_paths)  # (N, D)
        
        # Predicción
        probs = instance._keras_model.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1).tolist()
        
        if return_logits:
            return preds, probs.tolist()
        return preds


# ========================================
# EJEMPLOS COMPREHENSIVOS DE USO
# ========================================
"""
La clase SVNN (Shallow Vector Neural Network) implementa una red neuronal totalmente
conectada (MLP) para clasificación de EEG. Soporta arquitecturas personalizables con
múltiples capas Dense, regularización, BatchNorm y diferentes activaciones.

NUEVO: Soporta metadatos de dimensionality_change para interpretación flexible de datos.

# ------------------------------------------------------------------------------
# Ejemplo 1: Configuración básica con capas por defecto
# ------------------------------------------------------------------------------
model_basic = SVNN(
    learning_rate=0.001,
    epochs=50,
    batch_size=32,
    classification_units=2,
    input_adapter=InputAdapter(reduce_3d="flatten", scale="standard")
)
# Usa 2 capas con hidden_size=64 por defecto

metrics = SVNN.train(
    model_basic,
    xTest=['test1.npy', 'test2.npy'],
    yTest=['label1.npy', 'label2.npy'],
    xTrain=['train1.npy', 'train2.npy'],
    yTrain=['label_train1.npy', 'label_train2.npy']
)

# ------------------------------------------------------------------------------
# Ejemplo 2: Arquitectura personalizada con regularización L2
# ------------------------------------------------------------------------------
model_regularized = SVNN(
    learning_rate=5e-4,
    epochs=80,
    batch_size=64,
    layers=[
        DenseLayer(
            units=256,
            activation=ActivationFunction(kind="relu"),
            dropout=0.3,
            batchnorm=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer="l2",
            regularizer_value=0.01
        ),
        DenseLayer(
            units=128,
            activation=ActivationFunction(kind="relu"),
            dropout=0.2,
            batchnorm=True,
            kernel_regularizer="l2",
            regularizer_value=0.01
        ),
        DenseLayer(
            units=64,
            activation=ActivationFunction(kind="relu"),
            dropout=0.1,
            kernel_regularizer="l2",
            regularizer_value=0.005
        ),
    ],
    classification_units=4,
    input_adapter=InputAdapter(reduce_3d="flatten", scale="standard")
)

# ------------------------------------------------------------------------------
# Ejemplo 3: Arquitectura profunda con GELU y BatchNorm
# ------------------------------------------------------------------------------
model_deep = SVNN(
    learning_rate=1e-3,
    epochs=100,
    batch_size=32,
    layers=[
        DenseLayer(units=512, activation=ActivationFunction(kind="gelu"),
                   dropout=0.4, batchnorm=True, kernel_regularizer="l2", regularizer_value=0.02),
        DenseLayer(units=256, activation=ActivationFunction(kind="gelu"),
                   dropout=0.3, batchnorm=True, kernel_regularizer="l2", regularizer_value=0.02),
        DenseLayer(units=128, activation=ActivationFunction(kind="gelu"),
                   dropout=0.2, batchnorm=True, kernel_regularizer="l2", regularizer_value=0.01),
        DenseLayer(units=64, activation=ActivationFunction(kind="relu"),
                   dropout=0.1, batchnorm=False),
        DenseLayer(units=32, activation=ActivationFunction(kind="relu"),
                   dropout=0.0, batchnorm=False),
    ],
    classification_units=7,
    input_adapter=InputAdapter(reduce_3d="flatten", scale="standard", allow_mixed_dims=True)
)

# ------------------------------------------------------------------------------
# Ejemplo 4: Uso con metadatos de Experiment (recomendado)
# ------------------------------------------------------------------------------
from backend.classes.Experiment import Experiment

# Cargar experimento y extraer metadatos
experiment = Experiment._load_latest_experiment()
metadata = SVNN.extract_metadata_from_experiment(experiment.dict())

model_metadata = SVNN(
    learning_rate=0.001,
    epochs=60,
    batch_size=64,
    layers=[
        DenseLayer(units=256, activation=ActivationFunction(kind="relu"),
                   dropout=0.3, batchnorm=True),
        DenseLayer(units=128, activation=ActivationFunction(kind="relu"),
                   dropout=0.2, batchnorm=True),
        DenseLayer(units=64, activation=ActivationFunction(kind="tanh"),
                   dropout=0.1),
    ],
    classification_units=5,
    input_adapter=InputAdapter(reduce_3d="mean_time_flat", scale="standard")
)

# Entrenar con metadatos (input_adapter maneja la transformación)
metrics = SVNN.train(
    model_metadata,
    xTest=['test1.npy', 'test2.npy'],
    yTest=['label1.npy', 'label2.npy'],
    xTrain=['train1.npy', 'train2.npy'],
    yTrain=['label_train1.npy', 'label_train2.npy'],
    metadata_train=metadata,
    metadata_test=metadata
)

# ------------------------------------------------------------------------------
# Ejemplo 5: Reducción mean_time_flat para espectrogramas
# ------------------------------------------------------------------------------
model_spectrogram = SVNN(
    learning_rate=0.001,
    epochs=70,
    batch_size=32,
    layers=[
        DenseLayer(units=512, activation=ActivationFunction(kind="relu"),
                   dropout=0.4, batchnorm=True, kernel_initializer="he_normal"),
        DenseLayer(units=256, activation=ActivationFunction(kind="relu"),
                   dropout=0.3, batchnorm=True, kernel_initializer="he_normal"),
        DenseLayer(units=128, activation=ActivationFunction(kind="relu"),
                   dropout=0.2),
    ],
    classification_units=3,
    input_adapter=InputAdapter(
        reduce_3d="mean_time_flat",  # Promedio sobre tiempo, luego flatten
        scale="standard",
        allow_mixed_dims=False
    )
)

# ------------------------------------------------------------------------------
# Ejemplo 6: Regularización L1 para selección de features
# ------------------------------------------------------------------------------
model_l1 = SVNN(
    learning_rate=0.001,
    epochs=80,
    batch_size=64,
    layers=[
        DenseLayer(
            units=256,
            activation=ActivationFunction(kind="relu"),
            dropout=0.3,
            batchnorm=True,
            kernel_regularizer="l1",  # L1 para sparsity
            regularizer_value=0.01
        ),
        DenseLayer(
            units=128,
            activation=ActivationFunction(kind="relu"),
            dropout=0.2,
            kernel_regularizer="l1",
            regularizer_value=0.005
        ),
        DenseLayer(
            units=64,
            activation=ActivationFunction(kind="relu"),
            dropout=0.1
        ),
    ],
    classification_units=4,
    input_adapter=InputAdapter(reduce_3d="flatten", scale="standard")
)

# ------------------------------------------------------------------------------
# Ejemplo 7: Combinación L1+L2 (ElasticNet)
# ------------------------------------------------------------------------------
model_elastic = SVNN(
    learning_rate=5e-4,
    epochs=100,
    batch_size=32,
    layers=[
        DenseLayer(
            units=512,
            activation=ActivationFunction(kind="relu"),
            dropout=0.4,
            batchnorm=True,
            kernel_regularizer="l1_l2",  # Combinación L1 y L2
            regularizer_value=0.01
        ),
        DenseLayer(
            units=256,
            activation=ActivationFunction(kind="relu"),
            dropout=0.3,
            batchnorm=True,
            kernel_regularizer="l1_l2",
            regularizer_value=0.01
        ),
        DenseLayer(
            units=128,
            activation=ActivationFunction(kind="relu"),
            dropout=0.2,
            batchnorm=False
        ),
    ],
    classification_units=6,
    input_adapter=InputAdapter(reduce_3d="flatten", scale="standard")
)

# ------------------------------------------------------------------------------
# Ejemplo 8: Modelo pequeño y rápido con Tanh
# ------------------------------------------------------------------------------
model_small = SVNN(
    learning_rate=0.01,  # Learning rate más alto para convergencia rápida
    epochs=30,
    batch_size=128,
    layers=[
        DenseLayer(units=128, activation=ActivationFunction(kind="tanh"),
                   dropout=0.2, batchnorm=True),
        DenseLayer(units=64, activation=ActivationFunction(kind="tanh"),
                   dropout=0.1),
    ],
    classification_units=2,
    input_adapter=InputAdapter(reduce_3d="flatten", scale="standard")
)

# ------------------------------------------------------------------------------
# Ejemplo 9: Arquitectura con activaciones mixtas
# ------------------------------------------------------------------------------
model_mixed = SVNN(
    learning_rate=0.001,
    epochs=60,
    batch_size=64,
    layers=[
        DenseLayer(units=256, activation=ActivationFunction(kind="gelu"),
                   dropout=0.3, batchnorm=True, kernel_initializer="he_normal"),
        DenseLayer(units=128, activation=ActivationFunction(kind="relu"),
                   dropout=0.2, batchnorm=True, kernel_initializer="glorot_uniform"),
        DenseLayer(units=64, activation=ActivationFunction(kind="tanh"),
                   dropout=0.1, kernel_initializer="glorot_uniform"),
        DenseLayer(units=32, activation=ActivationFunction(kind="sigmoid"),
                   dropout=0.0),
    ],
    classification_units=5,
    input_adapter=InputAdapter(reduce_3d="mean_time_flat", scale="standard")
)

# ------------------------------------------------------------------------------
# Ejemplo 10: Reducción mean_all para feature escalar por muestra
# ------------------------------------------------------------------------------
model_scalar = SVNN(
    learning_rate=0.01,
    epochs=50,
    batch_size=64,
    layers=[
        DenseLayer(units=64, activation=ActivationFunction(kind="relu"),
                   dropout=0.2, batchnorm=True),
        DenseLayer(units=32, activation=ActivationFunction(kind="relu"),
                   dropout=0.1),
    ],
    classification_units=3,
    input_adapter=InputAdapter(
        reduce_3d="mean_all",  # Promedia todo a un escalar
        scale="standard",
        allow_mixed_dims=False
    )
)

# ------------------------------------------------------------------------------
# Ejemplo 11: Modelo sin regularización para datasets pequeños
# ------------------------------------------------------------------------------
model_no_reg = SVNN(
    learning_rate=0.001,
    epochs=100,
    batch_size=16,  # Batch pequeño para datasets limitados
    layers=[
        DenseLayer(units=128, activation=ActivationFunction(kind="relu"),
                   dropout=0.0, batchnorm=False),  # Sin dropout ni batchnorm
        DenseLayer(units=64, activation=ActivationFunction(kind="relu"),
                   dropout=0.0, batchnorm=False),
        DenseLayer(units=32, activation=ActivationFunction(kind="relu"),
                   dropout=0.0, batchnorm=False),
    ],
    classification_units=2,
    input_adapter=InputAdapter(reduce_3d="flatten", scale="standard")
)

# ------------------------------------------------------------------------------
# Ejemplo 12: Configuración para datos de dimensiones mixtas
# ------------------------------------------------------------------------------
model_mixed_dims = SVNN(
    learning_rate=0.001,
    epochs=60,
    batch_size=32,
    layers=[
        DenseLayer(units=256, activation=ActivationFunction(kind="relu"),
                   dropout=0.3, batchnorm=True),
        DenseLayer(units=128, activation=ActivationFunction(kind="relu"),
                   dropout=0.2, batchnorm=True),
        DenseLayer(units=64, activation=ActivationFunction(kind="relu"),
                   dropout=0.1),
    ],
    classification_units=4,
    input_adapter=InputAdapter(
        reduce_3d="flatten",
        scale="standard",
        allow_mixed_dims=True  # Permite muestras de diferentes tamaños (padding automático)
    )
)

# ------------------------------------------------------------------------------
# NOTAS IMPORTANTES
# ------------------------------------------------------------------------------
# 1. InputAdapter maneja la transformación de datos arbitrarios (1D, 2D, 3D) a vectores 1D
# 2. reduce_3d controla cómo se reduce un tensor 3D:
#    - "flatten": Aplana todo (N, H, W) -> (N, H*W)
#    - "mean_time_flat": Promedio sobre eje 0 (tiempo), luego flatten
#    - "mean_all": Reduce todo a un escalar por muestra
# 3. Los metadatos son opcionales; InputAdapter usa heurísticas internas
# 4. kernel_regularizer soporta: "l1", "l2", "l1_l2" (ElasticNet)
# 5. kernel_initializer soporta: "glorot_uniform", "he_normal", "glorot_normal", etc.
# 6. SVNN ahora usa TensorFlow/Keras en lugar de PyTorch
# 7. BatchNormalization se aplica antes de la activación si batchnorm=True
# 8. Dropout se aplica después de la activación
"""