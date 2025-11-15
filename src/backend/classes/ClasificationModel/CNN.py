from __future__ import annotations
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union
import math
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema
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

# === Tipos auxiliares ===
NDArray = np.ndarray
PaddingMode = Literal["valid", "same"]
PoolType = Literal["max", "avg"]
ActName = Literal["relu", "tanh", "sigmoid", "gelu", "softmax", "linear"]

# =========================================================
# 1) Kernel: matriz 2D con validadores
# =========================================================
class Kernel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: SkipJsonSchema[NDArray] = Field(..., description="Matriz 2D (kh, kw) con el kernel/convolución.")
    name: Optional[str] = Field(default=None, description="Etiqueta opcional para el kernel (diagnóstico).")

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, w: NDArray) -> NDArray:
        if not isinstance(w, np.ndarray):
            raise TypeError("Kernel.weights debe ser un np.ndarray")
        if w.ndim != 2:
            raise ValueError(f"Kernel debe ser 2D (kh, kw). Recibido ndim={w.ndim}")
        kh, kw = w.shape
        if kh < 1 or kw < 1:
            raise ValueError("Kernel no puede tener dimensiones vacías.")
        if not np.issubdtype(w.dtype, np.number):
            raise TypeError("Kernel debe contener valores numéricos.")
        return w.astype(np.float32, copy=False)

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(int(x) for x in self.weights.shape)  # (kh, kw)


# =========================================================
# 2) Función de activación (descriptor)
# =========================================================
class ActivationFunction(BaseModel):
    kind: ActName = Field("relu", description="Tipo de activación.")

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, v: ActName) -> ActName:
        return v


# =========================================================
# 3) Capa Convolucional (tupla de 3 kernels por filtro: RGB)
# =========================================================
class ConvolutionLayer(BaseModel):
    kernels: List[Tuple[Kernel, Kernel, Kernel]] = Field(..., description="Lista de filtros; cada filtro es (kR, kG, kB).")
    stride: Tuple[int, int] = Field((1, 1), description="(stride_h, stride_w)")
    padding: PaddingMode = Field("same", description="'same' o 'valid'")
    activation: ActivationFunction = Field(default_factory=lambda: ActivationFunction(kind="relu"))

    @field_validator("kernels")
    @classmethod
    def _validate_kernels(cls, ks: List[Tuple[Kernel, Kernel, Kernel]]) -> List[Tuple[Kernel, Kernel, Kernel]]:
        if len(ks) < 1:
            raise ValueError("Debe haber al menos 1 filtro en la capa convolucional.")
        for kR, kG, kB in ks:
            if kR.shape != kG.shape or kR.shape != kB.shape:
                raise ValueError("Los 3 kernels de cada filtro (R,G,B) deben tener la misma forma (kh, kw).")
        return ks

    @field_validator("stride")
    @classmethod
    def _validate_stride(cls, s: Tuple[int, int]) -> Tuple[int, int]:
        sh, sw = s
        if sh < 1 or sw < 1:
            raise ValueError("Stride debe ser >= 1 en ambas direcciones.")
        return s

    def num_filters(self) -> int:
        return len(self.kernels)

    def kernel_shape(self) -> Tuple[int, int]:
        return self.kernels[0][0].shape

    def output_shape(self, h_in: int, w_in: int, c_in: int) -> Tuple[int, int, int]:
        if c_in != 3:
            raise ValueError(f"ConvolutionLayer espera entrada con 3 canales (RGB). Recibido C_in={c_in}.")
        kh, kw = self.kernel_shape()
        sh, sw = self.stride
        if self.padding == "same":
            H_out = math.ceil(h_in / sh)
            W_out = math.ceil(w_in / sw)
        else:  # valid
            H_out = math.floor((h_in - kh + 1) / sh)
            W_out = math.floor((w_in - kw + 1) / sw)
            if H_out < 1 or W_out < 1:
                raise ValueError("Convolución 'valid' produce dimensión negativa/cero: revisa kernel/stride.")
        C_out = self.num_filters()
        return int(H_out), int(W_out), int(C_out)


# =========================================================
# 4) Capa de Pooling
# =========================================================
class PoolingLayer(BaseModel):
    kind: PoolType = Field("max", description="'max' o 'avg'")
    pool_size: Tuple[int, int] = Field((2, 2), description="(ph, pw)")
    stride: Optional[Tuple[int, int]] = Field(
        None, description="Stride del pooling; si None, se iguala a pool_size."
    )
    padding: PaddingMode = Field("valid", description="'same' o 'valid'.")

    @field_validator("pool_size")
    @classmethod
    def _validate_pool(cls, p: Tuple[int, int]) -> Tuple[int, int]:
        ph, pw = p
        if ph < 1 or pw < 1:
            raise ValueError("pool_size debe ser >= 1.")
        return p

    def output_shape(self, h_in: int, w_in: int, c_in: int) -> Tuple[int, int, int]:
        ph, pw = self.pool_size
        sh, sw = self.stride or self.pool_size
        if self.padding == "same":
            H_out = math.ceil(h_in / sh)
            W_out = math.ceil(w_in / sw)
        else:
            H_out = math.floor((h_in - ph) / sh) + 1
            W_out = math.floor((w_in - pw) / sw) + 1
            if H_out < 1 or W_out < 1:
                raise ValueError("Pooling 'valid' produce dimensión negativa/cero: revisa pool_size/stride.")
        return int(H_out), int(W_out), int(c_in)


# =========================================================
# 5) Capa densa (red)
# =========================================================
class DenseLayer(BaseModel):
    units: int = Field(..., ge=1, description="Número de neuronas.")
    activation: ActivationFunction = Field(default_factory=lambda: ActivationFunction(kind="relu"))

    @field_validator("units")
    @classmethod
    def _validate_units(cls, v: int) -> int:
        if v < 1:
            raise ValueError("DenseLayer.units debe ser >= 1.")
        return v


# =========================================================
# 6) Flatten (como capa)
# =========================================================
class FlattenLayer(BaseModel):
    name: Optional[str] = Field(default=None, description="Etiqueta opcional.")


# =========================================================
# 7) CNN completa (siempre per-frame)
# =========================================================
class CNN(BaseModel):
    feature_extractor: List[Union[ConvolutionLayer, PoolingLayer]] = Field(
        ..., description="Secuencia de capas conv/pooling en cualquier orden."
    )
    flatten: FlattenLayer = Field(default_factory=FlattenLayer)
    fc_layers: List[DenseLayer] = Field(default_factory=list, description="Capas densas intermedias.")
    fc_activation_common: ActivationFunction = Field(
        default_factory=lambda: ActivationFunction(kind="relu"),
        description="Activación común para fc_layers (se impone al instanciar)."
    )
    classification: DenseLayer = Field(..., description="Capa de salida con activación softmax (n_clases = units).")

    # NUEVO: hiperparámetros para crear imágenes por frame
    frame_context: int = Field(8, ge=0, description="nº de frames a cada lado para el 'ancho' de la imagen (W=2*context+1).")
    image_hw: Tuple[int, int] = Field((64, 128), description="(alto, ancho) destino de cada imagen.")

    # === Validaciones ===
    @field_validator("feature_extractor")
    @classmethod
    def _validate_fe_not_empty(cls, layers):
        if len(layers) == 0:
            raise ValueError("feature_extractor no puede estar vacío.")
        return layers

    @field_validator("fc_layers")
    @classmethod
    def _validate_fc_activation(cls, layers, info):
        values = info.data
        common: ActivationFunction = values.get("fc_activation_common", ActivationFunction(kind="relu"))
        for i, lyr in enumerate(layers):
            lyr.activation = common
            layers[i] = lyr
        return layers

    @field_validator("classification")
    @classmethod
    def _validate_softmax(cls, lyr: DenseLayer) -> DenseLayer:
        if lyr.activation.kind != "softmax":
            raise ValueError("La capa de clasificación DEBE usar activación 'softmax'.")
        return lyr

    # === Dimensionalidad para Flatten ===
    def infer_output_shape_after_fe(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        H, W, C = input_shape
        seen_conv = False
        for layer in self.feature_extractor:
            if isinstance(layer, ConvolutionLayer):
                if C != 3:
                    raise ValueError(
                        f"ConvolutionLayer espera C_in=3 (RGB). Recibido C={C}. "
                        "Extiende Kernel a (C_in,kh,kw) por filtro si quieres convs profundas."
                    )
                H, W, C = layer.output_shape(H, W, C)
                seen_conv = True
            else:
                H, W, C = layer.output_shape(H, W, C)
        if not seen_conv:
            raise ValueError("Debe existir al menos una ConvolutionLayer en el feature_extractor.")
        return H, W, C

    def flatten_size(self, input_shape: Tuple[int, int, int]) -> int:
        H, W, C = self.infer_output_shape_after_fe(input_shape)
        return int(H * W * C)

    # =================================================================================
    # Persistencia: Guardar/Cargar modelo entrenado
    # =================================================================================
    _tf_model: Optional[object] = None

    def save(self, path: str):
        """
        Guarda la instancia completa (arquitectura + modelo entrenado) en disco usando pickle.
        
        Args:
            path: Ruta donde guardar el archivo .pkl
        
        Example:
            cnn.save("src/backend/models/p300/cnn_20251109_143022.pkl")
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[CNN] Modelo guardado en: {path}")
    
    @classmethod
    def load(cls, path: str) -> "CNN":
        """
        Carga una instancia completa desde disco.
        
        Args:
            path: Ruta al archivo .pkl guardado previamente
        
        Returns:
            Instancia de CNN con modelo entrenado listo para query()
        
        Example:
            cnn_model = CNN.load("src/backend/models/p300/cnn_20251109_143022.pkl")
            predictions = CNN.query(cnn_model, images)
        """
        with open(path, 'rb') as f:
            instance = pickle.load(f)
        print(f"[CNN] Modelo cargado desde: {path}")
        return instance
    
    @staticmethod
    def _generate_model_path(label: str, base_dir: str = "src/backend/models") -> str:
        """
        Genera ruta única para guardar modelo.
        
        Args:
            label: Etiqueta del experimento (e.g., "p300", "inner")
            base_dir: Directorio base para modelos
        
        Returns:
            Ruta completa: "{base_dir}/{label}/cnn_{timestamp}.pkl"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_dir = Path(base_dir) / label
        label_dir.mkdir(parents=True, exist_ok=True)
        return str(label_dir / f"cnn_{timestamp}.pkl")

    # -------------------- Helpers de metadatos --------------------
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
            metadata = CNN.extract_metadata_from_experiment(experiment.dict(), [0, 1])
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

    # -------------------- Helpers de formato --------------------
    @staticmethod
    def _ensure_tc(x: NDArray) -> NDArray:
        """Normaliza a (n_times, n_channels)."""
        if x.ndim != 2:
            raise ValueError("Se esperaba 2D para _ensure_tc.")
        return x if x.shape[0] >= x.shape[1] else x.T

    @staticmethod
    def _pca_to_3_channels(cube_hw_c: NDArray) -> NDArray:
        H, W, C = cube_hw_c.shape
        X = cube_hw_c.reshape(-1, C).astype(np.float32)
        X -= X.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        X3 = (U[:, :3] * S[:3]).astype(np.float32)
        out = X3.reshape(H, W, 3)
        mn = out.min(axis=(0,1), keepdims=True); mx = out.max(axis=(0,1), keepdims=True) + 1e-8
        return (out - mn) / (mx - mn)

    @staticmethod
    def _cube_to_rgb(cube_fwc: NDArray) -> NDArray:
        """(F,Freqs,C) -> (Freqs,F,3) como imagen base por 'tira temporal'."""
        F, Freqs, C = cube_fwc.shape
        if C == 3:
            base = cube_fwc.astype(np.float32, copy=False)
        elif C == 1:
            base = np.repeat(cube_fwc, 3, axis=2).astype(np.float32, copy=False)
        elif C == 2:
            third = cube_fwc.mean(axis=2, keepdims=True).astype(np.float32)
            base = np.concatenate([cube_fwc.astype(np.float32), third], axis=2)
        else:
            base = CNN._pca_to_3_channels(cube_fwc.astype(np.float32))
        # salimos como (F,Freqs,3) y externamente reordenamos a (H,W,3) = (Freqs,window,3)
        return base

    @staticmethod
    def     _resize_2d_bilinear(src: NDArray, H: int, W: int) -> NDArray:
        h, w = src.shape
        if (h, w) == (H, W):
            return src.astype(np.float32, copy=False)
        ry = (h - 1) / max(1, (H - 1)); rx = (w - 1) / max(1, (W - 1))
        dst = np.empty((H, W), dtype=np.float32)
        for i in range(H):
            y = i * ry; y0, y1 = int(math.floor(y)), min(int(math.ceil(y)), h - 1); wy = y - y0
            for j in range(W):
                x = j * rx; x0, x1 = int(math.floor(x)), min(int(math.ceil(x)), w - 1); wx = x - x0
                v00 = src[y0, x0]; v01 = src[y0, x1]; v10 = src[y1, x0]; v11 = src[y1, x1]
                dst[i, j] = (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)
        return dst

    @staticmethod
    def _temporal_texture(x_tc: NDArray, H: int, W: int) -> NDArray:
        sig = x_tc.mean(axis=1)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        L = H * W
        if sig.shape[0] < L:
            reps = int(np.ceil(L / sig.shape[0])); sig = np.tile(sig, reps)[:L]
        else:
            sig = sig[:L]
        return sig.reshape(H, W)

    # ---------------- Imágenes per-frame ----------------
    @classmethod
    def _images_from_spec_per_frame(
        cls, cube_fwc: NDArray, frame_labels: NDArray, k_ctx: int, target_hw: Tuple[int,int]
    ) -> Tuple[NDArray, NDArray]:
        """
        cube: (F, Freqs, C). Para cada frame i, imagen = (Freqs, 2k+1, 3) con contexto.
        Devuelve X: (N, H, W, 3) y y: (N,)
        """
        base = cls._cube_to_rgb(cube_fwc)            # (F,Freqs,3)
        F, Freqs, _ = base.shape
        if frame_labels.size != F:
            raise ValueError(f"labels por frame ({frame_labels.size}) != n_frames ({F})")
        Ht, Wt = target_hw
        imgs: List[NDArray] = []
        ys: List[int] = []
        for i in range(F):
            lo = max(0, i - k_ctx); hi = min(F, i + k_ctx + 1)
            slice_ = base[lo:hi]                     # (W_ctx, Freqs, 3)
            img = np.transpose(slice_, (1, 0, 2))    # (Freqs, W_ctx, 3)
            # reescala
            if img.shape[:2] != target_hw:
                img = np.stack([
                    cls._resize_2d_bilinear(img[:,:,0], Ht, Wt),
                    cls._resize_2d_bilinear(img[:,:,1], Ht, Wt),
                    cls._resize_2d_bilinear(img[:,:,2], Ht, Wt),
                ], axis=2)
            imgs.append(img.astype(np.float32))
            ys.append(int(frame_labels[i]))
        X = np.stack(imgs, axis=0)
        y = np.array(ys, dtype=np.int64)
        return X, y


    # ----------------- CARGA X,y (siempre per-frame) -----------------
    @classmethod
    def _prepare_images_and_labels(
        cls,
        x_paths: Sequence[str],
        y_paths: Sequence[str],
        image_hw: Tuple[int, int],
        k_ctx: int,
        metadata_list: Optional[Sequence[dict]] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        REGLA DE NEGOCIO: Todas las señales DEBEN pasar por una transformada.
        Por tanto, X SIEMPRE es 3D con formato: (n_frames, features, n_channels)
        donde axis0 = ejemplos (ventanas/frames).

        Genera imágenes RGB (H, W, 3) a partir de cada frame usando contexto temporal.

        Args:
            x_paths: Rutas a archivos .npy con datos post-transform 3D
            y_paths: Rutas a archivos .npy con etiquetas (n_frames,)
            image_hw: Tamaño destino de las imágenes (H, W)
            k_ctx: Contexto de frames para ventanas (frames antes/después)
            metadata_list: OPCIONAL - metadatos de transforms (ya no usado)

        Returns:
            Tupla (X, y) donde:
                X: (n_total_frames, H, W, 3) - imágenes RGB
                y: (n_total_frames,) - etiquetas
        """
        if len(x_paths) != len(y_paths):
            raise ValueError("Se requiere correspondencia 1-1 entre x_paths e y_paths.")

        X_all: List[NDArray] = []
        y_all: List[NDArray] = []

        for idx, (xp, yp) in enumerate(zip(x_paths, y_paths)):
            X = np.load(xp, allow_pickle=False).astype(np.float32)
            y = np.load(yp, allow_pickle=False).reshape(-1).astype(np.int64)

            # Validar que X es 3D (regla de negocio)
            if X.ndim != 3:
                raise ValueError(
                    f"Los datos deben ser 3D (n_frames, features, n_channels) después de aplicar transform. "
                    f"Recibido shape={X.shape} en {xp}. "
                    f"Asegúrate de aplicar WindowingTransform, FFTTransform, DCTTransform o WaveletTransform."
                )

            n_frames, features, n_channels = X.shape

            if y.size != n_frames:
                raise ValueError(
                    f"Mismatch entre frames y etiquetas en {xp}: "
                    f"X tiene {n_frames} frames pero y tiene {y.size} labels."
                )

            # Convertir a imágenes RGB usando el método existente
            # X tiene formato (n_frames, features, n_channels) que es equivalente a "spec_3d"
            Xi, yi = cls._images_from_spec_per_frame(X, y, k_ctx, image_hw)

            X_all.append(Xi)
            y_all.append(yi)

        X_out = np.concatenate(X_all, axis=0)
        y_out = np.concatenate(y_all, axis=0)
        return X_out.astype(np.float32), y_out.astype(np.int64)

    # =====================================================
    # Entrenamiento (per-frame)
    # =====================================================
    @classmethod
    def train(
        cls,
        instance: "CNN",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
        metadata_train: Optional[List[dict]] = None,
        metadata_test: Optional[List[dict]] = None,
        model_label: Optional[str] = None,
    ):
        """
        Entrena una CNN usando TensorFlow/Keras (wrapper legacy).
        
        Este método mantiene compatibilidad con código existente retornando solo EvaluationMetrics.
        Internamente delega a fit() para evitar duplicación de código.

        Args:
            instance: Instancia de CNN con la arquitectura configurada
            xTest: Lista de rutas a archivos .npy de test
            yTest: Lista de rutas a archivos .npy con etiquetas de test
            xTrain: Lista de rutas a archivos .npy de entrenamiento
            yTrain: Lista de rutas a archivos .npy con etiquetas de entrenamiento
            metadata_train: Lista de diccionarios con metadatos de dimensionality_change para train
            metadata_test: Lista de diccionarios con metadatos para test
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
            return_history=False,
            model_label=model_label
        )
        return result.metrics

    @classmethod
    def fit(
        cls,
        instance: "CNN",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
        metadata_train: Optional[List[dict]] = None,
        metadata_test: Optional[List[dict]] = None,
        return_history: bool = True,
        model_label: Optional[str] = None,
        *,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> TrainResult:
        """
        Entrena y devuelve paquete TrainResult (modelo + métricas + historia).

        1) Carga artefactos y genera imágenes per-frame (cada ventana = 1 imagen).
        2) Entrena una CNN con TensorFlow que respeta tu feature_extractor + FC + Softmax.

        Args:
            instance: Instancia de CNN con la arquitectura configurada
            xTest: Lista de rutas a archivos .npy de test
            yTest: Lista de rutas a archivos .npy con etiquetas de test
            xTrain: Lista de rutas a archivos .npy de entrenamiento
            yTrain: Lista de rutas a archivos .npy con etiquetas de entrenamiento
            metadata_train: Lista de diccionarios con metadatos de dimensionality_change para train.
                           Cada diccionario debe contener:
                           - 'output_axes_semantics': dict con semántica de ejes (e.g., {"axis0": "time", "axis1": "channels"})
                           - 'output_shape': tuple/list con forma de salida (opcional si hay semantics)
                           Ejemplos:
                           [
                               {"output_axes_semantics": {"axis0": "frequency", "axis1": "time", "axis2": "channels"},
                                "output_shape": (128, 256, 3)},
                               {"output_axes_semantics": {"axis0": "time", "axis1": "channels"},
                                "output_shape": (1000, 64)}
                           ]
            metadata_test: Lista de diccionarios con metadatos para test (misma estructura)
            return_history: Si True, incluye historial completo de entrenamiento en TrainResult
            model_label: Etiqueta opcional para auto-guardar (e.g., "p300", "inner").
                        Si se proporciona, guarda automáticamente en src/backend/models/{label}/
            
        Returns:
            TrainResult con métricas, modelo, historial y hiperparámetros
        """
        try:
            Xtr, ytr = cls._prepare_images_and_labels(
                xTrain, yTrain,
                image_hw=instance.image_hw,
                k_ctx=instance.frame_context,
                metadata_list=metadata_train
            )
            Xte, yte = cls._prepare_images_and_labels(
                xTest, yTest,
                image_hw=instance.image_hw,
                k_ctx=instance.frame_context,
                metadata_list=metadata_test
            )
        except Exception as e:
            raise RuntimeError(f"Error preparando dataset per-frame: {e}") from e

        # Verifica coherencia del grafo (dim flatten)
        H_in, W_in = instance.image_hw
        _ = instance.flatten_size((H_in, W_in, 3))

        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, models

            # Determinar dispositivo (GPU si está disponible)
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU config error: {e}")

            def build_cnn_model(spec: CNN, in_shape=(64, 128, 3)):
                """Construye modelo CNN usando Keras Sequential API"""
                model_layers = []

                # Feature extractor
                for block in spec.feature_extractor:
                    if isinstance(block, ConvolutionLayer):
                        kh, kw = block.kernel_shape()
                        model_layers.append(layers.Conv2D(
                            filters=block.num_filters(),
                            kernel_size=(kh, kw),
                            strides=block.stride,
                            padding=block.padding,
                            use_bias=True,
                            input_shape=in_shape if len(model_layers) == 0 else None
                        ))
                        # Activación
                        act = block.activation.kind
                        if act == "relu":
                            model_layers.append(layers.ReLU())
                        elif act == "tanh":
                            model_layers.append(layers.Activation('tanh'))
                        elif act == "gelu":
                            model_layers.append(layers.Activation('gelu'))
                        elif act == "sigmoid":
                            model_layers.append(layers.Activation('sigmoid'))
                        else:
                            model_layers.append(layers.ReLU())
                    else:  # PoolingLayer
                        ph, pw = block.pool_size
                        sh, sw = block.stride or block.pool_size
                        if block.kind == "max":
                            model_layers.append(layers.MaxPooling2D(
                                pool_size=(ph, pw),
                                strides=(sh, sw),
                                padding=block.padding
                            ))
                        else:  # avg
                            model_layers.append(layers.AveragePooling2D(
                                pool_size=(ph, pw),
                                strides=(sh, sw),
                                padding=block.padding
                            ))

                # Flatten
                model_layers.append(layers.Flatten())

                # FC layers
                for d in spec.fc_layers:
                    model_layers.append(layers.Dense(d.units))
                    act = spec.fc_activation_common.kind
                    if act == "relu":
                        model_layers.append(layers.ReLU())
                    elif act == "tanh":
                        model_layers.append(layers.Activation('tanh'))
                    elif act == "gelu":
                        model_layers.append(layers.Activation('gelu'))
                    elif act == "sigmoid":
                        model_layers.append(layers.Activation('sigmoid'))

                # Capa de salida (clasificación)
                model_layers.append(layers.Dense(spec.classification.units, activation='softmax'))

                return models.Sequential(model_layers)

            n_classes = int(max(int(ytr.max()), int(yte.max()))) + 1
            H_in, W_in = instance.image_hw

            # Limpiar modelo previo (cada fit() reconstruye desde cero)
            instance._tf_model = None

            # Hiperparámetros (si no se proporcionan, usar defaults legacy)
            lr_val = float(learning_rate) if learning_rate is not None else 1e-3
            bs_val = int(batch_size) if batch_size is not None else 64
            ep_val = int(epochs) if epochs is not None else 2

            # IMPORTANTE: Configurar GPU ANTES de construir el modelo
            import time
            import tensorflow as tf
            import gc

            # Limpiar sesión anterior
            tf.keras.backend.clear_session()
            gc.collect()

            # Configurar estrategia de entrenamiento con fallback GPU -> CPU
            def train_with_fallback():
                """Intenta entrenar en GPU, si falla cae a CPU automáticamente"""
                nonlocal bs_val

                # Fase 1: Intentar con GPU optimizada
                print("🚀 [CNN] Intentando entrenamiento en GPU con optimizaciones...")

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
                        model = build_cnn_model(instance, in_shape=(H_in, W_in, 3))
                        model.compile(
                            optimizer=keras.optimizers.Adam(learning_rate=lr_val),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )

                        # Entrenar en GPU
                        t0 = time.perf_counter()
                        history = model.fit(
                            Xtr, ytr,
                            batch_size=gpu_batch_size,
                            epochs=ep_val,
                            verbose=1,
                            validation_data=(Xte, yte)
                        )
                        train_time = time.perf_counter() - t0

                        print(f"✅ [CNN] Entrenamiento en GPU completado exitosamente!")
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
                print("🔄 [CNN] Cambiando a entrenamiento en CPU...")
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
                    model = build_cnn_model(instance, in_shape=(H_in, W_in, 3))
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=lr_val),
                        loss="sparse_categorical_crossentropy",
                        metrics=['accuracy']
                    )

                    # Entrenar en CPU con batch_size reducido
                    t0 = time.perf_counter()
                    history = model.fit(
                        Xtr, ytr,
                        batch_size=cpu_batch_size,
                        epochs=ep_val,
                        verbose=1,
                        validation_data=(Xte, yte)
                    )
                    train_time = time.perf_counter() - t0

                    print(f"✅ [CNN] Entrenamiento en CPU completado exitosamente!")
                    print(f"   Tiempo: {train_time:.2f}s | Batch size: {cpu_batch_size}")
                    return history, train_time, cpu_batch_size, model, 'CPU'

            # Ejecutar entrenamiento con fallback
            history, train_time, final_batch_size, model, device_used = train_with_fallback()
            bs_val = final_batch_size  # Actualizar para predicción

            train_losses: List[float] = history.history['loss']

            # --- Evaluación y métricas ---
            # Predicciones - USAR EL MISMO DISPOSITIVO que entrenamiento
            if device_used == 'CPU':
                with tf.device('/CPU:0'):
                    logits = model.predict(Xte, batch_size=128, verbose=0)
            else:
                logits = model.predict(Xte, batch_size=128, verbose=0)
            y_pred = np.argmax(logits, axis=1)
            y_true = yte

            # Básicas
            acc  = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec  = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            f1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            cm   = confusion_matrix(y_true, y_pred).tolist()

            # AUC-ROC (multiclase One-vs-Rest con probabilidades)
            try:
                # logits ya contiene las probabilidades de softmax
                proba = logits
                auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))
            except Exception:
                auc = 0.0

            # Construye métricas
            metrics = EvaluationMetrics(
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1_score=f1,
                confusion_matrix=cm,
                auc_roc=auc,
                loss=train_losses,
                evaluation_time="",              
            )

            # Guardar modelo en la instancia
            instance._tf_model = model

            # Auto-guardar si se proporciona model_label
            if model_label:
                save_path = cls._generate_model_path(model_label)
                instance.save(save_path)
                print(f"[CNN] Modelo guardado automáticamente en: {save_path}")

            # Construir historial para TrainResult
            hist_dict = history.history if return_history else {}

            # Construir hiperparámetros
            hyperparams = {
                "epochs": ep_val,
                "batch_size": bs_val,
                "learning_rate": lr_val,
                "n_classes": n_classes,
                "image_hw": instance.image_hw,
                "frame_context": instance.frame_context,
                "n_train_images": len(Xtr),
                "n_test_images": len(Xte),
            }

            print(f"[CNN] Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")

            return TrainResult(
                metrics=metrics,
                model=instance,
                model_name="CNN",
                training_seconds=train_time,
                history=hist_dict,
                hyperparams=hyperparams,
            )


        except Exception as e:
            print("[CNN] Entrenamiento real no ejecutado (fallback).")
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
                model_name="CNN",
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
        instance: "CNN",
        data: List[NDArray],
        metadata_list: Optional[List[dict]] = None,
        batch_size: int = 128
    ) -> Tuple[NDArray, NDArray]:
        """
        Realiza inferencia con el modelo CNN entrenado.

        Args:
            instance: Instancia de CNN con modelo entrenado (_tf_model debe existir)
            data: Lista de imágenes como NDArray (cada una con shape (H, W, 3))
            metadata_list: Lista opcional de diccionarios con metadatos (no usado típicamente en CNN)
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
            
            # Validar que las imágenes tengan la forma correcta
            H_expected, W_expected = instance.image_hw
            for i, img in enumerate(data):
                if img.shape != (H_expected, W_expected, 3):
                    raise ValueError(
                        f"Imagen {i} tiene shape {img.shape}, esperado ({H_expected}, {W_expected}, 3)"
                    )

            # Convertir lista a array numpy
            X = np.stack(data, axis=0).astype(np.float32)

            # Predicción
            probabilities = instance._tf_model.predict(X, batch_size=batch_size, verbose=0)
            predictions = np.argmax(probabilities, axis=1)

            return predictions, probabilities

        except Exception as e:
            raise RuntimeError(f"Error en inferencia CNN: {e}") from e



# ======================= helpers internos =======================

def _resize_2d_bilinear(src: NDArray, H: int, W: int) -> NDArray:
    h, w = src.shape
    if h == H and w == W:
        return src.astype(np.float32, copy=False)
    ry = (h - 1) / max(1, (H - 1))
    rx = (w - 1) / max(1, (W - 1))
    dst = np.empty((H, W), dtype=np.float32)
    for i in range(H):
        y = i * ry
        y0, y1 = int(math.floor(y)), min(int(math.ceil(y)), h - 1)
        wy = y - y0
        for j in range(W):
            x = j * rx
            x0, x1 = int(math.floor(x)), min(int(math.ceil(x)), w - 1)
            wx = x - x0
            v00 = src[y0, x0]
            v01 = src[y0, x1]
            v10 = src[y1, x0]
            v11 = src[y1, x1]
            dst[i, j] = (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)
    return dst

def _tile_and_normalize_temporal(x: NDArray, H: int, W: int) -> NDArray:
    sig = np.mean(x, axis=0)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    L = H * W
    if sig.shape[0] < L:
        reps = int(np.ceil(L / sig.shape[0]))
        sig = np.tile(sig, reps)[:L]
    else:
        sig = sig[:L]
    return sig.reshape(H, W)


# ======================= Ejemplo rápido =======================
"""
# Construir arquitectura
k3 = Kernel(weights=np.ones((3,3), dtype=np.float32)/9.0)
conv1 = ConvolutionLayer(kernels=[(k3,k3,k3)]*16, stride=(1,1), padding="same", activation=ActivationFunction(kind="relu"))
conv2 = ConvolutionLayer(kernels=[(k3,k3,k3)]*32, stride=(1,1), padding="same", activation=ActivationFunction(kind="relu"))
pool = PoolingLayer(kind="max", pool_size=(2,2))

cnn = CNN(
    feature_extractor=[conv1, conv2, pool],
    fc_layers=[DenseLayer(units=128), DenseLayer(units=64)],
    fc_activation_common=ActivationFunction(kind="relu"),
    classification=DenseLayer(units=5, activation=ActivationFunction(kind="softmax")),
    frame_context=8,
    image_hw=(64,128)
)

# Flatten para imágenes (64x128x3)
print("Flatten size:", cnn.flatten_size((64,128,3)))

# Ejemplo 1: Metadatos manuales (si no usas Experiment)
metadata_train = [
    {"output_axes_semantics": {"axis0": "frequency", "axis1": "time", "axis2": "channels"},
     "output_shape": (128, 256, 3)},
    {"output_axes_semantics": {"axis0": "time", "axis1": "channels"},
     "output_shape": (1000, 64)}
]

metadata_test = [
    {"output_axes_semantics": {"axis0": "frequency", "axis1": "time", "axis2": "channels"},
     "output_shape": (128, 256, 3)}
]

# metrics = CNN.train(
#     cnn,
#     xTest=["path/to/test1.npy"],
#     yTest=["path/to/test1_labels.npy"],
#     xTrain=["path/to/train1.npy", "path/to/train2.npy"],
#     yTrain=["path/to/train1_labels.npy", "path/to/train2_labels.npy"],
#     metadata_train=metadata_train,
#     metadata_test=metadata_test
# )

# Ejemplo 2: Extraer metadatos desde Experiment (RECOMENDADO)
# from backend.classes.Experiment import Experiment
#
# experiment = Experiment._load_latest_experiment()
# metadata_train = CNN.extract_metadata_from_experiment(experiment.dict(), transform_indices=[0, 1])
# metadata_test = CNN.extract_metadata_from_experiment(experiment.dict(), transform_indices=[0])
#
# metrics = CNN.train(
#     cnn,
#     xTest=["path/to/test1.npy"],
#     yTest=["path/to/test1_labels.npy"],
#     xTrain=["path/to/train1.npy", "path/to/train2.npy"],
#     yTrain=["path/to/train1_labels.npy", "path/to/train2_labels.npy"],
#     metadata_train=metadata_train,
#     metadata_test=metadata_test
# )

# Ejemplo 3: Sin metadatos (fallback heurístico - menos robusto)
# metrics = CNN.train(cnn, xTest=[...], yTest=[...], xTrain=[...], yTrain=[...])
"""
