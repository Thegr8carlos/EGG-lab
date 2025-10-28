from __future__ import annotations
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union
import math
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema
from backend.classes.Metrics import EvaluationMetrics

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
<<<<<<< HEAD
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: SkipJsonSchema[NDArray] = Field(..., description="Matriz 2D (kh, kw) con el kernel/convolución.")
=======
    model_config = {"arbitrary_types_allowed": True}

    weights: NDArray = Field(..., description="Matriz 2D (kh, kw) con el kernel/convolución.")
>>>>>>> 3df3b9c710a5e124b94347711d5d01051b9026c3
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

    # -------------------- Helpers de formato --------------------
    @staticmethod
    def _detect_kind(x: NDArray) -> str:
        """
        Devuelve: 'spec_3d' (n_frames,n_freqs,n_channels) o 'signal_2d' (n_times,n_channels) o 'image_3d' (H,W,3).
        """
        if x.ndim == 3:
            H, W, C = x.shape
            if C == 3 and (H > 8 and W > 8):  # heurística
                return "image_3d"
            return "spec_3d"
        if x.ndim == 2:
            return "signal_2d"
        raise ValueError(f"Entrada no soportada: ndim={x.ndim}")

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

    @classmethod
    def _images_from_signal_per_frame(
        cls, x_tc: NDArray, frame_labels: NDArray, k_ctx: int, target_hw: Tuple[int,int]
    ) -> Tuple[NDArray, NDArray]:
        """
        x_tc: (n_times, n_channels). Divide tiempo en n_frames=len(labels) ventanas iguales.
        Para cada frame, saca espectro/dct simple y arma imagen con contexto (como spec).
        """
        x_tc = cls._ensure_tc(x_tc)
        n_times, n_ch = x_tc.shape
        F = int(frame_labels.size)
        if F < 1:
            raise ValueError("labels por frame vacías.")
        # Partición equi-espaciada
        L = int(math.ceil(n_times / F))
        # Generamos cubo 'espectrograma denso' ad-hoc: (F, n_freqs, C=3)
        nfft = min(256, 1 << int(np.ceil(np.log2(max(64, min(L, 256))))))
        win = np.hanning(nfft).astype(np.float32)
        n_freqs = nfft // 2 + 1
        cube = np.empty((F, n_freqs, 3), dtype=np.float32)
        for i in range(F):
            s = i * L
            e = min(s + L, n_times)
            seg = x_tc[s:e].T  # (n_ch, seg_len)
            if seg.shape[1] < nfft:
                pad = np.pad(seg, ((0,0),(0, nfft - seg.shape[1])), mode="constant")
            else:
                pad = seg[:, :nfft]
            # R: espectro promedio
            frames = (pad * win).astype(np.float32)
            S = np.fft.rfft(frames, n=nfft, axis=1)           # (n_ch, n_freqs)
            P = (np.abs(S) ** 2).mean(axis=0)                  # (n_freqs,)
            R = np.log1p(P)
            # G: usa P directo (o DCT si quieres)
            G = P.copy()
            # B: energía por canal resumida a n_freqs (tile)
            E = (pad**2).mean(axis=1, keepdims=True)           # (n_ch,1)
            B = np.tile(E.mean(), n_freqs)
            # normaliza por canal y apila
            def _mm(a):
                a = a - a.min(); return a / (a.max() + 1e-8)
            cube[i,:,0] = _mm(R)
            cube[i,:,1] = _mm(G)
            cube[i,:,2] = _mm(B)
        # Reutiliza el generador por contexto
        return cls._images_from_spec_per_frame(cube, frame_labels, k_ctx, target_hw)

    # ----------------- CARGA X,y (siempre per-frame) -----------------
    @classmethod
    def _prepare_images_and_labels(
        cls,
        x_paths: Sequence[str],
        y_paths: Sequence[str],
        image_hw: Tuple[int, int],
        k_ctx: int,
    ) -> Tuple[NDArray, NDArray]:
        """
        Siempre per-frame:
          - Si X es (F,Freqs,C): exige y con F etiquetas; genera (N=F, H,W,3).
          - Si X es (n_times,n_channels): trocea en F=len(y) segmentos; genera (N=F, H,W,3).
          - Si X es (H,W,3): NO se espera en este flujo, pero si aparece, lo tratamos como 'spec' con F=H y y.size==F.
        """
        if len(x_paths) != len(y_paths):
            raise ValueError("Se requiere correspondencia 1-1 entre x_paths e y_paths (per-frame).")
        X_all: List[NDArray] = []
        y_all: List[NDArray] = []
        for xp, yp in zip(x_paths, y_paths):
            X = np.load(xp, allow_pickle=True)
            y = np.load(yp, allow_pickle=True).reshape(-1)
            if y.size < 1:
                raise ValueError(f"Labels vacías para {yp}")
            kind = cls._detect_kind(X)
            if kind == "spec_3d":
                Xi, yi = cls._images_from_spec_per_frame(X.astype(np.float32), y.astype(np.int64), k_ctx, image_hw)
            elif kind == "signal_2d":
                Xi, yi = cls._images_from_signal_per_frame(X.astype(np.float32), y.astype(np.int64), k_ctx, image_hw)
            else:  # image_3d (raro aquí); interpretamos H como frames y W como 'freqs'
                H,W,_ = X.shape
                if y.size != H:
                    raise ValueError(f"Para image_3d, se espera y.size==H. Recibido {y.size} vs {H}")
                # reconstruye pseudo-cubo (H,W,3) -> tratar como spec
                Xi, yi = cls._images_from_spec_per_frame(X, y.astype(np.int64), k_ctx, image_hw)
            X_all.append(Xi); y_all.append(yi)
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
    ):
        """
        1) Carga artefactos y genera imágenes per-frame (cada ventana = 1 imagen).
        2) Si PyTorch está disponible, entrena una TinyCNN que respeta tu feature_extractor + FC + Softmax.
        """
        try:
            Xtr, ytr = cls._prepare_images_and_labels(xTrain, yTrain, image_hw=instance.image_hw, k_ctx=instance.frame_context)
            Xte, yte = cls._prepare_images_and_labels(xTest, yTest, image_hw=instance.image_hw, k_ctx=instance.frame_context)
        except Exception as e:
            raise RuntimeError(f"Error preparando dataset per-frame: {e}") from e

        # Verifica coherencia del grafo (dim flatten)
        H_in, W_in = instance.image_hw
        _ = instance.flatten_size((H_in, W_in, 3))

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            class TinyCNN(nn.Module):
                def __init__(self, spec: CNN, in_shape=(3, 64, 128)):
                    super().__init__()
                    layers = []
                    # feature extractor
                    for block in spec.feature_extractor:
                        if isinstance(block, ConvolutionLayer):
                            kh, kw = block.kernel_shape()
                            conv = nn.Conv2d(
                                in_channels=3, out_channels=block.num_filters(),
                                kernel_size=(kh, kw),
                                stride=block.stride,
                                padding=0 if block.padding=="valid" else (kh//2, kw//2),
                                bias=True
                            )
                            layers.append(conv)
                            # activación
                            act = block.activation.kind
                            layers.append(
                                nn.ReLU(inplace=True) if act=="relu" else
                                nn.Tanh() if act=="tanh" else
                                nn.GELU() if act=="gelu" else
                                nn.Sigmoid() if act=="sigmoid" else nn.ReLU(inplace=True)
                            )
                        else:
                            ph, pw = block.pool_size
                            sh, sw = block.stride or block.pool_size
                            pad = 0 if block.padding=="valid" else (ph//2, pw//2)
                            pool = nn.MaxPool2d(kernel_size=(ph, pw), stride=(sh, sw), padding=pad) \
                                   if block.kind=="max" else \
                                   nn.AvgPool2d(kernel_size=(ph, pw), stride=(sh, sw), padding=pad)
                            layers.append(pool)

                    self.fe = nn.Sequential(*layers)
                    # calcular flatten
                    with torch.no_grad():
                        dummy = torch.zeros(1, 3, in_shape[1], in_shape[2])
                        z = self.fe(dummy)
                        flat_dim = int(np.prod(list(z.shape[1:])))
                    # FC
                    fc_list = []
                    in_units = flat_dim
                    for d in spec.fc_layers:
                        fc_list.append(nn.Linear(in_units, d.units))
                        act = spec.fc_activation_common.kind
                        if act == "relu":
                            fc_list.append(nn.ReLU(inplace=True))
                        elif act == "tanh":
                            fc_list.append(nn.Tanh())
                        elif act == "gelu":
                            fc_list.append(nn.GELU())
                        elif act == "sigmoid":
                            fc_list.append(nn.Sigmoid())
                        in_units = d.units
                    # salida
                    fc_list.append(nn.Linear(in_units, spec.classification.units))
                    self.head = nn.Sequential(*fc_list)

                def forward(self, x):
                    z = self.fe(x)
                    z = torch.flatten(z, 1)
                    return self.head(z)

            n_classes = int(max(int(ytr.max()), int(yte.max()))) + 1
            H_in, W_in = instance.image_hw
            net = TinyCNN(instance, in_shape=(3, H_in, W_in)).to(device)

            tr = TensorDataset(torch.tensor(np.transpose(Xtr, (0,3,1,2))), torch.tensor(ytr))
            te = TensorDataset(torch.tensor(np.transpose(Xte, (0,3,1,2))), torch.tensor(yte))
            dl_tr = DataLoader(tr, batch_size=64, shuffle=True)
            dl_te = DataLoader(te, batch_size=128, shuffle=False)

            optim_ = optim.Adam(net.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()

            train_losses: List[float] = []  # <- NUEVO

            net.train()
            for _ in range(2):  # pocas épocas para demo; ajusta a tu gusto
                epoch_loss = 0.0
                n_batches = 0
                for xb, yb in dl_tr:
                    xb, yb = xb.float().to(device), yb.long().to(device)
                    optim_.zero_grad()
                    logits = net(xb)
                    loss = crit(logits, yb)
                    loss.backward()
                    optim_.step()
                    epoch_loss += float(loss.item())
                    n_batches += 1
                train_losses.append(epoch_loss / max(1, n_batches))


            # --- Evaluación y métricas ---
            net.eval()
            all_logits = []
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for xb, yb in dl_te:
                    xb, yb = xb.float().to(device), yb.long().to(device)
                    logits = net(xb)
                    pred = torch.argmax(logits, dim=1)
                    all_logits.append(logits.cpu().numpy())
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(yb.cpu().numpy())

            import numpy as np
            y_pred = np.concatenate(all_preds, axis=0)
            y_true = np.concatenate(all_targets, axis=0)

            # Básicas
            acc  = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec  = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            f1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            cm   = confusion_matrix(y_true, y_pred).tolist()

            # AUC-ROC (multiclase One-vs-Rest con probabilidades softmax)
            try:
                logits = np.concatenate(all_logits, axis=0)
                # softmax estable
                exps = np.exp(logits - logits.max(axis=1, keepdims=True))
                proba = exps / (exps.sum(axis=1, keepdims=True) + 1e-12)
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
                loss=train_losses,                # curva de pérdida
                evaluation_time="",              
            )
            return metrics


        except Exception as e:
            print("[CNN] Entrenamiento real no ejecutado (fallback).")
            print(f"Razón: {e}")
            metrics = EvaluationMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                confusion_matrix=[],
                auc_roc=0.0,
                loss=[],                 # sin entrenamiento real
                evaluation_time="",      # opcional
            )
            return metrics


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

# Entrenamiento per-frame (X_i.npy con (F,Freqs,C) ó (n_times,n_channels); y_i.npy con (F,))
# acc = CNN.train(cnn, xTest=[...], yTest=[...], xTrain=[...], yTrain=[...])
"""
