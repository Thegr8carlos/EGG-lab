from __future__ import annotations
from typing import List, Optional, Literal, Sequence, Tuple
import os
import numpy as np
from pydantic import BaseModel, Field, field_validator

# Importa tu base
from backend.classes.ClasificationModel.ClsificationModels import Classifier
# (Opcional) from backend.classes.Metrics import EvaluationMetrics
import time

from backend.classes.Metrics import EvaluationMetrics

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
        False,
        description="Si True, permite que cada archivo tenga forma distinta (si es 'flatten', se rechaza por no poder apilar)."
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

    # ----------------------------- I/O Helpers -----------------------------
    @staticmethod
    def _load_labels_scalar(paths: Sequence[str]) -> NDArray:
        """
        Carga y valida etiquetas escalares (1 por archivo). Devuelve (N,) int64.
        """
        ys: List[int] = []
        for p in paths:
            if (not os.path.exists(p)) or (not p.endswith(".npy")):
                raise FileNotFoundError(f"Archivo de etiqueta inválido: {p}")
            y = np.load(p, allow_pickle=True)
            y = np.array(y).reshape(-1)
            if y.size != 1:
                raise ValueError(f"Etiqueta inválida en {p}: se esperaba escalar.")
            ys.append(int(y[0]))
        return np.array(ys, dtype=np.int64)

    @classmethod
    def _prepare_xy(
        cls,
        instance: "SVNN",
        x_paths: Sequence[str],
        y_paths: Sequence[str],
    ) -> Tuple[NDArray, NDArray]:
        """
        - X: cada .npy -> vector 1D según input_adapter.
        - y: un escalar por archivo (.npy).
        """
        if len(x_paths) != len(y_paths):
            raise ValueError("x_paths y y_paths deben tener la misma longitud.")
        X = instance.input_adapter.transform_batch(x_paths)  # (N,D)
        y = cls._load_labels_scalar(y_paths)                 # (N,)
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
    ):
        """
        Entrenamiento real con PyTorch si está disponible; de lo contrario, imprime simulación.
        - Infiero input_dim del primer batch de X.
        - Si las clases reales difieren de 'classification_units', ajusto a max(y)+1.
        """
        # 1) Datos
        Xtr, ytr = cls._prepare_xy(instance, xTrain, yTrain)
        Xte, yte = cls._prepare_xy(instance, xTest, yTest)

        in_dim = int(Xtr.shape[1])
        n_classes = int(max(int(ytr.max()), int(yte.max()))) + 1
        out_dim = max(n_classes, int(instance.classification_units))

        # 2) Construcción del modelo
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            class MLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    seq = []
                    d_in = in_dim
                    # Capas ocultas declarativas
                    for lyr in instance.layers:
                        seq.append(nn.Linear(d_in, lyr.units, bias=True))
                        if lyr.batchnorm:
                            seq.append(nn.BatchNorm1d(lyr.units))
                        # activación
                        a = lyr.activation.kind
                        if a == "relu":
                            seq.append(nn.ReLU(inplace=True))
                        elif a == "tanh":
                            seq.append(nn.Tanh())
                        elif a == "gelu":
                            seq.append(nn.GELU())
                        elif a == "sigmoid":
                            seq.append(nn.Sigmoid())
                        elif a == "linear":
                            pass
                        elif a == "softmax":
                            # no aplicar softmax en intermedia; lo añadiremos al final con CE loss
                            pass
                        if lyr.dropout > 0:
                            seq.append(nn.Dropout(p=float(lyr.dropout)))
                        d_in = lyr.units
                    # Capa de salida
                    seq.append(nn.Linear(d_in, out_dim, bias=True))
                    self.net = nn.Sequential(*seq)

                def forward(self, x):
                    return self.net(x)

            net = MLP().to(device)
            opt = optim.Adam(net.parameters(), lr=float(instance.learning_rate))
            crit = torch.nn.CrossEntropyLoss()

            # DataLoaders
            tr = TensorDataset(
                torch.tensor(Xtr, dtype=torch.float32),
                torch.tensor(ytr, dtype=torch.long),
            )
            te = TensorDataset(
                torch.tensor(Xte, dtype=torch.float32),
                torch.tensor(yte, dtype=torch.long),
            )
            dl_tr = DataLoader(tr, batch_size=int(instance.batch_size), shuffle=True)
            dl_te = DataLoader(te, batch_size=max(64, int(instance.batch_size)), shuffle=False)

            # 3) Entrenamiento
           # 3) Entrenamiento (registrar pérdida por época)
            train_losses: List[float] = []

            net.train()
            for _ in range(int(instance.epochs)):
                epoch_loss = 0.0
                n_batches = 0
                for xb, yb in dl_tr:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    logits = net(xb)
                    loss = crit(logits, yb)
                    loss.backward()
                    opt.step()
                    epoch_loss += float(loss.item())
                    n_batches += 1
                train_losses.append(epoch_loss / max(1, n_batches))


           # 4) Evaluación + métricas completas
            t0 = time.perf_counter()

            net.eval()
            all_logits = []
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for xb, yb in dl_te:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = net(xb)                         # (B, out_dim)
                    pred = torch.argmax(logits, dim=1)       # (B,)
                    all_logits.append(logits.cpu().numpy())
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(yb.cpu().numpy())

            eval_seconds = time.perf_counter() - t0

            import numpy as np
            y_pred = np.concatenate(all_preds, axis=0)
            y_true = np.concatenate(all_targets, axis=0)

            # Métricas básicas
            acc  = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec  = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            f1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            cm   = confusion_matrix(y_true, y_pred).tolist()

            # AUC-ROC (binario/multiclase usando softmax y OVR)
            try:
                logits = np.concatenate(all_logits, axis=0)               # (N, out_dim)
                exps   = np.exp(logits - logits.max(axis=1, keepdims=True))
                proba  = exps / (exps.sum(axis=1, keepdims=True) + 1e-12) # (N, out_dim)
                auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))
            except Exception:
                auc = 0.0

            metrics = EvaluationMetrics(
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1_score=f1,
                confusion_matrix=cm,
                auc_roc=auc,
                loss=train_losses,                                 # curva de pérdida por época
                evaluation_time=f"{eval_seconds:.4f}s",
            )

            print(f"[SVNN] Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")
            return metrics


        except Exception as e:
            print(f"[SVNN] PyTorch no disponible o error: {e}")

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
            return metrics


# ========= Ejemplo de uso =========
"""
model = SVNN(
    learning_rate=5e-4,
    epochs=80,
    batch_size=64,
    layers=[
        DenseLayer(units=256, activation=ActivationFunction(kind="relu"), dropout=0.3, batchnorm=True),
        DenseLayer(units=128, activation=ActivationFunction(kind="gelu"), dropout=0.2, batchnorm=False),
        DenseLayer(units=64,  activation=ActivationFunction(kind="tanh"), dropout=0.0, batchnorm=False),
    ],
    fc_activation_common=ActivationFunction(kind="relu"),   # se impone a las capas que no pidan algo especial
    classification_units=7,
    input_adapter=InputAdapter(reduce_3d="mean_time_flat", scale="standard", allow_mixed_dims=True),
)
# acc = SVNN.train(model, xTest=[...], yTest=[...], xTrain=[...], yTrain=[...])
"""