from __future__ import annotations
from typing import List, Literal, Optional, Sequence, Tuple, Union
import math
import numpy as np
from pydantic import BaseModel, Field, field_validator
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
    num_layers: int = Field(1, ge=1, le=10, description="Pilas internas de GRU en esta capa lógica.")
    return_sequences: bool = Field(True, description="Si True, devolver secuencias (T,H); si False, solo el último paso.")

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
        # En RNN, el “flatten” es simplemente el vector tras pooling (o salida final).
        return self.feature_dim_after_encoder()

    # =================================================================================
    # Dataset helpers: esperamos clasificación por SECUENCIA (una etiqueta por archivo)
    # =================================================================================
    @staticmethod
    def _load_sequence(path: str) -> NDArray:
        """
        Espera .npy de forma (T, F) o (F, T). Devuelve (T, F) float32.
        """
        X = np.load(path, allow_pickle=True)
        if X.ndim != 2:
            raise ValueError(f"Secuencia inválida: {path} con ndim={X.ndim}; se esperaba 2D.")
        # normalizamos a (T,F)
        T, F = X.shape
        if T < F:  # suposición común cuando viene (F,T)
            X = X.T
        return X.astype(np.float32, copy=False)

    @staticmethod
    def _load_label_scalar(path: str) -> int:
        """
        Carga etiqueta escalar (int) desde .npy. Acepta (1,), (), o (n,) con n=1.
        """
        y = np.load(path, allow_pickle=True)
        y = np.array(y).reshape(-1)
        if y.size != 1:
            raise ValueError(f"Etiqueta inválida en {path}: se esperaba escalar; recibido shape={y.shape}")
        return int(y[0])

    @classmethod
    def _prepare_sequences_and_labels(
        cls,
        x_paths: Sequence[str],
        y_paths: Sequence[str],
        pad_value: float = 0.0,
    ) -> Tuple[List[NDArray], NDArray, NDArray]:
        """
        Devuelve:
          - sequences: lista de arrays (Ti, F) con longitudes variables
          - lengths: longitudes Ti (int64)
          - y: etiquetas (N,) int64
        """
        if len(x_paths) != len(y_paths):
            raise ValueError("x_paths y y_paths deben tener la misma longitud (clasificación por secuencia).")
        sequences: List[NDArray] = []
        lengths: List[int] = []
        labels: List[int] = []
        F_ref: Optional[int] = None
        for xp, yp in zip(x_paths, y_paths):
            X = cls._load_sequence(xp)     # (T, F)
            y = cls._load_label_scalar(yp) # escalar
            if F_ref is None:
                F_ref = int(X.shape[1])
            elif int(X.shape[1]) != F_ref:
                raise ValueError(f"Dimensión de características inconsistente: {xp} tiene F={X.shape[1]} vs F_ref={F_ref}")
            sequences.append(X)
            lengths.append(int(X.shape[0]))
            labels.append(int(y))
        return sequences, np.array(lengths, dtype=np.int64), np.array(labels, dtype=np.int64)

    # =================================================================================
    # Entrenamiento: PyTorch si disponible (con pack_padded_sequence), fallback si no
    # =================================================================================
    @classmethod
    def train(
        cls,
        instance: "GRUNet",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 1e-3,
    ):
        # 1) Prepara dataset
        try:
            seq_tr, len_tr, y_tr = cls._prepare_sequences_and_labels(xTrain, yTrain, pad_value=instance.encoder.pad_value)
            seq_te, len_te, y_te = cls._prepare_sequences_and_labels(xTest, yTest, pad_value=instance.encoder.pad_value)
        except Exception as e:
            raise RuntimeError(f"Error preparando dataset GRU: {e}") from e

        # 2) Checa compatibilidad de dims
        d_enc, is_seq = instance.encoder.infer_output_signature()
        in_F = instance.encoder.input_feature_dim
        if d_enc < 1 or in_F < 1:
            raise ValueError("Dimensionalidades inválidas en encoder.")
        n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
            from torch.utils.data import Dataset, DataLoader

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            class SeqDataset(Dataset):
                def __init__(self, seqs: List[NDArray], lengths: NDArray, labels: NDArray, pad_value: float):
                    self.seqs = [torch.tensor(s) for s in seqs]  # cada s: (T,F)
                    self.lengths = torch.tensor(lengths, dtype=torch.int64)
                    self.labels = torch.tensor(labels, dtype=torch.int64)
                    self.pad_value = float(pad_value)

                def __len__(self): return len(self.labels)

                def __getitem__(self, idx: int):
                    return self.seqs[idx], self.lengths[idx], self.labels[idx]

                def collate(self, batch):
                    seqs, lens, labels = zip(*batch)
                    # pad a (Tmax, B, F)
                    padded = pad_sequence(seqs, batch_first=False, padding_value=self.pad_value)  # (Tmax,B,F)
                    lengths = torch.stack(lens)
                    labels = torch.stack(labels)
                    return padded, lengths, labels

            ds_tr = SeqDataset(seq_tr, len_tr, y_tr, instance.encoder.pad_value)
            ds_te = SeqDataset(seq_te, len_te, y_te, instance.encoder.pad_value)
            dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=ds_tr.collate)
            dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=ds_te.collate)

            # ------- Construcción del modelo en PyTorch a partir de la especificación -------
            class TinyGRU(nn.Module):
                def __init__(self, spec: GRUNet):
                    super().__init__()
                    self.use_packed = bool(spec.encoder.use_packed_sequences)
                    self.pad_value = float(spec.encoder.pad_value)
                    # Construye pila GRU (una instancia pytorch por capa lógica)
                    self.rnns = nn.ModuleList()
                    in_f = int(spec.encoder.input_feature_dim)
                    for layer in spec.encoder.layers:
                        self.rnns.append(
                            nn.GRU(
                                input_size=in_f,
                                hidden_size=layer.hidden_size,
                                num_layers=layer.num_layers,
                                bidirectional=layer.bidirectional,
                                dropout=(layer.dropout if layer.num_layers > 1 else 0.0),
                                batch_first=False,
                            )
                        )
                        # salida de esta GRU
                        out_f = layer.hidden_size * (2 if layer.bidirectional else 1)
                        in_f = out_f
                        # si esta capa no devuelve secuencia, “colapsa” aquí
                        # pero para simplificar entrenamiento, devolvemos siempre secuencia
                        # y aplicamos pooling al final (más flexible).
                    self.out_feat = in_f

                    # Pooling
                    self.pool_kind = spec.pooling.kind
                    if self.pool_kind == "attn":
                        ah = int(spec.pooling.attn_hidden or 64)
                        self.attn = nn.Sequential(
                            nn.Linear(self.out_feat, ah),
                            nn.Tanh(),
                            nn.Linear(ah, 1)
                        )
                    # FC head
                    fcs = []
                    in_units = self.out_feat
                    for d in spec.fc_layers:
                        fcs.append(nn.Linear(in_units, d.units))
                        ak = spec.fc_activation_common.kind
                        if ak == "relu":   fcs.append(nn.ReLU(inplace=True))
                        elif ak == "tanh": fcs.append(nn.Tanh())
                        elif ak == "gelu": fcs.append(nn.GELU())
                        elif ak == "sigmoid": fcs.append(nn.Sigmoid())
                        in_units = d.units
                    fcs.append(nn.Linear(in_units, spec.classification.units))
                    self.head = nn.Sequential(*fcs)

                def forward(self, x_pad: torch.Tensor, lengths: torch.Tensor):
                    """
                    x_pad: (Tmax,B,F), lengths: (B,)
                    Salida: logits (B, n_classes)
                    """
                    z = x_pad
                    # pasar por GRUs
                    for gru in self.rnns:
                        if self.use_packed:
                            packed = pack_padded_sequence(z, lengths.cpu(), enforce_sorted=False)
                            packed_out, _ = gru(packed)         # (packed_T,B,D)
                            z, _ = pad_packed_sequence(packed_out)  # → (Tmax,B,D)
                        else:
                            z, _ = gru(z)  # (Tmax,B,D)
                    # Pooling temporal
                    if self.pool_kind == "last":
                        # tomar último válido para cada secuencia
                        B = z.shape[1]
                        idx = (lengths - 1).view(1, B, 1).expand(1, B, z.shape[2])
                        out = z.gather(dim=0, index=idx).squeeze(0)  # (B,D)
                    elif self.pool_kind == "mean":
                        mask = torch.arange(z.shape[0], device=z.device).unsqueeze(1) < lengths.unsqueeze(0)
                        z_masked = z * mask.unsqueeze(2)  # (T,B,D)
                        out = z_masked.sum(dim=0) / lengths.clamp_min(1).unsqueeze(1)  # (B,D)
                    elif self.pool_kind == "max":
                        mask = torch.arange(z.shape[0], device=z.device).unsqueeze(1) < lengths.unsqueeze(0)
                        z_masked = z.masked_fill(~mask.unsqueeze(2), float("-inf"))
                        out, _ = z_masked.max(dim=0)  # (B,D)
                    elif self.pool_kind == "attn":
                        # scores por tiempo, softmax en T con máscara
                        scores = self.attn(z)  # (T,B,1)
                        mask = torch.arange(z.shape[0], device=z.device).unsqueeze(1) < lengths.unsqueeze(0)
                        scores = scores.squeeze(-1)  # (T,B)
                        scores = scores.masked_fill(~mask, float("-inf"))
                        alpha = torch.softmax(scores, dim=0).unsqueeze(-1)  # (T,B,1)
                        out = (z * alpha).sum(dim=0)  # (B,D)
                    else:
                        out = z[-1]  # fallback
                    return self.head(out)

            net = TinyGRU(instance).to(device)
            opt = optim.Adam(net.parameters(), lr=lr)
            crit = nn.CrossEntropyLoss()

            # Entrenamiento
            train_losses: List[float] = []  # NUEVO: almacenamos pérdida promedio por época

            net.train()
            for _ in range(epochs):
                epoch_loss = 0.0
                n_batches = 0
                for x_pad, lengths, yb in dl_tr:
                    x_pad, lengths, yb = x_pad.float().to(device), lengths.to(device), yb.long().to(device)
                    opt.zero_grad()
                    logits = net(x_pad, lengths)
                    loss = crit(logits, yb)
                    loss.backward()
                    opt.step()
                    epoch_loss += float(loss.item())
                    n_batches += 1
                train_losses.append(epoch_loss / max(1, n_batches))


            # Evaluación
            # Evaluación + métricas completas
            t0 = time.perf_counter()

            net.eval()
            all_logits = []
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for x_pad, lengths, yb in dl_te:
                    x_pad, lengths, yb = x_pad.float().to(device), lengths.to(device), yb.long().to(device)
                    logits = net(x_pad, lengths)        # (B, n_classes)
                    pred = torch.argmax(logits, dim=1)  # (B,)
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

            # AUC-ROC (binario/multiclase con softmax OVR)
            try:
                logits = np.concatenate(all_logits, axis=0)             # (N, n_classes)
                # softmax estable
                exps = np.exp(logits - logits.max(axis=1, keepdims=True))
                proba = exps / (exps.sum(axis=1, keepdims=True) + 1e-12)
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
                loss=train_losses,                       # curva de pérdida por época
                evaluation_time=f"{eval_seconds:.4f}s",
            )

            print(f"[GRU] Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")
            return metrics


        except Exception as e:
            print("[GRU] Entrenamiento real no ejecutado (fallback).")
            print(f"Razón: {e}")
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
