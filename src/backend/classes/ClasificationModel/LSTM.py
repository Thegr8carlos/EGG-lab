from __future__ import annotations
from typing import List, Literal, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, Field, field_validator

# ===== Tipos =====
NDArray = np.ndarray
ActName = Literal["relu", "tanh", "sigmoid", "gelu", "softmax", "linear"]
PoolName = Literal["last", "mean", "max", "attn"]
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
    # num_layers internos de PyTorch (apila internamente):
    num_layers: int = Field(1, ge=1, le=10, description="Pilas internas de LSTM en esta capa.")
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout entre capas internas (solo si num_layers>1).")
    return_sequences: bool = Field(True, description="Si True, devolver secuencia (T,D); si False, solo el último paso.")
    # Opcionales “expert”: proyección (si tu torch lo soporta), bias, batch_first
    use_bias: bool = Field(True, description="Usar términos bias en LSTM.")
    batch_first: bool = Field(False, description="Esperar (T,B,F)=False o (B,T,F)=True en PyTorch (interno).")
    proj_size: int = Field(0, ge=0, description="Tamaño de proyección (0=sin proyección). Requiere PyTorch reciente.")

    def output_dim(self, input_dim: int) -> Tuple[int, bool]:
        """
        Calcula (feature_dim_salida, es_secuencia?)
        D = (proj_size si >0, si no hidden_size) * (2 si bidi else 1)
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

    # =================================================================================
    # Dataset helpers: clasificación por SECUENCIA (una etiqueta por archivo)
    # =================================================================================
    @staticmethod
    def _load_sequence(path: str) -> NDArray:
        """
        Espera .npy (T,F) o (F,T). Devuelve (T,F) float32.
        """
        X = np.load(path, allow_pickle=True)
        if X.ndim != 2:
            raise ValueError(f"Secuencia inválida en {path}: se esperaba 2D.")
        T, F = X.shape
        if T < F:  # si viene (F,T), trasponemos para (T,F)
            X = X.T
        return X.astype(np.float32, copy=False)

    @staticmethod
    def _load_label_scalar(path: str) -> int:
        y = np.load(path, allow_pickle=True)
        y = np.array(y).reshape(-1)
        if y.size != 1:
            raise ValueError(f"Etiqueta inválida en {path}: se esperaba escalar.")
        return int(y[0])

    @classmethod
    def _prepare_sequences_and_labels(
        cls,
        x_paths: Sequence[str],
        y_paths: Sequence[str],
        pad_value: float = 0.0,
    ):
        """
        Devuelve:
          - sequences: lista de arrays (Ti, F)
          - lengths: (N,) int64
          - y: (N,) int64
        """
        if len(x_paths) != len(y_paths):
            raise ValueError("x_paths y y_paths deben tener la misma longitud.")
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
                raise ValueError(f"Dim de características inconsistente: {xp} F={X.shape[1]} vs F_ref={F_ref}")
            sequences.append(X)
            lengths.append(int(X.shape[0]))
            labels.append(int(y))
        import numpy as _np
        return sequences, _np.array(lengths, dtype=_np.int64), _np.array(labels, dtype=_np.int64)

    # =================================================================================
    # Entrenamiento: PyTorch (si está) con packed sequences y pooling; fallback si no
    # =================================================================================
    @classmethod
    def train(
        cls,
        instance: "LSTMNet",
        xTest: List[str],
        yTest: List[str],
        xTrain: List[str],
        yTrain: List[str],
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 1e-3,
    ):
        # 1) Dataset
        try:
            seq_tr, len_tr, y_tr = cls._prepare_sequences_and_labels(xTrain, yTrain, pad_value=instance.encoder.pad_value)
            seq_te, len_te, y_te = cls._prepare_sequences_and_labels(xTest, yTest, pad_value=instance.encoder.pad_value)
        except Exception as e:
            raise RuntimeError(f"Error preparando dataset LSTM: {e}") from e

        # 2) Chequeo básico
        d_enc, _ = instance.encoder.infer_output_signature()
        in_F = instance.encoder.input_feature_dim
        n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1
        if d_enc < 1 or in_F < 1 or n_classes < 2:
            raise ValueError("Dimensionalidades/num clases inválidas.")

        # 3) Entrenamiento real si hay torch
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
            from torch.utils.data import Dataset, DataLoader

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            class SeqDataset(Dataset):
                def __init__(self, seqs, lengths, labels, pad_value: float):
                    self.seqs = [torch.tensor(s) for s in seqs]  # (T,F)
                    self.lengths = torch.tensor(lengths, dtype=torch.int64)
                    self.labels = torch.tensor(labels, dtype=torch.int64)
                    self.pad_value = float(pad_value)

                def __len__(self): return len(self.labels)
                def __getitem__(self, idx: int):
                    return self.seqs[idx], self.lengths[idx], self.labels[idx]
                def collate(self, batch):
                    seqs, lens, labels = zip(*batch)
                    padded = pad_sequence(seqs, batch_first=False, padding_value=self.pad_value)  # (Tmax,B,F)
                    lengths = torch.stack(lens)
                    labels = torch.stack(labels)
                    return padded, lengths, labels

            ds_tr = SeqDataset(seq_tr, len_tr, y_tr, instance.encoder.pad_value)
            ds_te = SeqDataset(seq_te, len_te, y_te, instance.encoder.pad_value)
            dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  collate_fn=ds_tr.collate)
            dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=ds_te.collate)

            # -------- Construcción del modelo PyTorch a partir de la especificación --------
            class TinyLSTM(nn.Module):
                def __init__(self, spec: LSTMNet):
                    super().__init__()
                    self.use_packed = bool(spec.encoder.use_packed_sequences)
                    self.pad_value = float(spec.encoder.pad_value)
                    self.enforce_sorted = bool(spec.encoder.enforce_sorted)

                    # Pila de LSTMs
                    self.lstms = nn.ModuleList()
                    in_f = int(spec.encoder.input_feature_dim)
                    for layer in spec.encoder.layers:
                        # kwargs dependientes de versión (proj_size puede no estar):
                        kwargs = dict(
                            input_size=in_f,
                            hidden_size=layer.hidden_size,
                            num_layers=layer.num_layers,
                            bidirectional=layer.bidirectional,
                            dropout=(layer.dropout if layer.num_layers > 1 else 0.0),
                            bias=layer.use_bias,
                            batch_first=layer.batch_first,  # internamente usaremos (T,B,F), pero soporta ambos
                        )
                        # proj_size si está disponible
                        if layer.proj_size and layer.proj_size > 0 and "proj_size" in nn.LSTM.__init__.__code__.co_varnames:
                            kwargs["proj_size"] = layer.proj_size
                        self.lstms.append(nn.LSTM(**kwargs))
                        base = layer.proj_size if layer.proj_size > 0 else layer.hidden_size
                        out_f = base * (2 if layer.bidirectional else 1)
                        in_f = out_f

                    self.out_feat = in_f

                    # Pooling temporal
                    self.pool_kind = spec.pooling.kind
                    if self.pool_kind == "attn":
                        ah = int(spec.pooling.attn_hidden or 64)
                        self.attn = nn.Sequential(
                            nn.Linear(self.out_feat, ah), nn.Tanh(), nn.Linear(ah, 1)
                        )

                    # FC
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
                    x_pad: (Tmax,B,F)
                    """
                    z = x_pad
                    for lstm in self.lstms:
                        if self.use_packed:
                            packed = pack_padded_sequence(z, lengths.cpu(), enforce_sorted=self.enforce_sorted)
                            packed_out, _ = lstm(packed)
                            z, _ = pad_packed_sequence(packed_out)  # (Tmax,B,D)
                        else:
                            z, _ = lstm(z)  # (Tmax,B,D)

                    # Pooling temporal
                    if self.pool_kind == "last":
                        B = z.shape[1]
                        idx = (lengths - 1).view(1, B, 1).expand(1, B, z.shape[2])
                        out = z.gather(dim=0, index=idx).squeeze(0)  # (B,D)
                    elif self.pool_kind == "mean":
                        mask = torch.arange(z.shape[0], device=z.device).unsqueeze(1) < lengths.unsqueeze(0)
                        z_masked = z * mask.unsqueeze(2)
                        out = z_masked.sum(dim=0) / lengths.clamp_min(1).unsqueeze(1)
                    elif self.pool_kind == "max":
                        mask = torch.arange(z.shape[0], device=z.device).unsqueeze(1) < lengths.unsqueeze(0)
                        z_masked = z.masked_fill(~mask.unsqueeze(2), float("-inf"))
                        out, _ = z_masked.max(dim=0)
                    elif self.pool_kind == "attn":
                        scores = self.attn(z).squeeze(-1)              # (T,B)
                        mask = torch.arange(z.shape[0], device=z.device).unsqueeze(1) < lengths.unsqueeze(0)
                        scores = scores.masked_fill(~mask, float("-inf"))
                        alpha = torch.softmax(scores, dim=0).unsqueeze(-1)  # (T,B,1)
                        out = (z * alpha).sum(dim=0)  # (B,D)
                    else:
                        out = z[-1]
                    return self.head(out)

            net = TinyLSTM(instance).to(device)
            opt = optim.Adam(net.parameters(), lr=lr)
            crit = nn.CrossEntropyLoss()

            # Entrenamiento ligero (demo)
            # Entrenamiento (registrando pérdida por época)
            train_losses: List[float] = []

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


            # Evaluación + métricas completas
            t0 = time.perf_counter()

            net.eval()
            all_logits = []
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for x_pad, lengths, yb in dl_te:
                    x_pad, lengths, yb = x_pad.float().to(device), lengths.to(device), yb.long().to(device)
                    logits = net(x_pad, lengths)                  # (B, n_classes)
                    pred = torch.argmax(logits, dim=1)            # (B,)
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

            # AUC-ROC (binario o multiclase con softmax OVR)
            try:
                logits = np.concatenate(all_logits, axis=0)               # (N, n_classes)
                exps = np.exp(logits - logits.max(axis=1, keepdims=True)) # softmax estable
                proba = exps / (exps.sum(axis=1, keepdims=True) + 1e-12)  # (N, n_classes)
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
                loss=train_losses,                         # curva de pérdida por época
                evaluation_time=f"{eval_seconds:.4f}s",
            )

            print(f"[LSTM] Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")
            return metrics


        except Exception as e:
            print("[LSTM] Entrenamiento real no ejecutado (fallback).")
            print(f"Razón: {e}")
            return None



## =====================================================================================
"""
enc = SequenceEncoder(
    input_feature_dim=64,  # p.ej., 64 features por frame
    layers=[
        LSTMLayer(hidden_size=128, bidirectional=True, num_layers=1, dropout=0.0, return_sequences=True),
        LSTMLayer(hidden_size=128, bidirectional=True, num_layers=1, dropout=0.0, return_sequences=True),
        # Puedes cerrar en vector: return_sequences=False en la última capa
    ],
    use_packed_sequences=True,
    pad_value=0.0,
    enforce_sorted=False,
)

net = LSTMNet(
    encoder=enc,
    pooling=TemporalPooling(kind="attn", attn_hidden=64),      # 'last'|'mean'|'max'|'attn'
    fc_layers=[DenseLayer(units=128), DenseLayer(units=64)],
    fc_activation_common=ActivationFunction(kind="relu"),
    classification=DenseLayer(units=5, activation=ActivationFunction(kind="softmax")),
)

print("Dim vector tras encoder+pooling:", net.flatten_dim())
# acc = LSTMNet.train(net, xTest=[...], yTest=[...], xTrain=[...], yTrain=[...], epochs=5, batch_size=64)
"""

# Alias for backward compatibility
LSTM = LSTMNet