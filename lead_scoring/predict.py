from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lead_scoring.artifacts import Artifacts, load_artifacts
from lead_scoring.model import LeadMLP
from lead_scoring.preprocess import preprocess_dataframe


@dataclass(frozen=True)
class PredictionResult:
    
    probabilities: np.ndarray
    labels: np.ndarray
    y_true: np.ndarray | None


def load_model(model_path: str | Path, input_dim: int | None = None, device: str | None = None) -> LeadMLP:
    
    #Загрузка модели
    chosen_device = torch.device(device or "cpu")
    state_dict = torch.load(str(model_path), map_location=chosen_device)

    expected_dim = None
    first_layer_weight = state_dict.get("net.0.weight")
    if first_layer_weight is not None and hasattr(first_layer_weight, "shape"):
        expected_dim = int(first_layer_weight.shape[1])

    if expected_dim is None:
        if input_dim is None:
            raise ValueError("Невозможно определить input_dim из state_dict;")
        expected_dim = int(input_dim)

    if input_dim is not None and int(input_dim) != expected_dim:
        raise ValueError(
            f"Модель ожидает input_dim={expected_dim}, но артефакты предоставляют input_dim={int(input_dim)}."
        )

    model = LeadMLP(expected_dim).to(chosen_device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_dataframe(
    df: pd.DataFrame,
    *,
    model: LeadMLP,
    artifacts: Artifacts | None = None,
    artifacts_dir: str | Path | None = None,
    threshold: float = 0.5,
) -> PredictionResult:
    # Предсказания по загруженному набору данных

    loaded = artifacts or load_artifacts(artifacts_dir)
    pp = preprocess_dataframe(
        df,
        ct=loaded.ct,
        scaler=loaded.scaler,
        config=loaded.config,
        feature_names=loaded.feature_names,
    )

    model_device = next(model.parameters()).device
    x_tensor = torch.from_numpy(pp.x).to(model_device)
    logits = model(x_tensor)
    probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    labels = (probs >= float(threshold)).astype(np.int64)
    return PredictionResult(probabilities=probs, labels=labels, y_true=pp.y)

