from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Исполняемый файл для обучения модели через веб-интерфейс
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lead_scoring.artifacts import load_artifacts
from lead_scoring.model import LeadMLP
from lead_scoring.preprocess import clean_raw_dataframe, preprocess_dataframe


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class NumpyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def prepare_raw_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    target_col = config["target_col"]

    df = clean_raw_dataframe(
        df,
        id_cols=config["id_cols"],
        nan_cols=config["nan_cols"],
        contact_events=set(config.get("contact_events", [])),
        select_cols=config["select_cols"],
        placeholders=config["placeholders"],
        map_cols=config["map_cols"],
        unknown_value=config.get("unknown_value", "Unknown"),
    )

    for col, median_value in config["num_medians"].items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median_value)

    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=[c for c in config["low_var_cols"] if c in df.columns], errors="ignore")

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    return df


def split(df: pd.DataFrame, target_col: str) -> Split:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train = X_train.copy()
    train[target_col] = y_train.values
    val = X_val.copy()
    val[target_col] = y_val.values
    test = X_test.copy()
    test[target_col] = y_test.values
    return Split(train=train, val=val, test=test)


def transform(df: pd.DataFrame, artifacts_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    artifacts = load_artifacts(artifacts_dir)
    result = preprocess_dataframe(
        df,
        ct=artifacts.ct,
        scaler=artifacts.scaler,
        config=artifacts.config,
        feature_names=artifacts.feature_names,
    )
    return result.x, result.y


def train_and_save(
    *,
    dataset_path: str | Path,
    artifacts_dir: str | Path,
    out_path: str | Path,
    epochs: int = 50,
    patience: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, object]:
    # Переобучение модели LeadMLP.
    device = torch.device("cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    artifacts = load_artifacts(artifacts_dir)

    df_raw = pd.read_csv(Path(dataset_path), low_memory=False)
    df = prepare_raw_dataframe(df_raw, artifacts.config)
    spl = split(df, artifacts.config["target_col"])

    x_train, y_train = transform(spl.train, artifacts_dir)
    x_val, y_val = transform(spl.val, artifacts_dir)

    train_loader = DataLoader(NumpyDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NumpyDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    model = LeadMLP(input_layer=len(artifacts.feature_names)).to(device)
    pos_weight_value = (len(y_train) - float(y_train.sum())) / max(float(y_train.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    best_state_dict = None
    out_path = Path(out_path)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(yb)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += int((preds == yb).sum().item())
            n_train += len(yb)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item()) * len(yb)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += int((preds == yb).sum().item())
                n_val += len(yb)

        train_loss /= max(n_train, 1)
        val_loss /= max(n_val, 1)
        train_acc = train_correct / max(n_train, 1)
        val_acc = val_correct / max(n_val, 1)
        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.6f} | train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.6f} | val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
        else:
            no_improve += 1

        if no_improve >= int(patience):
            print("early stopping")
            break

    # Подбор порога классификации по F1 на валидации
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(x_val).to(device)
        val_probs = torch.sigmoid(model(val_tensor)).cpu().numpy().ravel()
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y_val, (val_probs >= t).astype(int)) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(scores))])

    # Сохранение порога в config.json артефактов
    artifacts_dir = Path(artifacts_dir)
    config_path = artifacts_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["threshold"] = best_threshold
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "out_path": str(out_path),
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
    }


def main() -> int:
    data_dir = Path(os.environ.get("LS_DATA_DIR", REPO_ROOT / "data"))
    default_artifacts = Path(os.environ.get("LS_ARTIFACTS_DIR", data_dir / "artifacts"))
    default_out = Path(os.environ.get("LS_MODEL_PATH", data_dir / "models" / "model.pt"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lead-scoring-dataset/Lead Scoring.csv")
    parser.add_argument("--artifacts", default=str(default_artifacts))
    parser.add_argument("--out", default=str(default_out))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_and_save(
        dataset_path=args.dataset,
        artifacts_dir=args.artifacts,
        out_path=args.out,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed,
    )
    print(
        f"saved: {result['out_path']} | best_epoch={result['best_epoch']} "
        f"| best_val_loss={result['best_val_loss']:.6f} | best_val_acc={result['best_val_acc']:.4f} "
        f"| best_threshold={result['best_threshold']:.2f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
