from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

import sys

# Подготовка для обучения модели через веб-интерфейс, в данный момент не используется
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lead_scoring.artifacts import load_artifacts
from lead_scoring.model import LeadMLP


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


def _to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def clean_like_training(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    target_col = config["target_col"]

    df = df.drop(columns=[c for c in config["id_cols"] if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in config["nan_cols"] if c in df.columns], errors="ignore")

    contact_events = set(config.get("contact_events", []))
    if contact_events and ("Last Activity" in df.columns or "Last Notable Activity" in df.columns):
        empty = pd.Series([None] * len(df), index=df.index, dtype="object")
        df["Recent_Contact"] = (
            df.get("Last Activity", empty).isin(contact_events)
            | df.get("Last Notable Activity", empty).isin(contact_events)
        ).astype("int64")

    for col in config["select_cols"]:
        if col in df.columns:
            df[col] = df[col].replace(config["placeholders"], np.nan)

    for col in config["map_cols"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    unknown_value = config.get("unknown_value", "Unknown")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(unknown_value)

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
    target_col = artifacts.config["target_col"]

    y = pd.to_numeric(df[target_col], errors="coerce").astype(np.float32).to_numpy()
    X = df.drop(columns=[target_col])

    raw_cat_cols = artifacts.config["raw_cat_cols"]
    raw_num_cols = artifacts.config["raw_num_cols"]
    medians = artifacts.config["num_medians"]
    unknown_value = artifacts.config.get("unknown_value", "Unknown")

    for col in raw_cat_cols:
        if col not in X.columns:
            X[col] = unknown_value
        X[col] = X[col].astype(str).fillna(unknown_value)

    for col in raw_num_cols:
        if col not in X.columns:
            X[col] = medians.get(col, 0.0)
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(medians.get(col, 0.0))

    X_aligned = X[raw_cat_cols + raw_num_cols]
    encoded = artifacts.ct.transform(X_aligned)
    x_dense = _to_dense(encoded)
    feat_df = pd.DataFrame(x_dense, columns=artifacts.feature_names)

    num_feat_cols = [c for c in artifacts.feature_names if c.startswith("num__")]
    if num_feat_cols:
        feat_df[num_feat_cols] = artifacts.scaler.transform(feat_df[num_feat_cols])

    x = feat_df.to_numpy(dtype=np.float32)
    return x, y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lead-scoring-dataset/Lead Scoring.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="model_1.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cpu")
    artifacts = load_artifacts(args.artifacts)

    df_raw = pd.read_csv(Path(args.dataset), low_memory=False)
    df = clean_like_training(df_raw, artifacts.config)
    spl = split(df, artifacts.config["target_col"])

    x_train, y_train = transform(spl.train, args.artifacts)
    x_val, y_val = transform(spl.val, args.artifacts)

    train_loader = DataLoader(NumpyDataset(x_train, y_train), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(NumpyDataset(x_val, y_val), batch_size=args.batch, shuffle=False)

    model = LeadMLP(input_layer=len(artifacts.feature_names)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    no_improve = 0
    out_path = Path(args.out)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
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
            n_train += len(yb)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item()) * len(yb)
                n_val += len(yb)

        train_loss /= max(n_train, 1)
        val_loss /= max(n_val, 1)
        print(f"epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve = 0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
            print(f"saved: {out_path}")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print("early stopping")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
