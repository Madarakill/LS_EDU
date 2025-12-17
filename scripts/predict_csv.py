from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lead_scoring.artifacts import load_artifacts
from lead_scoring.predict import load_model, predict_dataframe


def main() -> int:
    # Проверки работы через консоль
    # Пример python scripts\predict_csv.py --input leads_1000.csv --output predictions_1000.csv --model model.pt --artifacts artifacts
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV with predictions")
    parser.add_argument("--model", default="model.pt", help="Path to model state_dict .pt")
    parser.add_argument("--artifacts", default="artifacts", help="Directory with saved preprocessing pipeline")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for label")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, low_memory=False)

    # Загружаем артефакты и модель
    artifacts = load_artifacts(args.artifacts)
    model = load_model(args.model, input_dim=len(artifacts.feature_names))

    # Считаем предсказания
    pred = predict_dataframe(df, model=model, artifacts=artifacts, threshold=args.threshold)

    # Добавляем колонки с результатом и сохраняем
    out = df.copy()
    out["pred_proba"] = pred.probabilities
    out["pred_label"] = pred.labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote: {output_path} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
