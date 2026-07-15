from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lead_scoring.artifacts import load_artifacts  # noqa: E402
from lead_scoring.predict import load_model, predict_dataframe  # noqa: E402


def main() -> int:
    # Проверки работы через консоль
    # Пример:
    # python scripts\predict_csv.py --input leads_1000.csv --output predictions_1000.csv \
    #   --model data/models/model.pt --artifacts data/artifacts
    data_dir = Path(os.environ.get("LS_DATA_DIR", REPO_ROOT / "data"))
    default_model = Path(os.environ.get("LS_MODEL_PATH", data_dir / "models" / "model.pt"))
    default_artifacts = Path(os.environ.get("LS_ARTIFACTS_DIR", data_dir / "artifacts"))
    parser = argparse.ArgumentParser(
        description="Пакетное предсказание по CSV.",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="Показать справку и выйти")
    parser.add_argument("--input", required=True, help="Путь к входному CSV")
    parser.add_argument("--output", required=True, help="Путь к выходному CSV с предсказаниями")
    parser.add_argument("--model", default=str(default_model), help="Путь к файлу модели (.pt)")
    parser.add_argument(
        "--artifacts",
        default=str(default_artifacts),
        help="Папка с артефактами препроцессинга",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Порог срабатывания (если не задан, берется из артефактов)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, low_memory=False)

    # Загружаем артефакты и модель
    artifacts = load_artifacts(args.artifacts)
    model = load_model(args.model, input_dim=len(artifacts.feature_names))

    # Считаем предсказания
    threshold = args.threshold
    if threshold is None:
        threshold = float(artifacts.config.get("threshold", 0.5))
    pred = predict_dataframe(df, model=model, artifacts=artifacts, threshold=threshold)

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
