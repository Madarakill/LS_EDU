from __future__ import annotations

# Загрузка артефактов препроцессинга.
# Артефакты используются для обработки новых наборов данных чтобы они соответствовали
# схеме, на которой обучалась модель.
# - ct.joblib: ColumnTransformer (OneHotEncoder)
# - scaler.joblib: StandardScaler для числовых признаков (num__*)
# - feature_names.json: фиксированный порядок признаков на вход нейросети
# - config.json: параметры очистки/заполнения пропусков/списки колонок

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class Artifacts:
    ct: Any
    scaler: Any
    config: dict[str, Any]
    feature_names: list[str]


def artifacts_dir() -> Path:
    # Папка с артефактами (data/artifacts по умолчанию)
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = Path(os.environ.get("LS_DATA_DIR", repo_root / "data"))
    return Path(os.environ.get("LS_ARTIFACTS_DIR", data_dir / "artifacts"))


def load_artifacts(dir_path: str | Path | None = None) -> Artifacts:
    # Загружает артефакты из указанной директории (data/artifacts по умолчанию)
    # Возвращает объект Artifacts, который затем используется в preprocess.py и predict.py
    base = Path(dir_path) if dir_path is not None else artifacts_dir()
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    feature_names = json.loads((base / "feature_names.json").read_text(encoding="utf-8"))
    ct = joblib.load(base / "ct.joblib")
    scaler = joblib.load(base / "scaler.joblib")
    return Artifacts(ct=ct, scaler=scaler, config=config, feature_names=feature_names)
