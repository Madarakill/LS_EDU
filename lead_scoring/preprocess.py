from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessResult:

    x: np.ndarray
    y: np.ndarray | None
    aligned_raw: pd.DataFrame


def _to_dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def clean_raw_dataframe(
    df: pd.DataFrame,
    *,
    id_cols: list[str],
    nan_cols: list[str],
    contact_events: set[str],
    select_cols: list[str],
    placeholders: list[str],
    map_cols: list[str],
    unknown_value: str = "Unknown",
) -> pd.DataFrame:
    # Общие шаги сырой очистки, одинаковые при сборке артефактов и при переобучении:
    # удаление id/шумных колонок, признак Recent_Contact, замена заглушек на NaN,
    # маппинг Yes/No, приведение категориальных колонок к нижнему регистру.
    df = df.copy()

    df = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in nan_cols if c in df.columns], errors="ignore")

    if contact_events and ("Last Activity" in df.columns or "Last Notable Activity" in df.columns):
        empty = pd.Series([None] * len(df), index=df.index, dtype="object")
        df["Recent_Contact"] = (
            df.get("Last Activity", empty).isin(contact_events)
            | df.get("Last Notable Activity", empty).isin(contact_events)
        ).astype("int64")

    for col in select_cols:
        if col in df.columns:
            df[col] = df[col].replace(placeholders, np.nan)

    for col in map_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(unknown_value).astype(str).str.strip().str.lower()

    return df


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    ct: Any,
    scaler: Any,
    config: dict[str, Any],
    feature_names: list[str],
) -> PreprocessResult:
    # Предобработка набора данных как в ноутбуке, игнорирование новых признаков отстутствующих на обучении
    # Создание отсутствующих колонок и фиксация порядка
    target_col = config["target_col"]
    raw_num_cols: list[str] = config["raw_num_cols"]
    raw_cat_cols: list[str] = config["raw_cat_cols"]
    medians: dict[str, float] = config["num_medians"]

    id_cols: list[str] = config["id_cols"]
    nan_cols: list[str] = config["nan_cols"]
    low_var_cols: list[str] = config["low_var_cols"]
    select_cols: list[str] = config["select_cols"]
    placeholders: list[str] = config["placeholders"]
    map_cols: list[str] = config["map_cols"]
    unknown_value: str = config.get("unknown_value", "Unknown")
    contact_events: set[str] = set(config.get("contact_events", []))

    df = df.copy()

    y: np.ndarray | None = None
    if target_col in df.columns:
        # Если целевая колонка присутствует, сохраняется отдельно
        y = pd.to_numeric(df[target_col], errors="coerce").astype(np.float32).to_numpy()
        df = df.drop(columns=[target_col])

    # Удаление идентификаторов и неинформативных столбцов
    df = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in nan_cols if c in df.columns], errors="ignore")

    # Добавление Recent_Contact, если есть нужные колонки (пропускаем, если уже посчитан
    # выше по пайплайну — иначе пересчёт по уже lowercase-строкам даст сплошные нули)
    if contact_events and "Recent_Contact" not in df.columns:
        last_activity = df.get("Last Activity")
        last_notable = df.get("Last Notable Activity")
        recent_contact = False
        if last_activity is not None:
            recent_contact = recent_contact | last_activity.isin(contact_events)
        if last_notable is not None:
            recent_contact = recent_contact | last_notable.isin(contact_events)
        if isinstance(recent_contact, pd.Series):
            df["Recent_Contact"] = recent_contact.fillna(False).astype("int64")

    # Замена заглушек (Select/Other/...) на NaN.
    for col in select_cols:
        if col in df.columns:
            df[col] = df[col].replace(placeholders, np.nan)

    # Маппинг Yes/No -> 1/0.
    for col in map_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # Приведение числовые поля к numeric, невалидные значения -> NaN
    for col in raw_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Заполняем пропусков
    for col, median_value in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_value)

    # Заполнение категориальных пропусков на Unknown.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(unknown_value)

    # Удаляем низковариативные колонки (где проктически все значения одинаковы, как было сделаено на обучении)
    df = df.drop(columns=[c for c in low_var_cols if c in df.columns], errors="ignore")

    # Выровнение схемы как в обучении (создание отсутствующих колонок и нормализовать формат)
    for col in raw_cat_cols:
        if col not in df.columns:
            df[col] = unknown_value
        else:
            df[col] = df[col].astype(str).fillna(unknown_value).str.strip().str.lower()

    for col in raw_num_cols:
        if col not in df.columns:
            df[col] = medians.get(col, 0.0)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(medians.get(col, 0.0))

    # Оставляем только колонки которые были в ColumnTransformer
    aligned_raw = df[raw_cat_cols + raw_num_cols]

    # Кодирование OneHotEncoder
    encoded = ct.transform(aligned_raw)
    x_dense = _to_dense(encoded)
    feat_df = pd.DataFrame(x_dense, columns=feature_names)

    # Скейлинг числовые признаки (как при обучении)
    num_feat_cols = [c for c in feature_names if c.startswith("num__")]
    if num_feat_cols:
        feat_df[num_feat_cols] = scaler.transform(feat_df[num_feat_cols])

    # Матрица признаков
    x = feat_df[feature_names].to_numpy(dtype=np.float32)
    return PreprocessResult(x=x, y=y, aligned_raw=aligned_raw)
