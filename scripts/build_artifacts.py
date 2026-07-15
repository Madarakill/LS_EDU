from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lead_scoring.preprocess import clean_raw_dataframe  # noqa: E402


def _to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def main() -> int:
    # Скрипт сборки артефактов препроцессинга (пайплайна обработки).
    #
    # Что делает:
    # - читает исходный датасет `Lead Scoring.csv`
    # - повторяет ключевые шаги очистки/преобразований (как сделано в ноутбуке с анализом)
    # - кодирование категорий OneHot и скейлинг числовых на train-части
    # - сохраняет всё в папку `data/artifacts/`, чтобы инференс работал стабильно
    #   на любых новых CSV
    #
    # На выходе создаются файлы:
    # - data/artifacts/ct.joblib: ColumnTransformer (OneHotEncoder + passthrough числовых)
    # - data/artifacts/scaler.joblib: StandardScaler для num__* признаков
    # - data/artifacts/feature_names.json: фиксированный порядок признаков на вход нейросети
    # - data/artifacts/config.json: параметры очистки/медианы/списки колонок.

    repo_root = REPO_ROOT
    dataset_path = repo_root / "lead-scoring-dataset" / "Lead Scoring.csv"
    data_dir = Path(os.environ.get("LS_DATA_DIR", repo_root / "data"))
    out_dir = Path(os.environ.get("LS_ARTIFACTS_DIR", data_dir / "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    target_col = "Converted"

    id_cols = ["Prospect ID", "Lead Number"]
    nan_cols = [
        "Lead Quality",
        "Asymmetrique Activity Index",
        "Asymmetrique Profile Index",
        "Asymmetrique Activity Score",
        "Asymmetrique Profile Score",
    ]

    select_cols = [
        "Specialization",
        "How did you hear about X Education",
        "What is your current occupation",
        "What matters most to you in choosing a course",
        "Lead Profile",
        "City",
        "Country",
        "Tags",
    ]
    placeholders = [
        "Select",
        "Select City",
        "Select Specialization",
        "Not specified",
        "Unknown",
        "Other",
    ]

    map_cols = [
        "Do Not Email",
        "Do Not Call",
        "Search",
        "Magazine",
        "Newspaper Article",
        "X Education Forums",
        "Newspaper",
        "Digital Advertisement",
        "Through Recommendations",
        "Receive More Updates About Our Courses",
        "Update me on Supply Chain Content",
        "Get updates on DM Content",
        "I agree to pay the amount through cheque",
        "A free copy of Mastering The Interview",
    ]

    df = pd.read_csv(dataset_path, low_memory=False)

    # Признак Recent_Contact
    contact_events = {
        "Had a Phone Conversation",
        "Visited Booth in Tradeshow",
        "Approached upfront",
        "Email Opened",
        "Email Received",
        "Email Link Clicked",
        "SMS Sent",
        "Form Submitted on Website",
        "Olark Chat Conversation",
        "View in browser link Clicked",
        "Page Visited on Website",
    }
    unknown_value = "Unknown"
    df = clean_raw_dataframe(
        df,
        id_cols=id_cols,
        nan_cols=nan_cols,
        contact_events=contact_events,
        select_cols=select_cols,
        placeholders=placeholders,
        map_cols=map_cols,
        unknown_value=unknown_value,
    )

    num_cols_all = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    num_cols_all = [c for c in num_cols_all if c != target_col]
    num_medians = {c: float(df[c].median()) for c in num_cols_all}
    for c in num_cols_all:
        df[c] = df[c].fillna(num_medians[c])

    df = df.drop_duplicates().reset_index(drop=True)

    exclude = {target_col}
    value_dominance = df.apply(lambda s: s.value_counts(normalize=True, dropna=False).max())
    low_var_cols = (
        value_dominance[(value_dominance >= 0.99) & ~value_dominance.index.isin(exclude)]
        .index.tolist()
    )
    df = df.drop(columns=low_var_cols, errors="ignore")

    if target_col not in df.columns:
        raise ValueError(f"Целевая переменная не найдена после очистки: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Определяем типы колонок на train
    raw_cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    raw_num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Собираем пайплайн кодирования признаков
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), raw_cat_cols),
            ("num", "passthrough", raw_num_cols),
        ]
    )

    X_train_enc = ct.fit_transform(X_train)
    X_val_enc = ct.transform(X_val)
    X_test_enc = ct.transform(X_test)

    feature_names = ct.get_feature_names_out().tolist()

    # Проверка признаков (должны совпадать с теми, на которых обучался `model.pt`)
    # (порядок из заголовка nn-dataset/leads_train_nn.csv)
    nn_train_path = repo_root / "nn-dataset" / "leads_train_nn.csv"
    if nn_train_path.exists():
        expected_cols = pd.read_csv(nn_train_path, nrows=0).columns.tolist()
        if target_col in expected_cols:
            expected_cols.remove(target_col)
        if feature_names != expected_cols:
            extra = sorted(set(feature_names) - set(expected_cols))
            missing = sorted(set(expected_cols) - set(feature_names))
            raise ValueError(
                "Несооствествие с nn-dataset/leads_train_nn.csv. "
                f"extra={extra[:10]} missing={missing[:10]}"
            )
        feature_names = expected_cols

    df_train_nn = pd.DataFrame(_to_dense(X_train_enc), columns=feature_names)
    df_val_nn = pd.DataFrame(_to_dense(X_val_enc), columns=feature_names)
    df_test_nn = pd.DataFrame(_to_dense(X_test_enc), columns=feature_names)

    # Скейлинг числовых признаков
    num_feat_cols = [c for c in feature_names if c.startswith("num__")]
    scaler = StandardScaler()
    if num_feat_cols:
        df_train_nn[num_feat_cols] = scaler.fit_transform(df_train_nn[num_feat_cols])
        df_val_nn[num_feat_cols] = scaler.transform(df_val_nn[num_feat_cols])
        df_test_nn[num_feat_cols] = scaler.transform(df_test_nn[num_feat_cols])

    # Сохранение предпроцессинга
    joblib.dump(ct, out_dir / "ct.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    (out_dir / "feature_names.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Порог классификации: существующий config.json > 0.5
    threshold = None
    existing = out_dir / "config.json"
    if existing.exists():
        try:
            existing_config = json.loads(existing.read_text(encoding="utf-8"))
            threshold = existing_config.get("threshold")
        except Exception:
            threshold = None
    if threshold is None:
        threshold = 0.5

    config = {
        "target_col": target_col,
        "unknown_value": unknown_value,
        "id_cols": id_cols,
        "nan_cols": nan_cols,
        "contact_events": sorted(contact_events),
        "select_cols": select_cols,
        "placeholders": placeholders,
        "map_cols": map_cols,
        "low_var_cols": low_var_cols,
        "raw_cat_cols": raw_cat_cols,
        "raw_num_cols": raw_num_cols,
        "num_medians": num_medians,
        "threshold": float(threshold),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Артефакты сохранены в: {out_dir}")
    print(
        f"Признаков: {len(feature_names)} | "
        f"Raw: {len(raw_cat_cols)} cat + {len(raw_num_cols)} num"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
