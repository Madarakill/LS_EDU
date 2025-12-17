from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
from flask import Flask, Response, render_template, request

from lead_scoring.artifacts import load_artifacts
from lead_scoring.predict import load_model, predict_dataframe
# Прототип АС "ИСПКП"
# Пользователь загружает CSV с "сырыми" колонками (как в исходном датасете), набор проходит через сохранённый пайплайн препроцессинга и нейронная сеть формирует предсказания.

def create_app() -> Flask:

    app = Flask(__name__)

    # Путь до папки с артефактами препроцессинга
    artifacts_dir = Path(os.environ.get("LS_ARTIFACTS_DIR", "artifacts"))

    # Путь до весов модели.
    model_path = Path(os.environ.get("LS_MODEL_PATH", "model.pt"))

    # Загрузка артефактов и модели при старте сервера
    artifacts = load_artifacts(artifacts_dir)
    model = load_model(model_path, input_dim=len(artifacts.feature_names))

    @app.get("/")
    def index():
        # Форма загрузки CSV
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        # Получаем файл из формы
        if "file" not in request.files:
            return render_template("index.html", error="Не найден файл в форме."), 400

        file = request.files["file"]
        if not file or not file.filename.lower().endswith(".csv"):
            return render_template("index.html", error="Нужен файл .csv."), 400

        # Порог вероятности для присвоения класса, по умолчанию 0.5
        threshold = float(request.form.get("threshold", "0.5"))

        # Читаем загруженный CSV в DataFrame.
        df = pd.read_csv(file, low_memory=False)

        # Инференс: препроцессинг -> модель -> вероятности/метки.
        result = predict_dataframe(df, model=model, artifacts=artifacts, threshold=threshold)
        out = df.copy()
        out["pred_proba"] = result.probabilities
        out["pred_label"] = result.labels

        # Формирование файла для скачивания с предсказаниями.
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        data = buf.getvalue().encode("utf-8-sig")
        return Response(
            data,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
