from __future__ import annotations

import io
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
from flask import Flask, Response, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

import auth_config
from lead_scoring.artifacts import load_artifacts
from lead_scoring.predict import load_model, predict_dataframe


def create_app() -> Flask:
    # веб-приложение Flask
    app = Flask(__name__)
    app.secret_key = auth_config.SECRET_KEY

    # Основные пути (корень данных: data/)
    data_dir = Path(os.environ.get("LS_DATA_DIR", "data"))
    artifacts_dir = Path(os.environ.get("LS_ARTIFACTS_DIR", data_dir / "artifacts"))
    model_path = Path(os.environ.get("LS_MODEL_PATH", data_dir / "models" / "model.pt"))
    runs_dir = Path(os.environ.get("LS_RUNS_DIR", data_dir / "runs"))
    runs_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    def reload_pipeline() -> None:
        # Перезагрузка при смене модели
        artifacts = load_artifacts(artifacts_dir)
        model = load_model(model_path, input_dim=len(artifacts.feature_names))
        app.config["LS_ARTIFACTS"] = artifacts
        app.config["LS_MODEL"] = model

    reload_pipeline()

    def get_current_user():
        login = session.get("login")
        if not login:
            return None
        record = auth_config.USERS.get(login)
        if not record:
            return None
        return {"login": login, "role": record["role"]}

    def require_login():
        if not get_current_user():
            return redirect(url_for("login"))
        return None

    def require_role(*roles: str):
        user = get_current_user()
        if not user:
            return redirect(url_for("login"))
        if user["role"] not in roles:
            return "Недостаточно прав.", 403
        return None

    @app.get("/login")
    def login():
        if get_current_user():
            return redirect(url_for("index"))
        return render_template("login.html", error=None)

    @app.post("/login")
    def login_post():
        login = (request.form.get("login") or "").strip()
        password = request.form.get("password") or ""

        record = auth_config.USERS.get(login)
        if not record or not check_password_hash(record["password_hash"], password):
            return render_template("login.html", error="Неверный логин или пароль."), 401

        session["login"] = login
        return redirect(url_for("index"))

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    def index():
        gate = require_login()
        if gate is not None:
            return gate
        default_threshold = float(app.config["LS_ARTIFACTS"].config.get("threshold", 0.5))
        return render_template(
            "index.html",
            user=get_current_user(),
            error=None,
            top10=None,
            download_url=None,
            threshold=default_threshold,
        )

    @app.post("/predict")
    def predict():
        gate = require_login()
        if gate is not None:
            return gate

        user = get_current_user()

        if "file" not in request.files:
            return (
                render_template(
                    "index.html",
                    user=user,
                    error="Не найден файл в форме (поле file).",
                    top10=None,
                    download_url=None,
                    threshold=default_threshold,
                ),
                400,
            )

        file = request.files["file"]
        if not file or not file.filename.lower().endswith(".csv"):
            return (
                render_template(
                    "index.html",
                    user=user,
                    error="Нужен файл .csv.",
                    top10=None,
                    download_url=None,
                    threshold=default_threshold,
                ),
                400,
            )

        # user: порог по умолчанию из артефактов (если есть)
        # operator/admin: можно менять порог
        default_threshold = float(app.config["LS_ARTIFACTS"].config.get("threshold", 0.5))
        threshold = default_threshold
        if user and user["role"] in {"operator", "admin"}:
            try:
                threshold = float(request.form.get("threshold", str(default_threshold)))
            except ValueError:
                return (
                    render_template(
                        "index.html",
                        user=user,
                        error="Некорректный порог. Укажите число (например 0.5).",
                        top10=None,
                        download_url=None,
                        threshold=default_threshold,
                    ),
                    400,
                )

        if file.filename is None or file.filename.strip() == "":
            return (
                render_template(
                    "index.html",
                    user=user,
                    error="Имя файла пустое.",
                    top10=None,
                    download_url=None,
                    threshold=default_threshold,
                ),
                400,
            )

        try:
            df = pd.read_csv(file, low_memory=False)
        except Exception:
            return (
                render_template(
                    "index.html",
                    user=user,
                    error="Не удалось прочитать CSV. Проверьте формат, разделители и кодировку.",
                    top10=None,
                    download_url=None,
                    threshold=default_threshold,
                ),
                400,
            )

        model = app.config["LS_MODEL"]
        artifacts = app.config["LS_ARTIFACTS"]
        try:
            result = predict_dataframe(df, model=model, artifacts=artifacts, threshold=threshold)
        except Exception:
            return (
                render_template(
                    "index.html",
                    user=user,
                    error="Не удалось выполнить предсказание. Проверьте структуру данных.",
                    top10=None,
                    download_url=None,
                    threshold=default_threshold,
                ),
                400,
            )

        out = df.copy()
        out["pred_proba"] = result.probabilities
        out["pred_label"] = result.labels

        # Визуализация списка наиболее заинтересованных абитуриентов
        # ID абитуриента взято из колонок, если отсутсвует, то по номеру строки
        id_col = None
        for candidate in ("Prospect ID", "Lead Number"):
            if candidate in out.columns:
                id_col = candidate
                break
        if id_col is None:
            out["_row_id"] = (out.index + 1).astype(int)
            id_col = "_row_id"

        top10_df = out[[id_col, "pred_proba"]].sort_values("pred_proba", ascending=False).head(10)
        top10 = [
            {"id": str(row[id_col]), "pred_proba": float(row["pred_proba"])}
            for _, row in top10_df.iterrows()
        ]

        # Сохранение результатов классификации в дикеркторию /runs
        run_id = uuid4().hex
        run_path = runs_dir / f"{run_id}.csv"
        out.to_csv(run_path, index=False, encoding="utf-8-sig")
        session["last_run_id"] = run_id

        return render_template(
            "index.html",
            user=user,
            error=None,
            top10=top10,
            download_url=url_for("download", run_id=run_id),
            threshold=default_threshold,
        )

    @app.get("/download/<run_id>")
    def download(run_id: str):
        gate = require_login()
        if gate is not None:
            return gate

        # Загрузка последнего сохраненного результата выполненного из под одного пользователя
        # Пример: http://127.0.0.1:5000/download/349ffde91b984ad589cd9075bbf1628c
        if session.get("last_run_id") != run_id:
            return "Недостаточно прав.", 403

        path = runs_dir / f"{run_id}.csv"
        if not path.exists():
            return "Файл не найден.", 404

        data = path.read_bytes()
        return Response(
            data,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    @app.get("/admin")
    def admin_index():
        gate = require_role("admin")
        if gate is not None:
            return gate
        return render_template("admin.html", user=get_current_user(), message=None, error=None)

    @app.post("/admin/rebuild")
    def admin_rebuild():
        gate = require_role("admin")
        if gate is not None:
            return gate
        try:
            from scripts.build_artifacts import main as build_artifacts_main

            build_artifacts_main()
            reload_pipeline()
            return render_template(
                "admin.html",
                user=get_current_user(),
                message="Артефакты пересобраны.",
                error=None,
            )
        except Exception as e:
            return render_template("admin.html", user=get_current_user(), message=None, error=str(e)), 500

    @app.post("/admin/model")
    def admin_upload_model():
        gate = require_role("admin")
        if gate is not None:
            return gate

        if "file" not in request.files:
            return render_template("admin.html", user=get_current_user(), message=None, error="Нет файла."), 400

        file = request.files["file"]
        if not file or not file.filename.lower().endswith(".pt"):
            return render_template("admin.html", user=get_current_user(), message=None, error="Нужен файл .pt."), 400

        filename = secure_filename(file.filename) or "model.pt"
        file.save(model_path)

        try:
            reload_pipeline()
        except Exception as e:
            return render_template(
                "admin.html",
                user=get_current_user(),
                message=None,
                error=f"Модель сохранена ({filename}), но не загрузилась: {e}",
            ), 500

        return render_template(
            "admin.html",
            user=get_current_user(),
            message=f"Модель обновлена ({filename}).",
            error=None,
        )

    @app.post("/admin/train")
    def admin_train():
        gate = require_role("admin")
        if gate is not None:
            return gate

        dataset = request.form.get("dataset") or "lead-scoring-dataset/Lead Scoring.csv"
        epochs = int(request.form.get("epochs") or "5")
        patience = int(request.form.get("patience") or "3")
        batch = int(request.form.get("batch") or "256")
        lr = float(request.form.get("lr") or "0.001")
        seed = int(request.form.get("seed") or "42")
        overwrite = (request.form.get("overwrite") or "") == "on"

        out_path = model_path if overwrite else model_path.with_name("model_1.pt")

        try:
            from scripts.train_model import train_and_save

            result = train_and_save(
                dataset_path=dataset,
                artifacts_dir=str(artifacts_dir),
                out_path=str(out_path),
                epochs=epochs,
                patience=patience,
                batch_size=batch,
                lr=lr,
                seed=seed,
            )

            # Перезапуск при замене основной модели model.pt после обучения
            if overwrite:
                reload_pipeline()

            return render_template(
                "admin.html",
                user=get_current_user(),
                message=(
                    f"Обучение завершено. saved={result['out_path']}, "
                    f"best_epoch={result['best_epoch']}, "
                    f"best_val_loss={result['best_val_loss']:.6f}, best_val_acc={result['best_val_acc']:.4f}"
                ),
                error=None,
            )
        except Exception as e:
            return render_template("admin.html", user=get_current_user(), message=None, error=str(e)), 500

    return app

# Запуск приложения на всех интерфейсах на порту 5000, для отладки включен debug
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
