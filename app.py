from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory

from models import create_model
from preparation import FEATURE_NAMES, INPUT_COLUMNS, last_window, rows_to_features, validate_rows

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
app = Flask(__name__)

_frame: pd.DataFrame | None = None
_preprocess: dict | None = None
_catalog: dict | None = None
_models: dict[str, torch.nn.Module] = {}


def _require_artifacts() -> None:
    required = [
        ARTIFACTS / "clean_series.csv",
        ARTIFACTS / "preprocess.json",
        ARTIFACTS / "catalog.json",
        ARTIFACTS / "results.csv",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Training artifacts are missing (" + ", ".join(missing) + "). Run: python train.py"
        )


def load_artifacts() -> None:
    global _frame, _preprocess, _catalog, _models
    _require_artifacts()
    _frame = pd.read_csv(ARTIFACTS / "clean_series.csv")
    _preprocess = json.loads((ARTIFACTS / "preprocess.json").read_text(encoding="utf-8"))
    _catalog = json.loads((ARTIFACTS / "catalog.json").read_text(encoding="utf-8"))
    if _preprocess["feature_names"] != FEATURE_NAMES:
        raise RuntimeError("Training feature order does not match preparation.py.")
    n_features = len(FEATURE_NAMES)
    _models = {}
    for row in _catalog["models"]:
        model = create_model(row["key"], n_features)
        state = torch.load(ARTIFACTS / "models" / f"{row['id']}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _models[row["id"]] = model


def _public_row(row: pd.Series) -> dict:
    return {
        "timestamp": str(row["timestamp"]),
        "demanda_mw": round(float(row["demanda_mw"]), 4),
        "temperatura_c": round(float(row["temperatura_c"]), 4),
        "humedad_pct": round(float(row["humedad_pct"]), 4),
        "viento_kmh": round(float(row["viento_kmh"]), 4),
        "radiacion_wm2": round(float(row["radiacion_wm2"]), 4),
        "precipitacion_mm": round(float(row["precipitacion_mm"]), 4),
        "precio_kwh": round(float(row["precio_kwh"]), 4),
        "hora": int(row["hora"]),
        "dia_semana": int(row["dia_semana"]),
        "fin_semana": int(row["fin_semana"]),
        "festivo": int(row["festivo"]),
        "mes": int(row["mes"]),
        "periodo_dia": str(row["periodo_dia"]),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/context")
def context():
    cleaning = json.loads((ARTIFACTS / "cleaning.json").read_text(encoding="utf-8"))
    results = json.loads(pd.read_csv(ARTIFACTS / "results.csv").to_json(orient="records"))
    importance = []
    importance_path = ARTIFACTS / "variable_importance.csv"
    if importance_path.exists():
        importance = json.loads(pd.read_csv(importance_path).to_json(orient="records"))
    group_importance = []
    group_path = ARTIFACTS / "group_importance.csv"
    if group_path.exists():
        group_importance = json.loads(pd.read_csv(group_path).to_json(orient="records"))
    analysis = None
    analysis_path = ARTIFACTS / "analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    return jsonify(
        {
            "recommended": _catalog["recommended"],
            "models": _catalog["models"],
            "baseline": _catalog["baseline"],
            "split": _preprocess["split"],
            "cleaning": {
                "original_rows": cleaning["original_rows"],
                "clean_rows": cleaning["clean_rows"],
                "exact_duplicates_removed": cleaning["exact_duplicates_removed"],
                "rows_with_missing_measurements": cleaning["rows_with_missing_measurements"],
                "rows_with_any_anomaly": cleaning["rows_with_any_anomaly"],
                "rows_dropped_missing_or_anomaly": cleaning["rows_dropped_missing_or_anomaly"],
                "gaps_longer_than_one_hour": cleaning["gaps_longer_than_one_hour"],
                "period_inconsistent_with_hour": cleaning["period_inconsistent_with_hour"],
                "rows_by_anomaly_rule": cleaning["rows_by_anomaly_rule"],
                "threshold_reasons": cleaning["threshold_reasons"],
                "target_column_vs_next_hour_demand": cleaning["target_column_vs_next_hour_demand"],
            },
            "results": results,
            "importance": importance,
            "group_importance": group_importance,
            "analysis": analysis,
            "charts": [
                "loss_train_validation.png",
                "actual_vs_predicted.png",
                "prediction_error.png",
                "model_comparison.png",
                "temporal_attention.png",
            ],
        }
    )


@app.get("/api/window")
def api_window():
    window = request.args.get("window", type=int)
    if window not in (12, 24, 48):
        return jsonify({"error": "la ventana debe ser 12, 24 o 48."}), 400
    block = last_window(_frame, window)
    last = pd.to_datetime(block["timestamp"].iloc[-1])
    target_time = (last + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    observed = _frame.loc[_frame["timestamp"] == target_time, "demanda_mw"]
    return jsonify(
        {
            "rows": [_public_row(row) for _, row in block.iterrows()],
            "predict_for": target_time,
            "observed_mw": None if observed.empty else round(float(observed.iloc[0]), 2),
        }
    )


@app.post("/api/predict")
def api_predict():
    body = request.get_json(silent=True) or {}
    model_id = str(body.get("model", ""))
    rows = body.get("rows")
    if not isinstance(rows, list):
        return jsonify({"error": "envíe la lista de horas de la ventana."}), 400

    if model_id.startswith("persistence_w"):
        try:
            window = int(model_id.rsplit("_w", 1)[1])
        except ValueError:
            return jsonify({"error": "modelo no reconocido."}), 400
        message = validate_rows(rows, window)
        if message:
            return jsonify({"error": message}), 400
        prediction = float(rows[-1]["demanda_mw"])
        return jsonify({"prediction_mw": prediction, "model": model_id, "kind": "persistence"})

    if model_id not in _models:
        return jsonify({"error": "modelo no reconocido."}), 400
    window = int(model_id.rsplit("_w", 1)[1])
    message = validate_rows(rows, window)
    if message:
        return jsonify({"error": message}), 400

    try:
        matrix = rows_to_features(rows)
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    mean = np.asarray(_preprocess["x_mean"], dtype=np.float64)
    scale = np.asarray(_preprocess["x_scale"], dtype=np.float64)
    normalized = ((matrix - mean) / scale).astype(np.float32)
    with torch.no_grad():
        scaled = _models[model_id](torch.from_numpy(normalized).unsqueeze(0)).item()
    prediction = float(scaled * _preprocess["y_scale"] + _preprocess["y_mean"])
    return jsonify({"prediction_mw": prediction, "model": model_id, "kind": "lstm"})


@app.get("/charts/<path:name>")
def charts(name: str):
    return send_from_directory(ARTIFACTS / "charts", name)


@app.get("/health")
def health():
    return jsonify({"models": sorted(_models), "columns": INPUT_COLUMNS})


if __name__ == "__main__":
    load_artifacts()
    app.run(host="127.0.0.1", port=5050, debug=False)
