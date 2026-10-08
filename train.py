from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import ARCHITECTURES, count_parameters, create_model
from preparation import (
    FEATURE_NAMES,
    VARIABLE_GROUPS,
    DATASET_PATH,
    build_features,
    chronological_split,
    clean_dataset,
    create_sequences,
    save_json,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CHARTS = ARTIFACTS / "charts"
MODELS = ARTIFACTS / "models"

WINDOWS = (12, 24, 48)
MAX_EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SEED = 42

COLORS = {
    "lstm_base": "#1f4e79",
    "lstm_deep": "#b85c38",
    "lstm_attention": "#2f6f4e",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(error) / np.clip(np.abs(actual), 1e-6, None)) * 100)
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}


def predict(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    batch = 1024
    with torch.no_grad():
        for start in range(0, len(features), batch):
            tensor = torch.from_numpy(features[start : start + batch]).to(device)
            outputs.append(model(tensor).cpu().numpy())
    return np.concatenate(outputs)


def eval_mse(model: nn.Module, features: np.ndarray, target: np.ndarray, device: torch.device) -> float:
    predicted = predict(model, features, device)
    return float(np.mean((predicted - target) ** 2))


def train_one_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    best_value = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    history = {"train": [], "val": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            error = loss_fn(model(xb), yb)
            error.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_mse = eval_mse(model, x_train, y_train, device)
        val_mse = eval_mse(model, x_val, y_val, device)
        history["train"].append(train_mse)
        history["val"].append(val_mse)
        print(
            f"    epoch {epoch:02d}  train MSE {train_mse:.5f}  val MSE {val_mse:.5f}  "
            f"({time.perf_counter() - started:.1f} s)",
            flush=True,
        )
        if val_mse < best_value - 1e-5:
            best_value = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    early stopping at epoch {epoch}. Restoring epoch {best_epoch}.", flush=True)
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["epochs_run"] = len(history["train"])
    return model, history


def persistence_metrics(raw_windows: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    last_demand = raw_windows[:, -1, 0]
    return compute_metrics(actual, last_demand)


def choose_recommended(rows: list[dict]) -> str:
    candidates = [row for row in rows if row["key"] != "persistence"]
    best_rmse = min(row["RMSE"] for row in candidates)
    close = [row for row in candidates if row["RMSE"] <= best_rmse + 0.5]
    close.sort(key=lambda row: (row["parameters"], row["Window"], row["RMSE"]))
    return close[0]["id"]


def permutation_importance(
    model: nn.Module,
    features: np.ndarray,
    actual_mw: np.ndarray,
    y_mean: float,
    y_scale: float,
    device: torch.device,
    repeats: int = 4,
    groups: list[tuple[str, list[str]]] | None = None,
) -> list[dict]:
    def to_mw(scaled: np.ndarray) -> np.ndarray:
        return scaled * y_scale + y_mean

    baseline = compute_metrics(actual_mw, to_mw(predict(model, features, device)))["MAE"]
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    rows = []
    rng = np.random.default_rng(SEED)
    for variable, columns in (groups or VARIABLE_GROUPS):
        cols = [index[name] for name in columns]
        increases = []
        for _ in range(repeats):
            shuffled = features.copy()
            order = rng.permutation(len(features))
            shuffled[:, :, cols] = features[order][:, :, cols]
            mae = compute_metrics(actual_mw, to_mw(predict(model, shuffled, device)))["MAE"]
            increases.append(mae - baseline)
        rows.append(
            {
                "variable": variable,
                "mae_increase_mw": float(np.mean(increases)),
                "std_mw": float(np.std(increases)),
            }
        )
    rows.sort(key=lambda row: row["mae_increase_mw"], reverse=True)
    return rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def plot_losses(histories: dict, path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 8.6), sharex=False, constrained_layout=True)
    for i, key in enumerate(ARCHITECTURES):
        for j, window in enumerate(WINDOWS):
            ax = axes[i, j]
            history = histories[f"{key}_w{window}"]
            epochs = np.arange(1, len(history["train"]) + 1)
            ax.plot(epochs, history["train"], color=COLORS[key], label="entrenamiento")
            ax.plot(epochs, history["val"], color=COLORS[key], linestyle="--", label="validación")
            ax.axvline(history["best_epoch"], color="#888888", linewidth=0.8, linestyle=":")
            ax.set_title(f"{ARCHITECTURES[key].name} · {window} h")
            ax.set_xlabel("época")
            ax.set_ylabel("mse (demanda normalizada)")
            if i == 0 and j == 0:
                ax.legend(frameon=False)
    fig.suptitle("pérdida de entrenamiento y validación, evaluada sin dropout", fontsize=13)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_actual_vs_predicted(
    stamps: np.ndarray, actual: np.ndarray, predicted: np.ndarray, title: str, path: Path
) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(12.2, 7.4), constrained_layout=True, gridspec_kw={"height_ratios": [1.3, 1]}
    )
    shown = min(24 * 14, len(actual))
    axis = pd.to_datetime(stamps[-shown:])
    axes[0].plot(axis, actual[-shown:], color="#1c1915", linewidth=1.1, label="demanda real")
    axes[0].plot(axis, predicted[-shown:], color="#c4551a", linewidth=1.1, label="demanda predicha")
    axes[0].set_title(f"demanda real y predicha en prueba · {title} · últimos {shown // 24} días")
    axes[0].set_ylabel("demanda (mw)")
    axes[0].set_xlabel("fecha")
    axes[0].legend(frameon=False)

    axes[1].scatter(actual, predicted, s=8, alpha=0.35, color="#1f4e79", linewidths=0)
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    axes[1].plot([low, high], [low, high], color="#c4551a", linewidth=1)
    axes[1].set_title("demanda real frente a predicha en todo el conjunto de prueba")
    axes[1].set_xlabel("demanda real (mw)")
    axes[1].set_ylabel("demanda predicha (mw)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_error(stamps: np.ndarray, actual: np.ndarray, predicted: np.ndarray, title: str, path: Path) -> None:
    error = predicted - actual
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), constrained_layout=True)
    shown = min(24 * 14, len(error))
    axes[0].plot(pd.to_datetime(stamps[-shown:]), error[-shown:], color="#1f4e79", linewidth=0.9)
    axes[0].axhline(0, color="#c4551a", linewidth=1)
    axes[0].set_title(f"error de predicción · {title}")
    axes[0].set_ylabel("predicho − real (mw)")
    axes[0].set_xlabel("fecha")
    axes[1].hist(error, bins=40, color="#1f4e79", alpha=0.85)
    axes[1].axvline(0, color="#c4551a", linewidth=1)
    axes[1].set_title("distribución del error en prueba")
    axes[1].set_xlabel("predicho − real (mw)")
    axes[1].set_ylabel("horas")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_comparison(rows: list[dict], path: Path) -> None:
    models = [row for row in rows if row["key"] != "persistence"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)
    width = 0.24
    positions = np.arange(len(WINDOWS))
    for metric, ax in (("MAE", axes[0]), ("RMSE", axes[1])):
        for i, key in enumerate(ARCHITECTURES):
            values = [
                next(row[metric] for row in models if row["key"] == key and row["Window"] == window)
                for window in WINDOWS
            ]
            ax.bar(
                positions + (i - 1) * width,
                values,
                width=width,
                color=COLORS[key],
                label=ARCHITECTURES[key].name,
            )
        ax.set_xticks(positions, [f"{window} h" for window in WINDOWS])
        ax.set_ylabel(f"{metric.lower()} (mw)")
        ax.set_xlabel("ventana de entrada")
        ax.set_title(f"{metric.lower()} en el conjunto de prueba")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("comparación de arquitecturas lstm", fontsize=13)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_attention(weights: np.ndarray, window: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.6), constrained_layout=True)
    ax.bar(np.arange(1, window + 1), weights, color="#2f6f4e")
    ax.set_xlabel(f"posición en la ventana (1 = hora más antigua, {window} = hora más reciente)")
    ax.set_ylabel("peso medio de atención")
    ax.set_title(f"atención temporal media en prueba · ventana de {window} h")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--clean-only", action="store_true")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    frame, report = clean_dataset(DATASET_PATH)
    series_columns = ["timestamp", "ts"] + [
        "demanda_mw",
        "temperatura_c",
        "humedad_pct",
        "viento_kmh",
        "radiacion_wm2",
        "precipitacion_mm",
        "precio_kwh",
        "hora",
        "dia_semana",
        "fin_semana",
        "festivo",
        "mes",
        "periodo_dia",
    ]
    frame[series_columns].to_csv(ARTIFACTS / "clean_series.csv", index=False)
    print(
        f"Cleaning: {report['original_rows']} -> {report['clean_rows']} rows. "
        f"Gaps: {report['gaps_longer_than_one_hour']}.",
        flush=True,
    )
    if args.clean_only:
        save_json(ARTIFACTS / "cleaning.json", report)
        return

    features = build_features(frame)
    demand = frame["demanda_mw"].to_numpy(dtype=np.float64)
    stamps = frame["timestamp"].to_numpy()
    train_mask, val_mask, test_mask, cuts = chronological_split(frame)
    report["split"] = cuts

    scaler_x = StandardScaler()
    scaler_x.fit(features[train_mask])
    scaler_y = StandardScaler()
    scaler_y.fit(demand[train_mask].reshape(-1, 1))
    features_z = scaler_x.transform(features).astype(np.float32)
    demand_z = scaler_y.transform(demand.reshape(-1, 1)).astype(np.float32).ravel()
    y_mean = float(scaler_y.mean_[0])
    y_scale = float(scaler_y.scale_[0])

    preprocess = {
        "feature_names": FEATURE_NAMES,
        "x_mean": scaler_x.mean_.tolist(),
        "x_scale": scaler_x.scale_.tolist(),
        "y_mean": y_mean,
        "y_scale": y_scale,
        "split": cuts,
        "note": "Mean and standard deviation were computed on training hours only.",
    }
    save_json(ARTIFACTS / "cleaning.json", report)
    save_json(ARTIFACTS / "preprocess.json", preprocess)

    history_path = ARTIFACTS / "histories.json"
    histories: dict = {}
    if history_path.exists() and not args.retrain:
        histories = json.loads(history_path.read_text(encoding="utf-8"))

    result_rows: list[dict] = []
    test_predictions: dict = {}

    for window in WINDOWS:
        print(f"\nWindow {window} h", flush=True)
        x_train, y_train, t_train = create_sequences(features_z, demand_z, stamps, window, train_mask)
        x_val, y_val, _t_val = create_sequences(features_z, demand_z, stamps, window, val_mask)
        x_test, y_test, t_test = create_sequences(features_z, demand_z, stamps, window, test_mask)
        x_train_raw, y_train_raw, _ = create_sequences(features, demand, stamps, window, train_mask)
        x_test_raw, y_test_raw, _ = create_sequences(features, demand, stamps, window, test_mask)
        if pd.to_datetime(t_test).min() < pd.Timestamp(cuts["test_start"]):
            raise RuntimeError("A test sequence targets an hour before the test cut.")
        if pd.to_datetime(t_train).max() > pd.Timestamp(cuts["train_end"]):
            raise RuntimeError("A training sequence targets an hour after the training cut.")
        print(f"  sequences train {len(x_train)}  val {len(x_val)}  test {len(x_test)}", flush=True)

        baseline = persistence_metrics(x_test_raw, y_test_raw)
        baseline_train = persistence_metrics(x_train_raw, y_train_raw)
        result_rows.append(
            {
                "id": f"persistence_w{window}",
                "key": "persistence",
                "Model": "persistencia (referencia)",
                "Window": window,
                **baseline,
                "Epochs": 0,
                "best_epoch": 0,
                "epochs_run": 0,
                "MAE_train": baseline_train["MAE"],
                "RMSE_train": baseline_train["RMSE"],
                "R2_train": baseline_train["R2"],
                "MAE_val": None,
                "parameters": 0,
            }
        )

        for key, model_cls in ARCHITECTURES.items():
            identifier = f"{key}_w{window}"
            weight_path = MODELS / f"{identifier}.pt"
            print(f"  {model_cls.name}", flush=True)
            set_seed()
            model = create_model(key, features.shape[1])
            if weight_path.exists() and identifier in histories and not args.retrain:
                model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
                model.to(device)
                history = histories[identifier]
                print("    reusing checkpoint", flush=True)
            else:
                model, history = train_one_model(model, x_train, y_train, x_val, y_val, device)
                torch.save(model.state_dict(), weight_path)
                histories[identifier] = history
                save_json(history_path, histories)

            pred_train = predict(model, x_train, device) * y_scale + y_mean
            pred_val = predict(model, x_val, device) * y_scale + y_mean
            pred_test = predict(model, x_test, device) * y_scale + y_mean
            train_metrics = compute_metrics(y_train_raw, pred_train)
            val_metrics = compute_metrics(scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel(), pred_val)
            test_metrics = compute_metrics(y_test_raw, pred_test)
            row = {
                "id": identifier,
                "key": key,
                "Model": model_cls.name,
                "Window": window,
                **test_metrics,
                "Epochs": history["best_epoch"],
                "best_epoch": history["best_epoch"],
                "epochs_run": history["epochs_run"],
                "MAE_train": train_metrics["MAE"],
                "RMSE_train": train_metrics["RMSE"],
                "R2_train": train_metrics["R2"],
                "MAE_val": val_metrics["MAE"],
                "RMSE_val": val_metrics["RMSE"],
                "R2_val": val_metrics["R2"],
                "parameters": count_parameters(create_model(key, features.shape[1])),
            }
            result_rows.append(row)
            test_predictions[identifier] = {
                "timestamp": t_test,
                "actual": y_test_raw,
                "predicted": pred_test,
                "x_test": x_test,
            }
            print(
                f"    test MAE {test_metrics['MAE']:.2f}  RMSE {test_metrics['RMSE']:.2f}  "
                f"MAPE {test_metrics['MAPE']:.2f}%  R2 {test_metrics['R2']:.4f}",
                flush=True,
            )

    recommended = choose_recommended(result_rows)
    pd.DataFrame(result_rows).to_csv(ARTIFACTS / "results.csv", index=False)
    save_json(
        ARTIFACTS / "catalog.json",
        {
            "recommended": recommended,
            "models": [row for row in result_rows if row["key"] != "persistence"],
            "baseline": [row for row in result_rows if row["key"] == "persistence"],
        },
    )

    configure_style()
    plot_losses(histories, CHARTS / "loss_train_validation.png")
    bundle = test_predictions[recommended]
    model_name = next(row["Model"] for row in result_rows if row["id"] == recommended)
    recommended_window = next(row["Window"] for row in result_rows if row["id"] == recommended)
    title = f"{model_name}, ventana de {recommended_window} h".lower()
    plot_actual_vs_predicted(
        bundle["timestamp"],
        bundle["actual"],
        bundle["predicted"],
        title,
        CHARTS / "actual_vs_predicted.png",
    )
    plot_error(bundle["timestamp"], bundle["actual"], bundle["predicted"], title, CHARTS / "prediction_error.png")
    plot_comparison(result_rows, CHARTS / "model_comparison.png")

    pd.DataFrame(
        {
            "timestamp": bundle["timestamp"],
            "actual_demand_mw": bundle["actual"],
            "predicted_demand_mw": bundle["predicted"],
            "error_mw": bundle["predicted"] - bundle["actual"],
        }
    ).to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    recommended_key = recommended.rsplit("_w", 1)[0]
    recommended_model = create_model(recommended_key, features.shape[1])
    recommended_model.load_state_dict(torch.load(MODELS / f"{recommended}.pt", map_location=device, weights_only=True))
    recommended_model.to(device)
    importance = permutation_importance(
        recommended_model, bundle["x_test"], bundle["actual"], y_mean, y_scale, device
    )
    pd.DataFrame(importance).to_csv(ARTIFACTS / "variable_importance.csv", index=False)
    broad_groups = [
        ("recent_demand", ["demanda_mw"]),
        (
            "calendar",
            [
                "fin_semana",
                "festivo",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "month_sin",
                "month_cos",
                "period_dawn",
                "period_morning",
                "period_afternoon",
                "period_night",
            ],
        ),
        ("weather", ["temperatura_c", "humedad_pct", "viento_kmh", "radiacion_wm2", "precipitacion_mm"]),
        ("price", ["precio_kwh"]),
    ]
    grouped = permutation_importance(
        recommended_model,
        bundle["x_test"],
        bundle["actual"],
        y_mean,
        y_scale,
        device,
        groups=broad_groups,
    )
    pd.DataFrame(grouped).to_csv(ARTIFACTS / "group_importance.csv", index=False)

    attention_id = "lstm_attention_w48"
    attention_model = create_model("lstm_attention", features.shape[1])
    attention_model.load_state_dict(
        torch.load(MODELS / f"{attention_id}.pt", map_location=device, weights_only=True)
    )
    attention_model.to(device)
    attention_model.eval()
    sample = torch.from_numpy(test_predictions[attention_id]["x_test"][:512]).to(device)
    with torch.no_grad():
        weights = attention_model.attention_weights(sample).mean(dim=0).cpu().numpy()
    np.save(ARTIFACTS / "mean_attention.npy", weights)
    plot_attention(weights, 48, CHARTS / "temporal_attention.png")

    print(f"\nRecommended model: {recommended}", flush=True)
    table = pd.DataFrame(result_rows)
    print(
        table.loc[table["key"] != "persistence", ["Model", "Window", "MAE", "RMSE", "MAPE", "R2", "Epochs"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
