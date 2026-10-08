from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_PATH = Path(__file__).resolve().parent / "dataset_demanda_energia_LSTM_2025_2026.csv"

MEASUREMENT_COLUMNS = [
    "demanda_mw",
    "temperatura_c",
    "humedad_pct",
    "viento_kmh",
    "radiacion_wm2",
    "precipitacion_mm",
    "precio_kwh",
]

INPUT_COLUMNS = MEASUREMENT_COLUMNS + [
    "hora",
    "dia_semana",
    "fin_semana",
    "festivo",
    "mes",
    "periodo_dia",
]

FEATURE_NAMES = [
    "demanda_mw",
    "temperatura_c",
    "humedad_pct",
    "viento_kmh",
    "radiacion_wm2",
    "precipitacion_mm",
    "precio_kwh",
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
]

VARIABLE_GROUPS = [
    ("demanda_mw", ["demanda_mw"]),
    ("temperatura_c", ["temperatura_c"]),
    ("humedad_pct", ["humedad_pct"]),
    ("viento_kmh", ["viento_kmh"]),
    ("radiacion_wm2", ["radiacion_wm2"]),
    ("precipitacion_mm", ["precipitacion_mm"]),
    ("precio_kwh", ["precio_kwh"]),
    ("fin_semana", ["fin_semana"]),
    ("festivo", ["festivo"]),
    ("hour", ["hour_sin", "hour_cos"]),
    ("day_of_week", ["dow_sin", "dow_cos"]),
    ("month", ["month_sin", "month_cos"]),
    ("period_of_day", ["period_dawn", "period_morning", "period_afternoon", "period_night"]),
]

PERIODS = ("dawn", "morning", "afternoon", "night")

_PERIOD_ALIASES = {
    "madrugada": "dawn",
    "manana": "morning",
    "maana": "morning",
    "tarde": "afternoon",
    "noche": "night",
    "dawn": "dawn",
    "morning": "morning",
    "afternoon": "afternoon",
    "night": "night",
}


def period_from_hour(hour: int) -> str:
    hour = int(hour)
    if hour <= 5:
        return "dawn"
    if hour <= 11:
        return "morning"
    if hour <= 17:
        return "afternoon"
    return "night"


def _normalize_period_text(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower().str.normalize("NFKD")
    text = text.str.encode("ascii", "ignore").str.decode("ascii")
    text = text.str.replace(r"[^a-z]", "", regex=True)
    return text.replace(_PERIOD_ALIASES)


def clean_dataset(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    path = Path(path) if path else DATASET_PATH
    raw = pd.read_csv(path)
    report: dict = {"original_rows": int(len(raw))}

    raw["ts"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    report["invalid_timestamps"] = int(raw["ts"].isna().sum())
    raw = raw.loc[raw["ts"].notna()].copy()

    before = len(raw)
    raw = raw.drop_duplicates(keep="first")
    report["exact_duplicates_removed"] = int(before - len(raw))
    conflicts = int(raw.duplicated(subset=["timestamp"]).sum())
    report["timestamps_with_conflicting_values"] = conflicts
    if conflicts:
        raise ValueError(
            "Repeated timestamps have different values. "
            "They are not averaged; that would invent a measurement."
        )

    raw = raw.sort_values("ts").reset_index(drop=True)
    report["sorted_chronologically"] = True

    hours = raw["ts"].dt.hour
    report["hour_mismatch"] = int((raw["hora"] != hours).sum())
    report["weekday_mismatch"] = int((raw["dia_semana"] != raw["ts"].dt.dayofweek).sum())
    report["month_mismatch"] = int((raw["mes"] != raw["ts"].dt.month).sum())
    weekend = (raw["ts"].dt.dayofweek >= 5).astype(int)
    report["weekend_mismatch"] = int((raw["fin_semana"] != weekend).sum())

    period_text = _normalize_period_text(raw["periodo_dia"])
    period_hour = hours.map(period_from_hour)
    report["period_label_not_canonical"] = int(
        (raw["periodo_dia"].astype(str).str.strip() != period_hour).sum()
    )
    report["period_inconsistent_with_hour"] = int((period_text != period_hour).sum())

    missing = raw[MEASUREMENT_COLUMNS].isna().any(axis=1)
    report["rows_with_missing_measurements"] = int(missing.sum())
    report["missing_by_column"] = {
        col: int(raw[col].isna().sum()) for col in MEASUREMENT_COLUMNS if raw[col].isna().any()
    }

    rules = {
        "demand_outside_300_to_1050_mw": (raw["demanda_mw"] < 300) | (raw["demanda_mw"] > 1050),
        "temperature_outside_0_to_40_c": (raw["temperatura_c"] < 0) | (raw["temperatura_c"] > 40),
        "humidity_outside_0_to_100": (raw["humedad_pct"] < 0) | (raw["humedad_pct"] > 100),
        "wind_negative_or_above_80_kmh": (raw["viento_kmh"] < 0) | (raw["viento_kmh"] > 80),
        "price_non_positive_or_above_60": (raw["precio_kwh"] <= 0) | (raw["precio_kwh"] > 60),
        "negative_radiation": raw["radiacion_wm2"] < 0,
        "negative_precipitation": raw["precipitacion_mm"] < 0,
    }
    rules = {name: mask.fillna(False) for name, mask in rules.items()}
    report["rows_by_anomaly_rule"] = {name: int(mask.sum()) for name, mask in rules.items()}
    anomalies = np.logical_or.reduce([mask.to_numpy() for mask in rules.values()])
    report["rows_with_any_anomaly"] = int(anomalies.sum())

    drop = missing.to_numpy() | anomalies
    report["rows_dropped_missing_or_anomaly"] = int(drop.sum())
    report["threshold_reasons"] = {
        "demand_mw": (
            "La masa de la serie está entre unos 350 y 970 MW. Por debajo de 300 MW o por encima "
            "de 1050 MW hay picos aislados que a la hora siguiente vuelven a 400-800 MW: no es una "
            "rampa real. Esas filas se eliminan, no se reemplazan."
        ),
        "temperature_c": (
            "El clima de esta serie vive entre unos 4 °C y 33 °C. Bajo 0 °C o sobre 40 °C no hay "
            "continuidad con esa masa: son fallas de sensor. No se sustituyen por un promedio."
        ),
        "humidity_pct": "Una humedad fuera de 0-100 % es físicamente imposible. La fila se elimina.",
        "wind_kmh": (
            "Una velocidad negativa es imposible. Por encima de 80 km/h hay un hueco respecto del "
            "tramo continuo (hasta unos 67 km/h) y los valores llegan a 117 km/h. Esas filas se eliminan."
        ),
        "price_kwh": (
            "Un precio no positivo, o por encima de 60, queda fuera de la banda observada "
            "(el percentil 99.9 está cerca de 42). La fila se elimina, no se recorta."
        ),
        "radiation_and_precipitation": "Un valor negativo sería imposible. En este archivo no aparece.",
        "missing_values": "No se imputa nada. Una hora sin medición deja un hueco.",
        "period_of_day": (
            "El texto trae mayúsculas, espacios y la palabra mañana con distinta codificación. "
            "Algunas etiquetas tampoco coinciden con la hora. Hora, día, mes y fin de semana ya "
            "coinciden con la marca de tiempo, así que el periodo se recalcula desde la hora."
        ),
    }

    clean = raw.loc[~drop].copy().reset_index(drop=True)
    clean["hora"] = clean["ts"].dt.hour.astype(int)
    clean["dia_semana"] = clean["ts"].dt.dayofweek.astype(int)
    clean["mes"] = clean["ts"].dt.month.astype(int)
    clean["fin_semana"] = (clean["dia_semana"] >= 5).astype(int)
    clean["festivo"] = clean["festivo"].astype(int)
    clean["periodo_dia"] = clean["hora"].map(period_from_hour)
    clean["timestamp"] = clean["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    delta_hours = clean["ts"].diff().dt.total_seconds().div(3600)
    report["clean_rows"] = int(len(clean))
    report["gaps_longer_than_one_hour"] = int((delta_hours.iloc[1:] != 1).sum())
    report["start"] = clean["timestamp"].iloc[0]
    report["end"] = clean["timestamp"].iloc[-1]

    nxt = clean["demanda_mw"].shift(-1)
    consecutive = clean["ts"].shift(-1) - clean["ts"] == pd.Timedelta(hours=1)
    comparable = consecutive & clean["demanda_objetivo"].notna()
    difference = (clean.loc[comparable, "demanda_objetivo"] - nxt.loc[comparable]).abs()
    report["target_column_vs_next_hour_demand"] = {
        "consecutive_pairs_with_target": int(comparable.sum()),
        "exact_matches": int((difference < 1e-6).sum()),
        "mismatches": int((difference >= 1e-6).sum()),
    }
    return clean, report


def build_features(frame: pd.DataFrame) -> np.ndarray:
    hour = frame["hora"].to_numpy(dtype=np.float64)
    weekday = frame["dia_semana"].to_numpy(dtype=np.float64)
    month = frame["mes"].to_numpy(dtype=np.float64)
    period = frame["periodo_dia"].astype(str)
    columns = [
        frame["demanda_mw"].to_numpy(dtype=np.float64),
        frame["temperatura_c"].to_numpy(dtype=np.float64),
        frame["humedad_pct"].to_numpy(dtype=np.float64),
        frame["viento_kmh"].to_numpy(dtype=np.float64),
        frame["radiacion_wm2"].to_numpy(dtype=np.float64),
        frame["precipitacion_mm"].to_numpy(dtype=np.float64),
        frame["precio_kwh"].to_numpy(dtype=np.float64),
        frame["fin_semana"].to_numpy(dtype=np.float64),
        frame["festivo"].to_numpy(dtype=np.float64),
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * weekday / 7),
        np.cos(2 * np.pi * weekday / 7),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        (period == "dawn").to_numpy(dtype=np.float64),
        (period == "morning").to_numpy(dtype=np.float64),
        (period == "afternoon").to_numpy(dtype=np.float64),
        (period == "night").to_numpy(dtype=np.float64),
    ]
    matrix = np.column_stack(columns)
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("Feature column order does not match FEATURE_NAMES.")
    return matrix


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.70, val_fraction: float = 0.15):
    count = len(frame)
    train_cut = int(count * train_fraction)
    val_cut = int(count * (train_fraction + val_fraction))
    if not (0 < train_cut < val_cut < count):
        raise RuntimeError("Chronological cuts produced an empty split.")
    index = np.arange(count)
    train_mask = index < train_cut
    val_mask = (index >= train_cut) & (index < val_cut)
    test_mask = index >= val_cut
    cuts = {
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_start": frame["timestamp"].iloc[0],
        "train_end": frame["timestamp"].iloc[train_cut - 1],
        "val_start": frame["timestamp"].iloc[train_cut],
        "val_end": frame["timestamp"].iloc[val_cut - 1],
        "test_start": frame["timestamp"].iloc[val_cut],
        "test_end": frame["timestamp"].iloc[-1],
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": round(1 - train_fraction - val_fraction, 2),
    }
    return train_mask, val_mask, test_mask, cuts


def create_sequences(
    features: np.ndarray,
    demand: np.ndarray,
    stamps: np.ndarray,
    window: int,
    target_mask: np.ndarray,
):
    if len(features) != len(demand) or len(features) != len(stamps):
        raise RuntimeError("Features, demand, and timestamps are not aligned.")
    parsed = pd.to_datetime(pd.Series(stamps))
    delta = parsed.diff().dt.total_seconds().div(3600).to_numpy()
    xs: list[np.ndarray] = []
    ys: list[float] = []
    times: list[str] = []
    for target in range(window, len(features)):
        if not target_mask[target]:
            continue
        span = delta[target - window + 1 : target + 1]
        if span.shape[0] != window or not np.all(span == 1):
            continue
        xs.append(features[target - window : target])
        ys.append(float(demand[target]))
        times.append(str(stamps[target]))
    if not xs:
        raise RuntimeError(f"No continuous {window}-hour sequences in this split.")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), np.asarray(times)


def last_window(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    stamps = pd.to_datetime(frame["timestamp"])
    delta = stamps.diff().dt.total_seconds().div(3600).to_numpy()
    for end in range(len(frame) - 1, window - 2, -1):
        start = end - window + 1
        if start < 0:
            break
        span = delta[start + 1 : end + 1]
        if span.shape[0] == window - 1 and np.all(span == 1):
            return frame.iloc[start : end + 1].copy()
    raise RuntimeError(f"No {window} consecutive hours at the end of the clean series.")


def rows_to_features(rows: list[dict]) -> np.ndarray:
    frame = pd.DataFrame(rows)
    required = set(INPUT_COLUMNS)
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError("faltan columnas: " + ", ".join(missing))
    period = frame["periodo_dia"].astype(str).str.strip().str.lower()
    period = period.str.replace("ñ", "n", regex=False).str.replace("á", "a", regex=False)
    frame["periodo_dia"] = period.replace(_PERIOD_ALIASES)
    return build_features(frame)


def validate_rows(rows: list[dict], window: int) -> str | None:
    if len(rows) != window:
        return f"se esperaban {window} horas y llegaron {len(rows)}."
    for index, row in enumerate(rows, start=1):
        try:
            demand = float(row["demanda_mw"])
            temperature = float(row["temperatura_c"])
            humidity = float(row["humedad_pct"])
            wind = float(row["viento_kmh"])
            radiation = float(row["radiacion_wm2"])
            precipitation = float(row["precipitacion_mm"])
            price = float(row["precio_kwh"])
            hour = int(row["hora"])
            weekday = int(row["dia_semana"])
            weekend = int(row["fin_semana"])
            holiday = int(row["festivo"])
            month = int(row["mes"])
        except (KeyError, TypeError, ValueError):
            return f"la hora {index} de la ventana tiene un valor no numérico."
        period = str(row.get("periodo_dia", "")).strip().lower().replace("ñ", "n").replace("á", "a")
        period = _PERIOD_ALIASES.get(period, period)
        if demand <= 0:
            return f"la demanda de la hora {index} tiene que ser positiva."
        if not 0 <= hour <= 23:
            return f"la hora {index} debe estar entre 0 y 23."
        if not 0 <= weekday <= 6:
            return f"el día de la semana de la fila {index} debe estar entre 0 (lunes) y 6 (domingo)."
        if weekend not in (0, 1) or holiday not in (0, 1):
            return f"fin de semana y festivo de la fila {index} deben ser 0 o 1."
        if not 1 <= month <= 12:
            return f"el mes de la fila {index} debe estar entre 1 y 12."
        if period not in PERIODS:
            return f"el periodo de la fila {index} debe ser madrugada, mañana, tarde o noche."
        if not np.isfinite([demand, temperature, humidity, wind, radiation, precipitation, price]).all():
            return f"la hora {index} tiene un valor vacío o infinito."
    return None


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
