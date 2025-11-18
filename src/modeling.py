"""Modeling utilities for air quality analysis.

This module provides helpers for preparing aggregated AQI features/targets,
training regression models, and generating scenario-agnostic visualizations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import json
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:  # Optional dependency
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - handled gracefully below
    XGBRegressor = None  # type: ignore

try:
    import joblib
except ImportError as exc:  # pragma: no cover - sklearn installs joblib, but guard anyway
    raise ImportError("joblib is required to persist artifacts") from exc

try:  # Optional dependency for user feedback
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - keep tqdm optional
    tqdm = None  # type: ignore

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_STATE_COL = "State Name"
DEFAULT_COUNTY_COL = "county Name"
DEFAULT_STATE_CODE_COL = "State Code"
DEFAULT_COUNTY_CODE_COL = "County Code"
DEFAULT_DATE_COL = "Date"
DEFAULT_AQI_VALUE_COL = "AQI"

STATE_ABBREVIATIONS = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

DEFAULT_STATE_METRICS = ["avg_aqi", "median_aqi", "max_aqi", "min_aqi", "aqi_observations"]
COUNTY_GEOJSON_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)
_COUNTY_GEOJSON_CACHE: Dict[str, Any] | None = None


def _load_county_geojson() -> Dict[str, Any]:
    """Fetch and cache the US counties GeoJSON needed for county-level choropleths."""

    global _COUNTY_GEOJSON_CACHE
    if _COUNTY_GEOJSON_CACHE is not None:
        return _COUNTY_GEOJSON_CACHE

    import urllib.request

    LOGGER.info("Downloading US counties GeoJSON from %s", COUNTY_GEOJSON_URL)
    with urllib.request.urlopen(COUNTY_GEOJSON_URL) as response:  # pragma: no cover - I/O heavy
        _COUNTY_GEOJSON_CACHE = json.loads(response.read().decode("utf-8"))
    return _COUNTY_GEOJSON_CACHE


@dataclass
class DatasetSplits:
    """Container for train/test splits."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class ModelResult:
    """Encapsulates model artifacts and evaluation outputs."""

    name: str
    model: object
    predictions: pd.Series
    metrics: Mapping[str, float]
    feature_importances: pd.Series


def load_csv_dataset(path: str | Path, parse_dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load a CSV dataset with optional date parsing."""

    LOGGER.info("Loading dataset from %s", path)
    df = pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None)
    LOGGER.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
    return df


def aggregate_aqi_by_region_year(
    aqi_df: pd.DataFrame,
    state_col: str = DEFAULT_STATE_COL,
    county_col: str = DEFAULT_COUNTY_COL,
    date_col: str = DEFAULT_DATE_COL,
    value_col: str = DEFAULT_AQI_VALUE_COL,
) -> pd.DataFrame:
    """Aggregate AQI measurements per state/county/month with summary stats."""

    df = aqi_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month_start"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["month_start"].dt.year
    df["month"] = df["month_start"].dt.month
    group_cols = [state_col, county_col, "year", "month", "month_start"]
    aggregated = (
        df.groupby(group_cols)
        .agg(
            avg_aqi=(value_col, "mean"),
            median_aqi=(value_col, "median"),
            max_aqi=(value_col, "max"),
            min_aqi=(value_col, "min"),
            aqi_observations=(value_col, "count"),
        )
        .reset_index()
    )
    LOGGER.info("Aggregated AQI data shape: %s", aggregated.shape)
    return aggregated


def create_feature_target_matrices(
    dataset: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Split a dataset into features/target, inferring numeric features when unspecified."""

    df = dataset.copy()
    if dropna:
        df = df.dropna(subset=[target_col])

    if feature_cols is None:
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]
    LOGGER.info("Feature matrix shape: %s; Target length: %d", X.shape, len(y))
    return X, y, feature_cols


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplits:
    """Create deterministic train/test splits."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return DatasetSplits(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics for regression outputs."""

    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _series_from_values(values: np.ndarray, index: pd.Index, name: str) -> pd.Series:
    return pd.Series(values, index=index, name=name)


def train_linear_regression(
    splits: DatasetSplits,
    feature_names: Iterable[str],
) -> ModelResult:
    """Train/evaluate a Linear Regression model."""

    model = LinearRegression()
    model.fit(splits.X_train, splits.y_train)
    predictions = model.predict(splits.X_test)
    metrics = evaluate_predictions(splits.y_test, predictions)
    feature_importances = _series_from_values(
        model.coef_, pd.Index(feature_names), "coefficient"
    )
    return ModelResult(
        name="linear_regression",
        model=model,
        predictions=_series_from_values(predictions, splits.y_test.index, "prediction"),
        metrics=metrics,
        feature_importances=feature_importances,
    )


def train_random_forest(
    splits: DatasetSplits,
    feature_names: Iterable[str],
    n_estimators: int = 300,
    random_state: int = 42,
) -> ModelResult:
    """Train/evaluate a Random Forest regressor."""

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(splits.X_train, splits.y_train)
    predictions = model.predict(splits.X_test)
    metrics = evaluate_predictions(splits.y_test, predictions)
    feature_importances = _series_from_values(
        model.feature_importances_, pd.Index(feature_names), "importance"
    )
    return ModelResult(
        name="random_forest",
        model=model,
        predictions=_series_from_values(predictions, splits.y_test.index, "prediction"),
        metrics=metrics,
        feature_importances=feature_importances,
    )


def train_xgboost(
    splits: DatasetSplits,
    feature_names: Iterable[str],
    random_state: int = 42,
) -> ModelResult:
    """Train/evaluate an XGBoost regressor (if available)."""

    if XGBRegressor is None:
        raise RuntimeError(
            "XGBoost is not installed. Install xgboost to enable this trainer."
        )

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=random_state,
    )
    model.fit(splits.X_train, splits.y_train)
    predictions = model.predict(splits.X_test)
    metrics = evaluate_predictions(splits.y_test, predictions)
    feature_importances = _series_from_values(
        model.feature_importances_, pd.Index(feature_names), "gain"
    )
    return ModelResult(
        name="xgboost",
        model=model,
        predictions=_series_from_values(predictions, splits.y_test.index, "prediction"),
        metrics=metrics,
        feature_importances=feature_importances,
    )


def summarize_state_statistics(
    dataset: pd.DataFrame,
    state_col: str = DEFAULT_STATE_COL,
    year_col: str = "year",
    metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Aggregate county-level rows into state/time summaries for visualization."""

    if state_col not in dataset.columns:
        raise ValueError(f"Dataset must include '{state_col}' column")
    if year_col not in dataset.columns:
        raise ValueError(f"Dataset must include '{year_col}' column")

    default_metrics = [col for col in DEFAULT_STATE_METRICS if col in dataset.columns]
    metric_list = list(metrics) if metrics else default_metrics
    if not metric_list:
        raise ValueError("No numeric metrics available to summarize at the state level")

    missing_metrics = [col for col in metric_list if col not in dataset.columns]
    if missing_metrics:
        raise ValueError(f"Missing required metric columns: {missing_metrics}")

    keep_cols = [state_col, year_col] + metric_list
    grouped = (
        dataset[keep_cols]
        .dropna(subset=[state_col, year_col])
        .groupby([state_col, year_col])[metric_list]
        .mean()
        .reset_index()
    )
    grouped["state_abbrev"] = grouped[state_col].map(STATE_ABBREVIATIONS)
    grouped = grouped.dropna(subset=["state_abbrev"])
    if year_col == "year":
        grouped["year"] = grouped["year"].astype(int)
    else:
        values = grouped[year_col]
        if pd.api.types.is_datetime64_any_dtype(values):
            grouped["year"] = pd.to_datetime(values).dt.year.astype(int)
        else:
            grouped["year"] = pd.to_numeric(values, errors="coerce").astype(int)
    return grouped


def _prepare_time_axis(values: pd.Series, target_value: Any) -> tuple[np.ndarray, float]:
    """Convert a time-like series and target value into numeric space for regression."""

    series = pd.Series(values).dropna()
    if series.empty:
        return np.array([]), float("nan")

    if pd.api.types.is_datetime64_any_dtype(series) or isinstance(series.iloc[0], pd.Timestamp):
        timestamps = pd.to_datetime(series)
        numeric = timestamps.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
        target_numeric = pd.Timestamp(target_value).toordinal()
    else:
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        target_numeric = float(pd.to_numeric([target_value], errors="coerce")[0])
    base = numeric.min()
    numeric = numeric - base
    target_numeric -= base
    return numeric, target_numeric


def project_metric_trend_by_state(
    state_stats: pd.DataFrame,
    metric: str,
    target_year: int,
    *,
    state_col: str = DEFAULT_STATE_COL,
    time_col: str = "year",
    target_time_value: Any | None = None,
    min_history_points: int = 2,
) -> pd.DataFrame:
    """Extrapolate a metric for each state using a simple linear trend."""

    required_cols = {state_col, metric, time_col}
    missing = required_cols - set(state_stats.columns)
    if missing:
        raise ValueError(f"state_stats must include columns: {sorted(missing)}")

    if time_col != "year" and target_time_value is None:
        raise ValueError("target_time_value must be provided when using a non-year time column")

    projections: List[Dict[str, Any]] = []
    for state, group in state_stats.groupby(state_col):
        history = group.dropna(subset=[time_col, metric]).sort_values(time_col)
        if history.empty:
            continue

        axis_values = history[time_col]
        values = history[metric].astype(float).to_numpy()
        target_value = target_year if target_time_value is None else target_time_value
        numeric_axis, numeric_target = _prepare_time_axis(axis_values, target_value)
        if numeric_axis.size == 0:
            continue

        unique_points = np.unique(numeric_axis)
        if len(unique_points) >= min_history_points:
            slope, intercept = np.polyfit(numeric_axis, values, 1)
            forecast_value = slope * numeric_target + intercept
        else:
            forecast_value = values[-1]

        history_min = float(np.nanmin(values))
        history_max = float(np.nanmax(values))
        forecast_value = max(history_min, min(history_max, forecast_value))
        forecast_value = max(0.0, min(200.0, forecast_value))

        state_abbrev = (
            history["state_abbrev"].iloc[-1]
            if "state_abbrev" in history.columns
            else STATE_ABBREVIATIONS.get(state)
        )
        row: Dict[str, Any] = {
            state_col: state,
            "year": int(target_year),
            "state_abbrev": state_abbrev,
            metric: float(forecast_value),
        }
        if time_col != "year":
            row[time_col] = target_value
        projections.append(row)

    if not projections:
        raise ValueError("Unable to build projections; no valid state histories were found.")
    return pd.DataFrame(projections)


def plot_us_state_choropleth(
    state_stats: pd.DataFrame,
    metric: str,
    year: Optional[int] = None,
    output_path: str | Path | None = None,
    state_col: str = DEFAULT_STATE_COL,
) -> Any:
    """Render a choropleth map of the United States using Plotly."""

    try:
        import plotly.express as px  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotly is required to build the US choropleth map. Install plotly>=5.0."
        ) from exc

    if "year" not in state_stats.columns:
        raise ValueError("state_stats must include a 'year' column")
    if "state_abbrev" not in state_stats.columns:
        raise ValueError("state_stats must include a 'state_abbrev' column")
    if state_col not in state_stats.columns:
        raise ValueError(f"state_stats must include '{state_col}' column")
    if metric not in state_stats.columns:
        raise ValueError(f"Metric '{metric}' not found in state_stats")

    df = state_stats.copy()
    target_year = int(df["year"].max()) if year is None else year
    year_slice = df[df["year"] == target_year]
    if year_slice.empty:
        raise ValueError(f"No rows available for year {target_year}")

    metric_label = metric.replace("_", " ").title()
    fig = px.choropleth(
        year_slice,
        locations="state_abbrev",
        locationmode="USA-states",
        color=metric,
        scope="usa",
        color_continuous_scale="YlOrRd",
        hover_name=state_col,
        labels={metric: metric_label},
        title=f"{metric_label} by State ({target_year})",
    )
    if output_path:
        path = Path(output_path)
        if path.suffix.lower() == ".html":
            fig.write_html(str(path))
        else:  # Attempt static export; requires kaleido
            try:
                fig.write_image(str(path))
            except ValueError as exc:  # pragma: no cover - depends on kaleido
                raise ValueError(
                    "Writing static image files requires the 'kaleido' package. "
                    "Install kaleido or provide an .html output path."
                ) from exc
    return fig


def plot_us_county_choropleth(
    county_stats: pd.DataFrame,
    metric: str,
    year: Optional[int] = None,
    output_path: str | Path | None = None,
    state_col: str = DEFAULT_STATE_COL,
    county_col: str = DEFAULT_COUNTY_COL,
    fips_col: str = "fips",
) -> Any:
    """Render a county-level choropleth map for the United States."""

    try:
        import plotly.express as px  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotly is required to build the US county choropleth map. Install plotly>=5.0."
        ) from exc

    required_cols = {"year", metric, county_col, state_col, fips_col}
    missing_cols = required_cols - set(county_stats.columns)
    if missing_cols:
        raise ValueError(f"county_stats must include columns: {sorted(missing_cols)}")

    df = county_stats.copy()
    target_year = int(df["year"].max()) if year is None else year
    year_slice = df[df["year"] == target_year]
    if year_slice.empty:
        raise ValueError(f"No rows available for year {target_year}")

    metric_label = metric.replace("_", " ").title()
    geojson = _load_county_geojson()
    fig = px.choropleth(
        year_slice,
        geojson=geojson,
        locations=fips_col,
        color=metric,
        scope="usa",
        hover_name=county_col,
        hover_data={
            state_col: True,
            "year": False,
        },
        labels={metric: metric_label},
        title=f"{metric_label} by County ({target_year})",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    if output_path:
        path = Path(output_path)
        if path.suffix.lower() == ".html":
            fig.write_html(str(path))
        else:
            try:
                fig.write_image(str(path))
            except ValueError as exc:  # pragma: no cover - depends on kaleido
                raise ValueError(
                    "Writing static image files requires the 'kaleido' package. "
                    "Install kaleido or provide an .html output path."
                ) from exc
    return fig


def plot_top_states_bar_chart(
    state_stats: pd.DataFrame,
    metric: str,
    year: Optional[int] = None,
    top_n: int = 10,
    output_path: str | Path | None = None,
    state_col: str = DEFAULT_STATE_COL,
) -> Any:
    """Create a Matplotlib bar chart with the leaders for a metric."""

    import matplotlib.pyplot as plt  # Lazy import to avoid unnecessary dependency

    if "year" not in state_stats.columns:
        raise ValueError("state_stats must include a 'year' column")
    if state_col not in state_stats.columns:
        raise ValueError(f"state_stats must include '{state_col}' column")
    if metric not in state_stats.columns:
        raise ValueError(f"Metric '{metric}' not found in state_stats")

    df = state_stats.copy()
    target_year = int(df["year"].max()) if year is None else year
    filtered = df[df["year"] == target_year]
    if filtered.empty:
        raise ValueError(f"No rows available for year {target_year}")

    ranked = filtered.nlargest(top_n, metric)
    ranked = ranked.sort_values(metric)
    metric_label = metric.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ranked[state_col], ranked[metric], color="#d73027")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("State")
    ax.set_title(f"Top {min(top_n, len(ranked))} States by {metric_label} ({target_year})")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if output_path:
        fig.savefig(Path(output_path), bbox_inches="tight")
    return fig


def plot_yearly_metric_trend(
    state_stats: pd.DataFrame,
    metric: str,
    agg: str
    | Callable[[pd.Series], Any]
    | Mapping[str, str | Callable[[pd.Series], Any]]
    | None = "mean",
    output_path: str | Path | None = None,
    time_col: str = "year",
) -> Any:
    """Plot a trend line for a metric aggregated across states."""

    import matplotlib.dates as mdates  # Lazy import for date formatting
    import matplotlib.pyplot as plt  # Lazy import

    if time_col not in state_stats.columns:
        raise ValueError(f"state_stats must include a '{time_col}' column")
    if metric not in state_stats.columns:
        raise ValueError(f"Metric '{metric}' not found in state_stats")

    agg_expr = {metric: agg} if isinstance(agg, str) or callable(agg) else agg
    if agg_expr is None:
        agg_expr = {metric: "mean"}

    grouped = (
        state_stats.groupby(time_col)
        .agg(agg_expr)
        .reset_index()
        .sort_values(time_col)
    )
    metric_label = metric.replace("_", " ").title()
    treat_as_daily = time_col != "year"
    x_values = grouped[time_col]
    if treat_as_daily:
        x_values = pd.to_datetime(x_values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_values, grouped[metric], marker="o", color="#1b9e77")
    ax.set_xlabel("Date" if treat_as_daily else "Year")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} Trend Across States")
    ax.grid(True, linestyle="--", alpha=0.4)
    if treat_as_daily:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        start_date = pd.to_datetime(x_values.min())
        end_date = pd.to_datetime(x_values.max())
        ax.set_xlim(start_date, end_date)
        fig.autofmt_xdate()
    fig.tight_layout()

    if output_path:
        fig.savefig(Path(output_path), bbox_inches="tight")
    return fig


class AQIModelTrainer:
    """Train and evaluate models on aggregated AQI datasets."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.dataset = dataset.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        _, _, inferred_cols = create_feature_target_matrices(
            dataset, target_col=target_col, feature_cols=feature_cols
        )
        self.feature_cols = inferred_cols
        self.trainers = {
            "linear_regression": train_linear_regression,
            "random_forest": train_random_forest,
        }
        if XGBRegressor is not None:
            self.trainers["xgboost"] = train_xgboost
        else:
            LOGGER.warning("XGBoost not available; skipping XGBoost trainer")

    def _split(self) -> DatasetSplits:
        X = self.dataset[self.feature_cols]
        y = self.dataset[self.target_col]
        return make_train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def run(self) -> Dict[str, ModelResult]:
        splits = self._split()
        results: Dict[str, ModelResult] = {}
        trainer_items = list(self.trainers.items())
        iterator = (
            tqdm(
                trainer_items,
                desc="Training AQI models",
                unit="model",
                leave=False,
            )
            if tqdm is not None
            else trainer_items
        )
        for name, trainer in iterator:
            LOGGER.info("Training %s model", name)
            results[name] = trainer(splits, feature_names=self.feature_cols)
        return results

    def persist_results(
        self,
        results: Mapping[str, ModelResult],
        output_dir: str | Path,
    ) -> None:
        """Persist trained models, predictions, and metrics to disk."""

        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        for name, result in results.items():
            model_path = base / f"{name}.joblib"
            preds_path = base / f"{name}_predictions.csv"
            metrics_path = base / f"{name}_metrics.json"
            importances_path = base / f"{name}_feature_importances.csv"

            joblib.dump(result.model, model_path)
            result.predictions.to_csv(preds_path)
            result.feature_importances.to_csv(importances_path)
            with metrics_path.open("w", encoding="utf-8") as fp:
                json.dump(result.metrics, fp, indent=2)
            LOGGER.info("Persisted artifacts for %s to %s", name, base)


__all__ = [
    "DatasetSplits",
    "ModelResult",
    "AQIModelTrainer",
    "load_csv_dataset",
    "aggregate_aqi_by_region_year",
    "create_feature_target_matrices",
    "make_train_test_split",
    "train_linear_regression",
    "train_random_forest",
    "train_xgboost",
    "summarize_state_statistics",
    "plot_us_state_choropleth",
    "plot_us_county_choropleth",
    "plot_top_states_bar_chart",
    "plot_yearly_metric_trend",
    "project_metric_trend_by_state",
]
