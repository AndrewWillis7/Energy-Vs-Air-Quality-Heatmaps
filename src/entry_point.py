"""Command-line entry point for air quality modeling."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple
import webbrowser

import pandas as pd

from modeling import (
    AQIModelTrainer,
    DEFAULT_AQI_VALUE_COL,
    DEFAULT_COUNTY_COL,
    DEFAULT_COUNTY_CODE_COL,
    DEFAULT_DATE_COL,
    DEFAULT_STATE_COL,
    DEFAULT_STATE_CODE_COL,
    aggregate_aqi_by_region_year,
    load_csv_dataset,
    plot_top_states_bar_chart,
    plot_us_county_choropleth,
    plot_us_state_choropleth,
    plot_yearly_metric_trend,
    project_metric_trend_by_state,
    summarize_state_statistics,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_AQI_CSV = DATA_DIR / "AQI2025.csv"
DEFAULT_VIZ_DIR = PROJECT_ROOT / "visualizations"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AQI projection models using aggregated AQI datasets"
    )
    parser.add_argument(
        "--aqi-csv",
        default=str(DEFAULT_AQI_CSV),
        help="Path to raw AQI measurements (defaults to data/AQI2025.csv)",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        help="Optional directory where trained models and outputs will be written",
    )
    parser.add_argument(
        "--target-col",
        default="avg_aqi",
        help="Target column from the aggregated dataset",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=DEFAULT_VIZ_DIR,
        help=(
            "Directory where visualization artifacts (US map + Matplotlib charts) will be saved "
            "(defaults to visualizations/)"
        ),
    )
    parser.add_argument(
        "--choropleth-metric",
        default="avg_aqi",
        help="Metric used to color the spatial map and bar chart",
    )
    parser.add_argument(
        "--choropleth-geo",
        choices=("state", "county"),
        default="county",
        help="Geographic granularity for the Plotly choropleth (state or county level).",
    )
    parser.add_argument(
        "--viz-year",
        type=int,
        help="Specific year to visualize. Defaults to the latest year in the dataset.",
    )
    parser.add_argument(
        "--forecast-year-offset",
        type=int,
        default=1,
        help=(
            "If greater than zero, extrapolate the selected metric this many years beyond "
            "the visualization year using per-state trends."
        ),
    )
    parser.add_argument(
        "--top-n-states",
        type=int,
        default=10,
        help="Number of states shown in the horizontal bar chart visualization",
    )
    args = parser.parse_args(argv)
    if args.forecast_year_offset < 0:
        parser.error("--forecast-year-offset must be zero or positive")
    return args


def load_and_prepare_dataset(aqi_csv: str) -> Tuple["pd.DataFrame", "pd.DataFrame"]:  # type: ignore[name-defined]
    import pandas as pd

    aqi_df = load_csv_dataset(aqi_csv)
    aqi_agg = aggregate_aqi_by_region_year(aqi_df)
    if aqi_agg.empty:
        raise ValueError("Aggregated AQI dataset is empty. Check source file and column names.")
    return aqi_agg, aqi_df


def _build_county_visualization_table(
    dataset: "pd.DataFrame", raw_dataset: "pd.DataFrame | None"
) -> "pd.DataFrame":
    if raw_dataset is None:
        raise ValueError("County-level visualizations require the raw dataset to compute FIPS codes.")

    required_cols = {
        DEFAULT_STATE_COL,
        DEFAULT_COUNTY_COL,
        DEFAULT_STATE_CODE_COL,
        DEFAULT_COUNTY_CODE_COL,
    }
    missing = required_cols - set(raw_dataset.columns)
    if missing:
        raise ValueError(f"Raw dataset missing required columns for county plotting: {sorted(missing)}")

    metadata = (
        raw_dataset[list(required_cols)]
        .drop_duplicates(subset=[DEFAULT_STATE_COL, DEFAULT_COUNTY_COL])
        .copy()
    )
    metadata["state_code"] = metadata[DEFAULT_STATE_CODE_COL].astype(str).str.zfill(2)
    metadata["county_code"] = metadata[DEFAULT_COUNTY_CODE_COL].astype(str).str.zfill(3)
    metadata["fips"] = metadata["state_code"] + metadata["county_code"]

    county_stats = dataset.merge(
        metadata[
            [
                DEFAULT_STATE_COL,
                DEFAULT_COUNTY_COL,
                "state_code",
                "county_code",
                "fips",
            ]
        ],
        on=[DEFAULT_STATE_COL, DEFAULT_COUNTY_COL],
        how="left",
    )
    missing_fips = county_stats["fips"].isna().sum()
    if missing_fips:
        LOGGER.warning("Unable to compute FIPS codes for %d county rows", missing_fips)
        county_stats = county_stats.dropna(subset=["fips"])
    return county_stats


def create_visualizations(
    dataset: "pd.DataFrame",
    args: argparse.Namespace,
    raw_dataset: "pd.DataFrame | None" = None,
) -> List[str]:  # type: ignore[name-defined]
    if not args.viz_dir:
        return []

    viz_dir = Path(args.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    state_stats = summarize_state_statistics(dataset)
    if state_stats.empty:
        raise ValueError("State statistics table is empty; cannot create visualizations.")

    monthly_state_stats = None
    if "month_start" in dataset.columns:
        try:
            monthly_state_stats = summarize_state_statistics(dataset, year_col="month_start")
        except Exception as exc:
            LOGGER.warning("Unable to build monthly state statistics: %s", exc)

    metric = args.choropleth_metric
    if metric not in state_stats.columns:
        available = [col for col in state_stats.columns if col not in {"year", "state_abbrev"}]
        raise ValueError(
            f"Metric '{metric}' not available for visualization. Available options: {available}"
        )

    year_label = args.viz_year if args.viz_year is not None else "latest"
    slug = f"{metric}_{year_label}"

    preferred_geo = getattr(args, "choropleth_geo", "state")
    county_stats = None
    if raw_dataset is not None:
        try:
            county_stats = _build_county_visualization_table(dataset, raw_dataset)
            if metric not in county_stats.columns:
                LOGGER.warning(
                    "Metric '%s' unavailable for county-level visualization. Skipping county map.", metric
                )
                county_stats = None
        except Exception as exc:
            LOGGER.warning("Unable to prepare county-level visualization data: %s", exc)

    map_jobs: List[tuple[str, "pd.DataFrame"]] = [("state", state_stats)]
    if county_stats is not None:
        map_jobs.append(("county", county_stats))
    map_jobs.sort(key=lambda job: 0 if job[0] == preferred_geo else 1)

    map_paths: List[str] = []

    for geo_level, data_frame in map_jobs:
        map_path = viz_dir / f"us_{geo_level}_{slug}_map.html"
        if geo_level == "county":
            plot_us_county_choropleth(
                data_frame,
                metric=metric,
                year=args.viz_year,
                output_path=map_path,
                state_col=DEFAULT_STATE_COL,
                county_col=DEFAULT_COUNTY_COL,
                fips_col="fips",
            )
        else:
            plot_us_state_choropleth(state_stats, metric=metric, year=args.viz_year, output_path=map_path)
        LOGGER.info("Saved US %s choropleth visualization to %s", geo_level, map_path)
        try:
            webbrowser.open(map_path.resolve().as_uri())
            LOGGER.info("Opened US %s choropleth visualization in default browser", geo_level)
        except Exception as exc:
            LOGGER.warning("Unable to automatically open %s choropleth visualization: %s", geo_level, exc)
        map_paths.append(str(map_path))

    bar_path = viz_dir / f"top_states_{slug}.png"
    plot_top_states_bar_chart(
        state_stats,
        metric=metric,
        year=args.viz_year,
        top_n=args.top_n_states,
        output_path=bar_path,
    )
    LOGGER.info("Saved Matplotlib bar chart to %s", bar_path)

    daily_metric_df = None
    if raw_dataset is not None:
        try:
            import pandas as pd

            missing = {DEFAULT_DATE_COL, DEFAULT_AQI_VALUE_COL} - set(raw_dataset.columns)
            if missing:
                LOGGER.warning(
                    "Raw dataset missing %s; falling back to yearly trend data for time series plot",
                    ", ".join(sorted(missing)),
                )
            else:
                daily_subset = raw_dataset[[DEFAULT_DATE_COL, DEFAULT_AQI_VALUE_COL]].copy()
                daily_subset[DEFAULT_DATE_COL] = pd.to_datetime(daily_subset[DEFAULT_DATE_COL])
                daily_subset["date"] = daily_subset[DEFAULT_DATE_COL].dt.normalize()
                daily_metric_df = (
                    daily_subset.groupby("date")[DEFAULT_AQI_VALUE_COL]
                    .agg(["mean", "median", "max", "min", "count"])
                    .rename(
                        columns={
                            "mean": "avg_aqi",
                            "median": "median_aqi",
                            "max": "max_aqi",
                            "min": "min_aqi",
                            "count": "aqi_observations",
                        }
                    )
                    .reset_index()
                    .sort_values("date")
                )
        except Exception as exc:  # pragma: no cover - visualization helper
            LOGGER.warning("Unable to compute daily AQI trend data, using yearly aggregates: %s", exc)

    trend_source = state_stats
    time_column = "year"
    if daily_metric_df is not None and metric in daily_metric_df.columns:
        trend_source = daily_metric_df
        time_column = "date"

    trend_path = viz_dir / f"{metric}_trend.png"
    plot_yearly_metric_trend(trend_source, metric=metric, output_path=trend_path, time_col=time_column)
    LOGGER.info(
        "Saved %s trend line chart to %s",
        "daily" if time_column == "date" else "yearly",
        trend_path,
    )

    forecast_artifacts: List[str] = []
    forecast_offset = getattr(args, "forecast_year_offset", 0)
    if forecast_offset:
        base_year = int(state_stats["year"].max()) if args.viz_year is None else args.viz_year
        forecast_year = base_year + forecast_offset
        forecast_source = monthly_state_stats if monthly_state_stats is not None else state_stats
        time_col = "month_start" if monthly_state_stats is not None else "year"
        target_time_value = None
        if time_col == "month_start":
            latest_period = pd.to_datetime(forecast_source["month_start"]).max()
            if pd.isna(latest_period):
                LOGGER.warning("Unable to determine latest month for projections; falling back to yearly trend.")
                forecast_source = state_stats
                time_col = "year"
            else:
                target_time_value = latest_period + pd.DateOffset(years=forecast_offset)
        LOGGER.info(
            "Building %d-year ahead %s projection using %s-granularity history (target year %d)",
            forecast_offset,
            metric,
            "monthly" if time_col == "month_start" else "yearly",
            forecast_year,
        )
        try:
            forecast_stats = project_metric_trend_by_state(
                forecast_source,
                metric=metric,
                target_year=forecast_year,
                time_col=time_col,
                target_time_value=target_time_value,
            )
        except Exception as exc:
            LOGGER.warning("Unable to build forecast for %s: %s", metric, exc)
        else:
            forecast_slug = f"{metric}_{forecast_year}_forecast"
            forecast_csv = viz_dir / f"{forecast_slug}.csv"
            forecast_stats.to_csv(forecast_csv, index=False)
            forecast_artifacts.append(str(forecast_csv))
            LOGGER.info("Saved forecast table to %s", forecast_csv)

            forecast_map_path = viz_dir / f"us_state_{forecast_slug}_map.html"
            plot_us_state_choropleth(
                forecast_stats,
                metric=metric,
                year=forecast_year,
                output_path=forecast_map_path,
            )
            forecast_artifacts.append(str(forecast_map_path))
            LOGGER.info("Saved forecast US state choropleth visualization to %s", forecast_map_path)
            try:
                webbrowser.open(forecast_map_path.resolve().as_uri())
                LOGGER.info("Opened forecast choropleth visualization in default browser")
            except Exception as exc:
                LOGGER.warning("Unable to automatically open forecast choropleth: %s", exc)

            forecast_bar_path = viz_dir / f"top_states_{forecast_slug}.png"
            plot_top_states_bar_chart(
                forecast_stats,
                metric=metric,
                year=forecast_year,
                top_n=args.top_n_states,
                output_path=forecast_bar_path,
            )
            forecast_artifacts.append(str(forecast_bar_path))
            LOGGER.info("Saved forecast bar chart to %s", forecast_bar_path)

    return map_paths + [str(bar_path), str(trend_path)] + forecast_artifacts


def summarize_results(model_names: Iterable[str], results: dict) -> None:
    for name in model_names:
        result = results[name]
        projected_mean = float(result.predictions.mean())
        LOGGER.info(
            "%s -> R2 %.3f | RMSE %.3f | MAE %.3f | Mean AQI %.2f",
            name,
            result.metrics["r2"],
            result.metrics["rmse"],
            result.metrics["mae"],
            projected_mean,
        )


def main(argv: Iterable[str] | None = None) -> List[str]:
    args = parse_args(argv)

    dataset, raw_dataset = load_and_prepare_dataset(args.aqi_csv)
    trainer = AQIModelTrainer(
        dataset,
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    artifact_paths: List[str] = []
    results = trainer.run()
    summarize_results(results.keys(), results)
    if args.persist_dir:
        trainer.persist_results(results, args.persist_dir)
        artifact_paths.append(str(Path(args.persist_dir)))

    if args.viz_dir:
        artifact_paths.extend(create_visualizations(dataset, args, raw_dataset=raw_dataset))

    return artifact_paths


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
