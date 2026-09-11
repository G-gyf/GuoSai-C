"""Run the question 2 causal PV day-ahead forecasting pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data_pipeline.build_actuals import build_dispatch_10min
from src.data_pipeline.ingest import PROJECT_ROOT, load_contract, read_attachments
from src.forecasting.pv_day_ahead import (
    PVDayAheadConfig,
    build_pv_day_ahead_forecasts,
    validate_pv_day_ahead,
)


def run_pv_forecast_pipeline(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    project_root = Path(project_root).resolve()
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "question2_forecast"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = load_contract(project_root)
    raw = read_attachments(project_root)
    dispatch_path = processed_dir / "dispatch_10min.parquet"
    dispatch = pd.read_parquet(dispatch_path) if dispatch_path.exists() else build_dispatch_10min(raw)
    config = PVDayAheadConfig.from_contract(contract)
    result = build_pv_day_ahead_forecasts(
        dispatch,
        raw["attachment_1"],
        config=config,
        key_dates=contract["key_dates"],
    )
    expected = contract["expected_rows"]
    checks = validate_pv_day_ahead(
        result,
        expected_hourly_rows=expected["pv_day_ahead_hourly"],
        expected_ten_minute_rows=expected["pv_day_ahead_10min"],
        expected_result_rows=expected["result_period_intervals"],
    )

    paths = {
        "hourly": processed_dir / "pv_day_ahead_hourly.parquet",
        "ten_minute": processed_dir / "pv_day_ahead_10min.parquet",
        "residuals": processed_dir / "pv_day_ahead_residuals.parquet",
        "metrics": output_dir / "model_metrics.csv",
        "key_dates": output_dir / "key_dates_metrics.csv",
        "quality_checks": output_dir / "quality_checks.csv",
        "summary": output_dir / "forecast_summary.json",
    }
    result.hourly.to_parquet(paths["hourly"], index=False)
    result.ten_minute.to_parquet(paths["ten_minute"], index=False)
    result.residuals.to_parquet(paths["residuals"], index=False)
    result.metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    result.key_dates.to_csv(paths["key_dates"], index=False, encoding="utf-8-sig")
    checks.to_csv(paths["quality_checks"], index=False, encoding="utf-8-sig")
    summary = {
        **result.summary,
        "quality_status": "PASS" if not checks["status"].eq("FAIL").any() else "FAIL",
        "failed_checks": checks.loc[checks["status"].eq("FAIL"), "check_id"].tolist(),
        "artifacts": {key: str(path.relative_to(project_root)) for key, path in paths.items() if key != "summary"},
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if checks["status"].eq("FAIL").any():
        failed = checks.loc[checks["status"].eq("FAIL"), "check_id"].tolist()
        raise RuntimeError(f"PV day-ahead quality checks failed: {failed}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal question 2 PV forecasts")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    paths = run_pv_forecast_pipeline(args.project_root)
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
