"""Run the question 2 causal load forecast comparison pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data_pipeline.ingest import PROJECT_ROOT, load_contract, read_attachments
from src.forecasting.load_day_ahead import (
    GRID_H,
    GRID_K,
    GRID_TAU,
    KERNEL_H,
    KERNEL_K,
    KERNEL_TAU,
    build_day_type_frame,
    build_load_day_ahead,
    verify_kernel_grid,
)


def run_load_pipeline(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    project_root = Path(project_root).resolve()
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "question2" / "analysis" / "load_forecast"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = load_contract(project_root)
    raw = read_attachments(project_root)
    dispatch = pd.read_parquet(processed_dir / "dispatch_10min.parquet")
    representative_load = pd.to_numeric(
        raw["attachment_1"].iloc[:, 2], errors="raise"
    ).to_numpy(float)
    baseline = pd.read_parquet(processed_dir / "day_ahead_baseline_10min.parquet")

    result = build_load_day_ahead(
        dispatch,
        representative_load,
        baseline=baseline,
        k=KERNEL_K,
        h=KERNEL_H,
        tau=KERNEL_TAU,
        key_dates=contract["key_dates"],
    )

    paths = {
        "ten_minute": processed_dir / "load_day_ahead_10min.parquet",
        "metrics": output_dir / "model_metrics.csv",
        "key_dates": output_dir / "key_dates_metrics.csv",
        "bound_stats": output_dir / "bound_coverage.csv",
        "quality_checks": output_dir / "quality_checks.csv",
        "summary": output_dir / "load_forecast_summary.json",
        "grid": output_dir / "kernel_grid_verification.csv",
    }
    result.ten_minute.to_parquet(paths["ten_minute"], index=False)
    result.metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    result.key_dates.to_csv(paths["key_dates"], index=False, encoding="utf-8-sig")
    result.bound_stats.to_csv(paths["bound_stats"], index=False, encoding="utf-8-sig")
    result.quality_checks.to_csv(paths["quality_checks"], index=False, encoding="utf-8-sig")

    dates = pd.DatetimeIndex(dispatch["plan_date"].drop_duplicates()).sort_values()
    day_types = build_day_type_frame(dates)["day_type"].to_numpy()
    load_matrix = (
        dispatch.sort_values(["plan_date", "slot_index"])
        .pivot(index="plan_date", columns="slot_index", values="load_actual_kw")
        .reindex(index=dates, columns=range(1, 145))
        .to_numpy(float)
    )
    grid = verify_kernel_grid(load_matrix, day_types, dates)
    grid.to_csv(paths["grid"], index=False, encoding="utf-8-sig")
    best = grid.iloc[0]
    summary = {
        **result.summary,
        "grid_verification": {
            "combinations": len(grid),
            "k_set": list(GRID_K),
            "h_set": list(GRID_H),
            "tau_set": list(GRID_TAU),
            "best": {
                "k": int(best["k"]),
                "h": float(best["h"]),
                "tau": float(best["tau"]),
                "mape_pct": float(best["mape_pct"]),
            },
            "docx_optimum_k10_h2_tau14_mape_pct": float(
                grid[
                    (grid["k"] == 10) & (grid["h"] == 2.0) & (grid["tau"] == 14)
                ]["mape_pct"].iloc[0]
            ),
            "docx_reference_mape_pct": 2.43,
        },
        "artifacts": {
            key: str(path.relative_to(project_root))
            for key, path in paths.items()
            if key != "summary"
        },
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result.quality_checks["status"].eq("FAIL").any():
        failed = result.quality_checks.loc[
            result.quality_checks["status"].eq("FAIL"), "check_id"
        ].tolist()
        raise RuntimeError(f"Load day-ahead quality checks failed: {failed}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal question 2 load forecasts")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    paths = run_load_pipeline(args.project_root)
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
