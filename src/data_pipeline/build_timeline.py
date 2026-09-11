"""Construction of the official 144-slot plan-date time axis."""

from __future__ import annotations

from datetime import time
from typing import Any, Iterable

import pandas as pd


def parse_time_label(value: Any) -> tuple[int, bool]:
    """Return minutes after midnight and whether the label is next-day 00:00."""
    if isinstance(value, time):
        return value.hour * 60 + value.minute, False
    text = str(value).strip()
    if text in {"0:00+1", "00:00+1", "0:00 +1", "00:00 +1"}:
        return 0, True
    parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="raise")
    return int(parsed.hour * 60 + parsed.minute), False


def validate_source_headers(headers: Iterable[Any]) -> None:
    values = list(headers)
    if len(values) != 144:
        raise ValueError(f"Expected 144 time headers, received {len(values)}")
    parsed = [parse_time_label(value) for value in values]
    expected = [(10 * idx) % (24 * 60) for idx in range(1, 145)]
    observed = [minute for minute, _ in parsed]
    if observed != expected:
        raise ValueError("Source time headers do not follow the official 10-minute order")
    if not parsed[-1][1]:
        raise ValueError("The final source time header must be the next-day 0:00 label")


def build_timeline(plan_dates: Iterable[Any], headers: Iterable[Any]) -> pd.DataFrame:
    validate_source_headers(headers)
    dates = pd.DatetimeIndex(pd.to_datetime(list(plan_dates))).normalize()
    if dates.has_duplicates:
        raise ValueError("Plan dates contain duplicates")

    base = pd.DataFrame(
        {
            "plan_date": dates.repeat(144),
            "slot_index": list(range(1, 145)) * len(dates),
        }
    )
    base["interval_start"] = base["plan_date"] + pd.to_timedelta(
        base["slot_index"] * 10, unit="m"
    )
    base["interval_end"] = base["interval_start"] + pd.Timedelta(minutes=10)
    base["calendar_date"] = base["interval_start"].dt.normalize()
    base["is_cross_day"] = base["slot_index"].eq(144)
    base["is_result_period"] = base["plan_date"].between(
        pd.Timestamp("2025-02-01"), pd.Timestamp("2025-12-31")
    )
    return base
