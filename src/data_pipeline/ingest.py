"""Read-only ingestion utilities for the four official attachments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_contract(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    with (project_root / "config" / "data_contract.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_paths(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    contract = load_contract(project_root)
    return {
        key: project_root / relative_path
        for key, relative_path in contract["required_inputs"].items()
    }


def read_attachments(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load source workbooks without modifying them.

    Sheet position is used only to select the two semantically fixed sheets in
    attachment 2; time alignment is never inferred from Excel column letters.
    """
    paths = input_paths(project_root)
    return {
        "attachment_1": pd.read_excel(paths["attachment_1"], sheet_name=0),
        "attachment_2_load": pd.read_excel(paths["attachment_2"], sheet_name=0),
        "attachment_2_pv": pd.read_excel(paths["attachment_2"], sheet_name=1),
        "attachment_3": pd.read_excel(paths["attachment_3"], sheet_name=0),
        "attachment_4": pd.read_excel(paths["attachment_4"], sheet_name=0),
    }


def build_inventory(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for attachment, path in input_paths(project_root).items():
        digest = sha256_file(path)
        with pd.ExcelFile(path, engine="openpyxl") as workbook:
            for sheet in workbook.sheet_names:
                frame = pd.read_excel(workbook, sheet_name=sheet)
                records.append(
                    {
                        "attachment": attachment,
                        "file": path.relative_to(project_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": digest,
                        "sheet": sheet,
                        "data_rows": int(frame.shape[0]),
                        "data_columns": int(frame.shape[1]),
                    }
                )
    return pd.DataFrame.from_records(records)
