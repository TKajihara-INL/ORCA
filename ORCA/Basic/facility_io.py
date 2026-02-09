from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _coerce_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _normalize_keys(mapping: Mapping) -> dict:
    return {str(k).strip().lower(): v for k, v in mapping.items()}


def _get_value(mapping: Mapping, keys: Sequence[str], label: str) -> float:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return float(mapping[key])
    raise KeyError(f"Missing {label} in facility data (tried keys: {keys})")


def read_json_payload(path: str | Path) -> Mapping:
    """Read a JSON payload from disk."""
    path = _coerce_path(path)
    data = json.loads(path.read_text())
    if isinstance(data, list):
        if not data:
            raise ValueError(f"No entries found in {path}")
        data = data[-1]
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected object in {path}, got {type(data).__name__}")
    return data


def read_csv_payload(path: str | Path) -> Mapping:
    """Read the last row of a CSV file as a mapping."""
    path = _coerce_path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[-1]


def read_payload(path: str | Path) -> Mapping:
    """Read a JSON or CSV payload from disk."""
    path = _coerce_path(path)
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return read_json_payload(path)
    if suffix == ".csv":
        return read_csv_payload(path)
    raise ValueError(f"Unsupported state file type: {path.suffix}")


def _normalize_field_keys(
    fields: Mapping[str, Sequence[str]] | Iterable[str],
) -> Mapping[str, Sequence[str]]:
    if isinstance(fields, Mapping):
        return fields
    return {name: (name,) for name in fields}


def read_measurements(
    path: str | Path,
    fields: Mapping[str, Sequence[str]] | Iterable[str],
) -> dict[str, float]:
    """Read specific measurements from a JSON/CSV payload.

    `fields` can be a dict mapping output names to candidate keys, or a list
    of field names to read directly.
    """
    payload = _normalize_keys(read_payload(path))
    field_map = _normalize_field_keys(fields)
    return {name: _get_value(payload, keys, name) for name, keys in field_map.items()}


def write_json_setpoints(path: str | Path, values: Mapping[str, float]) -> None:
    """Write setpoints to a JSON file."""
    path = _coerce_path(path)
    payload = {key: float(value) for key, value in values.items()}
    path.write_text(json.dumps(payload, indent=2))


def write_csv_setpoints(path: str | Path, values: Mapping[str, float]) -> None:
    """Append setpoints to a CSV file."""
    path = _coerce_path(path)
    values = {key: float(value) for key, value in values.items()}
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(values)


def send_setpoints(path: str | Path, values: Mapping[str, float]) -> None:
    """Send commands by writing setpoints to a JSON or CSV file."""
    path = _coerce_path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        write_json_setpoints(path, values)
        return
    if suffix == ".csv":
        write_csv_setpoints(path, values)
        return
    raise ValueError(f"Unsupported setpoint file type: {path.suffix}")
