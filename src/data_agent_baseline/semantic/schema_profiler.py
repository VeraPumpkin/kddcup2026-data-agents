from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.semantic.models import JoinCandidate, SchemaFieldProfile

SUPPORTED_STRUCTURED_EXTENSIONS = {".csv", ".json", ".sqlite", ".db"}
NUMERIC_DTYPE_NAMES = {"int", "float", "number"}


@dataclass(frozen=True, slots=True)
class SchemaProfilingResult:
    schema_profiles: list[SchemaFieldProfile]
    join_candidates: list[JoinCandidate]
    debug: dict[str, object]
    warnings: list[str]


def _normalize_identifier(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _is_null_like(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"na", "n/a", "null", "none", "nan"}
    return False


def _as_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _infer_dtype(values: list[object]) -> str:
    non_null = [value for value in values if not _is_null_like(value)]
    if not non_null:
        return "unknown"

    def is_int_like(item: object) -> bool:
        text = _as_string(item).strip()
        return bool(re.fullmatch(r"[+-]?\d+", text))

    def is_float_like(item: object) -> bool:
        text = _as_string(item).strip()
        return bool(re.fullmatch(r"[+-]?(?:\d+\.\d+|\d+)", text))

    def is_bool_like(item: object) -> bool:
        text = _as_string(item).strip().lower()
        return text in {"0", "1", "true", "false", "yes", "no"}

    def is_date_like(item: object) -> bool:
        text = _as_string(item).strip()
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ t]\d{2}:\d{2}:\d{2})?", text))

    if all(is_int_like(item) for item in non_null):
        return "int"
    if all(is_float_like(item) for item in non_null):
        return "float"
    if all(is_bool_like(item) for item in non_null):
        return "bool"
    if all(is_date_like(item) for item in non_null):
        return "date"
    return "string"


def _numeric_range(values: list[object], dtype: str) -> tuple[float | int | None, float | int | None]:
    if dtype not in NUMERIC_DTYPE_NAMES:
        return None, None
    numeric_values: list[float] = []
    for value in values:
        if _is_null_like(value):
            continue
        try:
            numeric_values.append(float(_as_string(value)))
        except ValueError:
            return None, None
    if not numeric_values:
        return None, None
    min_value = min(numeric_values)
    max_value = max(numeric_values)
    if dtype == "int":
        return int(min_value), int(max_value)
    return min_value, max_value


def _semantic_tags(field_name: str, dtype: str, sample_values: list[str]) -> list[str]:
    name = field_name.lower()
    tags: list[str] = []
    if name == "id" or name.endswith("_id"):
        tags.append("identifier")
    if "date" in name or dtype == "date":
        tags.append("date")
    if "time" in name:
        tags.append("time")
    if dtype in NUMERIC_DTYPE_NAMES and re.search(r"count|num|amount|score|rate|ratio|pct|percent", name):
        tags.append("measure")
    if dtype in {"string", "bool"} and len({value for value in sample_values if value}) <= 12:
        tags.append("categorical")
    if re.search(r"status|type|category|label|code|flag|indicator|severity", name):
        tags.append("coded")
    return tags


def _profile_field(
    *,
    source_file: str,
    table_name: str,
    field_name: str,
    values: list[object],
) -> SchemaFieldProfile:
    dtype = _infer_dtype(values)
    non_null_strings = [_as_string(value) for value in values if not _is_null_like(value)]
    sample_values = non_null_strings[:8]
    total = len(values) if values else 1
    null_count = sum(1 for value in values if _is_null_like(value))
    distinct = len(set(non_null_strings))
    min_value, max_value = _numeric_range(values, dtype)
    return SchemaFieldProfile(
        source_file=source_file,
        table_name=table_name,
        field_name=field_name,
        dtype=dtype,
        sample_values=sample_values,
        min_value=min_value,
        max_value=max_value,
        unique_ratio=round(distinct / max(total - null_count, 1), 4),
        null_ratio=round(null_count / total, 4),
        semantic_tags=_semantic_tags(field_name, dtype, sample_values),
    )


def _read_csv_rows(path: Path, *, max_rows: int) -> tuple[str, list[str], list[dict[str, object]]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append(dict(row))
            if len(rows) >= max_rows:
                break
    return path.stem, list(reader.fieldnames or []), rows


def _read_json_tables(path: Path, *, max_rows: int) -> list[tuple[str, list[dict[str, object]]]]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    tables: list[tuple[str, list[dict[str, object]]]] = []
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload[: min(len(payload), max_rows)]):
        tables.append((path.stem, [dict(item) for item in payload[:max_rows]]))
        return tables
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value[: min(len(value), max_rows)]):
                tables.append((f"{path.stem}.{key}", [dict(item) for item in value[:max_rows]]))
    return tables


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _read_sqlite_tables(path: Path, *, max_rows: int) -> list[tuple[str, list[dict[str, object]], dict[str, str]]]:
    with _connect_read_only(path) as conn:
        table_rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        tables: list[tuple[str, list[dict[str, object]], dict[str, str]]] = []
        for (table_name,) in table_rows:
            pragma_rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            declared_types = {str(row[1]): str(row[2] or "") for row in pragma_rows}
            columns = [str(row[1]) for row in pragma_rows]
            sample_query = f"SELECT * FROM '{table_name}' LIMIT {int(max_rows)}"
            sample_rows = conn.execute(sample_query).fetchall()
            row_dicts = [dict(zip(columns, row, strict=False)) for row in sample_rows]
            tables.append((str(table_name), row_dicts, declared_types))
    return tables


def _profile_rows(
    source_file: str,
    table_name: str,
    rows: list[dict[str, object]],
    *,
    known_fields: list[str] | None = None,
) -> list[SchemaFieldProfile]:
    values_by_field: dict[str, list[object]] = defaultdict(list)
    for field_name in known_fields or []:
        values_by_field[str(field_name)] = []
    for row in rows:
        for field_name, value in row.items():
            values_by_field[str(field_name)].append(value)
    if not values_by_field:
        return []
    profiles = [
        _profile_field(
            source_file=source_file,
            table_name=table_name,
            field_name=field_name,
            values=values,
        )
        for field_name, values in sorted(values_by_field.items())
    ]
    return profiles


def _value_overlap_score(left_values: list[str], right_values: list[str]) -> float:
    left = {value for value in left_values if value}
    right = {value for value in right_values if value}
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return overlap / max(min(len(left), len(right)), 1)


def _is_identifier_name(field_name: str) -> bool:
    normalized = _normalize_identifier(field_name)
    return normalized == "id" or normalized.endswith("_id")


def _table_reference_score(referenced_table: str, candidate_foreign_key_field: str) -> float:
    normalized_table = _normalize_identifier(referenced_table).rstrip("s")
    normalized_field = _normalize_identifier(candidate_foreign_key_field)
    if not normalized_table or not normalized_field:
        return 0.0
    if normalized_field == f"{normalized_table}_id":
        return 1.0
    if normalized_table in normalized_field:
        return 0.5
    return 0.0


def _build_join_candidates(schema_profiles: list[SchemaFieldProfile]) -> tuple[list[JoinCandidate], list[dict[str, object]]]:
    profiles_by_table: dict[str, list[SchemaFieldProfile]] = defaultdict(list)
    for profile in schema_profiles:
        profiles_by_table[profile.table_name].append(profile)

    candidates: list[JoinCandidate] = []
    debug_rows: list[dict[str, object]] = []
    table_names = sorted(profiles_by_table)
    for index, left_table in enumerate(table_names):
        for right_table in table_names[index + 1 :]:
            for left_profile in profiles_by_table[left_table]:
                for right_profile in profiles_by_table[right_table]:
                    reasons: list[str] = []
                    score = 0.0

                    if _normalize_identifier(left_profile.field_name) == _normalize_identifier(right_profile.field_name):
                        score += 0.55
                        reasons.append("same field name")

                    if _is_identifier_name(left_profile.field_name) and _is_identifier_name(right_profile.field_name):
                        score += 0.15
                        reasons.append("identifier-like pair")

                    left_ref = _table_reference_score(right_table, left_profile.field_name)
                    right_primary = _normalize_identifier(right_profile.field_name) == "id"
                    right_named_key = _normalize_identifier(right_profile.field_name) == f"{_normalize_identifier(right_table).rstrip('s')}_id"
                    if left_ref > 0 and (right_primary or right_named_key):
                        score += 0.2 * left_ref
                        reasons.append(f"{left_profile.field_name} references {right_table}")

                    right_ref = _table_reference_score(left_table, right_profile.field_name)
                    left_primary = _normalize_identifier(left_profile.field_name) == "id"
                    left_named_key = _normalize_identifier(left_profile.field_name) == f"{_normalize_identifier(left_table).rstrip('s')}_id"
                    if right_ref > 0 and (left_primary or left_named_key):
                        score += 0.2 * right_ref
                        reasons.append(f"{right_profile.field_name} references {left_table}")

                    overlap_score = _value_overlap_score(left_profile.sample_values, right_profile.sample_values)
                    if overlap_score > 0:
                        score += min(overlap_score, 1.0) * 0.35
                        reasons.append(f"sample overlap={overlap_score:.2f}")

                    if left_profile.dtype == right_profile.dtype and left_profile.dtype != "unknown":
                        score += 0.05
                        reasons.append("dtype compatible")

                    if score >= 0.35:
                        rounded_score = round(min(score, 0.99), 4)
                        candidates.append(
                            JoinCandidate(
                                left_table=left_table,
                                right_table=right_table,
                                left_field=left_profile.field_name,
                                right_field=right_profile.field_name,
                                score=rounded_score,
                                reason="; ".join(reasons),
                            )
                        )
                    debug_rows.append(
                        {
                            "left_table": left_table,
                            "right_table": right_table,
                            "left_field": left_profile.field_name,
                            "right_field": right_profile.field_name,
                            "score": round(score, 4),
                            "reasons": reasons,
                        }
                    )
    candidates.sort(key=lambda item: (-item.score, item.left_table, item.right_table, item.left_field))
    return candidates[:50], debug_rows


def profile_structured_sources(context_dir: Path, *, max_rows_per_source: int = 200) -> SchemaProfilingResult:
    warnings: list[str] = []
    debug_sources: list[dict[str, object]] = []
    schema_profiles: list[SchemaFieldProfile] = []

    structured_paths = [
        path
        for path in sorted(context_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_STRUCTURED_EXTENSIONS
    ]

    for path in structured_paths:
        relative_path = path.relative_to(context_dir).as_posix()
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                table_name, header, rows = _read_csv_rows(path, max_rows=max_rows_per_source)
                profiles = _profile_rows(relative_path, table_name, rows, known_fields=header)
                schema_profiles.extend(profiles)
                debug_sources.append(
                    {
                        "source_file": relative_path,
                        "kind": "csv",
                        "table_count": 1,
                        "sample_row_count": len(rows),
                        "field_count": len(profiles),
                    }
                )
            elif suffix == ".json":
                tables = _read_json_tables(path, max_rows=max_rows_per_source)
                total_fields = 0
                for table_name, rows in tables:
                    profiles = _profile_rows(relative_path, table_name, rows)
                    total_fields += len(profiles)
                    schema_profiles.extend(profiles)
                if not tables:
                    warnings.append(f"Skipped JSON source without list-of-object structure: {relative_path}")
                debug_sources.append(
                    {
                        "source_file": relative_path,
                        "kind": "json",
                        "table_count": len(tables),
                        "field_count": total_fields,
                    }
                )
            else:
                tables = _read_sqlite_tables(path, max_rows=max_rows_per_source)
                total_fields = 0
                for table_name, rows, declared_types in tables:
                    profiles = _profile_rows(
                        relative_path,
                        table_name,
                        rows,
                        known_fields=list(declared_types),
                    )
                    total_fields += len(profiles)
                    schema_profiles.extend(profiles)
                debug_sources.append(
                    {
                        "source_file": relative_path,
                        "kind": "sqlite",
                        "table_count": len(tables),
                        "field_count": total_fields,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed profiling {relative_path}: {exc}")

    join_candidates, join_debug = _build_join_candidates(schema_profiles)
    debug = {
        "sources": debug_sources,
        "structured_source_count": len(structured_paths),
        "join_candidate_scoring": join_debug,
    }
    return SchemaProfilingResult(
        schema_profiles=schema_profiles,
        join_candidates=join_candidates,
        debug=debug,
        warnings=warnings,
    )
