from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from data_agent_baseline.agents.understanding.tools.normalization import (
    normalize_identifier,
)
from data_agent_baseline.tools.structured_context import StructuredContextStore, StructuredTable


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def normalize_value(text: Any) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def value_tokens(text: Any) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(str(text))}


def coerce_scalar(value: Any) -> str | int | float:
    text = str(value).strip()
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?\d+\.\d+", text):
        return float(text)
    return text


@dataclass(frozen=True, slots=True)
class FieldValueMatch:
    score: float
    candidate_value: str | int | float | dict[str, Any] | None
    evidence: list[str]
    match_type: str


@dataclass(frozen=True, slots=True)
class FieldProfile:
    field: str
    path: str
    table: str | None
    name: str
    kind: str
    data_type: str
    sample_values: list[str]
    non_null_count: int
    null_count: int
    min_value: str | int | float | None
    max_value: str | int | float | None
    is_id_like: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "path": self.path,
            "table": self.table,
            "name": self.name,
            "kind": self.kind,
            "data_type": self.data_type,
            "sample_values": list(self.sample_values),
            "non_null_count": self.non_null_count,
            "null_count": self.null_count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "is_id_like": self.is_id_like,
        }


def numeric_profile_bounds(profile: FieldProfile) -> tuple[float | None, float | None]:
    lower = parse_number(profile.min_value) if profile.min_value is not None else None
    upper = parse_number(profile.max_value) if profile.max_value is not None else None
    return (
        float(lower) if lower is not None else None,
        float(upper) if upper is not None else None,
    )


def numeric_threshold_feasibility(
    *,
    operator: str,
    threshold: float,
    profile: FieldProfile,
) -> tuple[bool, float, list[str]]:
    lower, upper = numeric_profile_bounds(profile)
    if lower is None or upper is None:
        return True, 1.0, []

    if operator == ">" and upper <= threshold:
        return False, 0.0, [
            f"numeric threshold `{operator} {threshold:g}` is outside observed field range {lower:g}..{upper:g}"
        ]
    if operator == ">=" and upper < threshold:
        return False, 0.0, [
            f"numeric threshold `{operator} {threshold:g}` is outside observed field range {lower:g}..{upper:g}"
        ]
    if operator == "<" and lower >= threshold:
        return False, 0.0, [
            f"numeric threshold `{operator} {threshold:g}` is outside observed field range {lower:g}..{upper:g}"
        ]
    if operator == "<=" and lower > threshold:
        return False, 0.0, [
            f"numeric threshold `{operator} {threshold:g}` is outside observed field range {lower:g}..{upper:g}"
        ]

    if operator in {"<", "<="} and upper <= threshold:
        return True, 0.7, [
            f"numeric threshold `{operator} {threshold:g}` is above observed field maximum {upper:g}"
        ]
    if operator in {">", ">="} and lower >= threshold:
        return True, 0.7, [
            f"numeric threshold `{operator} {threshold:g}` is below observed field minimum {lower:g}"
        ]
    return True, 1.0, [f"numeric threshold overlaps observed field range {lower:g}..{upper:g}"]


def numeric_range_feasibility(
    *,
    lower_bound: float,
    upper_bound: float,
    profile: FieldProfile,
) -> tuple[bool, float, list[str]]:
    profile_lower, profile_upper = numeric_profile_bounds(profile)
    if profile_lower is None or profile_upper is None:
        return True, 1.0, []
    if profile_upper < lower_bound or profile_lower > upper_bound:
        return False, 0.0, [
            f"range `{lower_bound:g}..{upper_bound:g}` does not overlap observed field range "
            f"{profile_lower:g}..{profile_upper:g}"
        ]
    if lower_bound <= profile_lower and profile_upper <= upper_bound:
        return True, 0.85, [
            f"range `{lower_bound:g}..{upper_bound:g}` fully covers observed field range "
            f"{profile_lower:g}..{profile_upper:g}"
        ]
    return True, 1.0, [
        f"range `{lower_bound:g}..{upper_bound:g}` overlaps observed field range "
        f"{profile_lower:g}..{profile_upper:g}"
    ]


@dataclass(frozen=True, slots=True)
class TableSchema:
    path: str
    table: str
    kind: str
    columns: list[str]
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "table": self.table,
            "kind": self.kind,
            "columns": list(self.columns),
            "row_count": self.row_count,
        }


class CandidateStore:
    """Task-local hybrid candidate store for schema fields and values."""

    def __init__(
        self,
        *,
        tables: list[TableSchema] | None = None,
        fields: list[FieldProfile],
        structured_store: StructuredContextStore | None = None,
    ) -> None:
        self.tables = tables or []
        self.fields = fields
        self.structured_store = structured_store
        self._by_field = {normalize_identifier(item.field): item for item in fields}

    def to_dict(self) -> dict[str, Any]:
        return {
            "tables": [item.to_dict() for item in self.tables],
            "field_count": len(self.fields),
            "fields": [item.to_dict() for item in self.fields],
        }

    def field_profile(self, field: str | None) -> FieldProfile | None:
        if not field:
            return None
        return self._by_field.get(normalize_identifier(field))

    def match_filter_value(
        self,
        *,
        raw_value: str,
        operator: str,
        profile: FieldProfile,
        use_sample_values: bool = True,
    ) -> FieldValueMatch | None:
        parsed_range = parse_range_value(raw_value)
        parsed_number = parse_number(raw_value)
        normalized_raw = normalize_value(raw_value)
        raw_tokens = value_tokens(raw_value)
        is_year_value = looks_like_year(raw_value)

        if is_year_value:
            year_match = match_year_filter_value(raw_value, profile)
            if year_match is not None:
                return year_match

        if parsed_range is not None:
            if profile.data_type in {"number", "integer", "date"}:
                evidence = [
                    f"range value `{raw_value}` is compatible with {profile.data_type} field"
                ]
                score_multiplier = 1.0
                if profile.data_type in {"number", "integer"}:
                    feasible, score_multiplier, feasibility_evidence = numeric_range_feasibility(
                        lower_bound=float(parsed_range["lower"]),
                        upper_bound=float(parsed_range["upper"]),
                        profile=profile,
                    )
                    if not feasible:
                        return None
                    evidence.extend(feasibility_evidence)
                return FieldValueMatch(
                    score=round(0.62 * score_multiplier, 2),
                    candidate_value=parsed_range,
                    evidence=evidence,
                    match_type="range_type_compatible",
                )
            return None

        if operator in {">", "<", ">=", "<="} and parsed_number is not None:
            if profile.data_type in {"number", "integer"}:
                feasible, score_multiplier, feasibility_evidence = numeric_threshold_feasibility(
                    operator=operator,
                    threshold=float(parsed_number),
                    profile=profile,
                )
                if not feasible:
                    return None
                evidence = [
                    f"numeric comparison `{operator} {raw_value}` is compatible with numeric field",
                    *feasibility_evidence,
                ]
                return FieldValueMatch(
                    score=round(0.58 * score_multiplier, 2),
                    candidate_value=parsed_number,
                    evidence=evidence,
                    match_type="numeric_threshold",
                )
            if profile.data_type == "date":
                return FieldValueMatch(
                    score=0.4,
                    candidate_value=str(raw_value).strip(),
                    evidence=[f"comparison operator `{operator}` can apply to date-like field"],
                    match_type="date_threshold",
                )
            return None

        if not use_sample_values:
            return None

        value_pool = profile.sample_values
        for value in value_pool:
            if normalize_value(value) == normalized_raw:
                return FieldValueMatch(
                    score=0.92,
                    candidate_value=coerce_scalar(value),
                    evidence=[f"exact normalized value match `{raw_value}` in `{profile.field}`"],
                    match_type="exact_value",
                )

        for value in value_pool:
            normalized_candidate = normalize_value(value)
            allow_fuzzy_numeric_year = not (
                is_year_value and profile.data_type in {"date", "number", "integer"}
            )
            if normalized_raw and normalized_raw in normalized_candidate and allow_fuzzy_numeric_year:
                return FieldValueMatch(
                    score=0.72,
                    candidate_value=coerce_scalar(value),
                    evidence=[f"filter value `{raw_value}` is contained in sample value `{value}`"],
                    match_type="contains_value",
                )
            candidate_tokens = value_tokens(value)
            if raw_tokens and raw_tokens <= candidate_tokens and allow_fuzzy_numeric_year:
                return FieldValueMatch(
                    score=0.68,
                    candidate_value=coerce_scalar(value),
                    evidence=[f"filter value tokens are covered by sample value `{value}`"],
                    match_type="token_value",
                )
            if (
                candidate_tokens
                and candidate_tokens < raw_tokens
                and len(candidate_tokens) >= 1
                and allow_fuzzy_numeric_year
            ):
                overlap = len(candidate_tokens & raw_tokens) / max(1, len(candidate_tokens))
                if overlap >= 0.8:
                    return FieldValueMatch(
                        score=0.54,
                        candidate_value=coerce_scalar(value),
                        evidence=[
                            f"sample value `{value}` mostly matches filter phrase `{raw_value}`"
                        ],
                        match_type="partial_token_value",
                    )

        return None

class CandidateStoreBuilder:
    def __init__(
        self,
        *,
        sample_limit: int = 10,
    ) -> None:
        self.sample_limit = sample_limit

    def build(
        self,
        *,
        structured_store: StructuredContextStore,
    ) -> CandidateStore:
        tables: list[TableSchema] = []
        fields: list[FieldProfile] = []
        for table in structured_store.tables():
            table_schema, table_fields = self._collect_structured_table(structured_store, table)
            tables.append(table_schema)
            fields.extend(table_fields)

        fields.sort(key=lambda item: (item.path, item.field))
        return CandidateStore(
            tables=tables,
            fields=fields,
            structured_store=structured_store,
        )

    def _collect_structured_table(
        self,
        structured_store: StructuredContextStore,
        table: StructuredTable,
    ) -> tuple[TableSchema, list[FieldProfile]]:
        columns = [
            column
            for column in table.columns
            if column != "__row_index"
        ]
        table_schema = TableSchema(
            path=table.path,
            table=table.name,
            kind=table.kind,
            columns=[str(column) for column in columns],
            row_count=structured_store.row_count(table.name),
        )
        fields: list[FieldProfile] = []
        for column in columns:
            profile = structured_store.field_profile(f"{table.name}.{column}")
            if profile is None:
                continue
            fields.append(self._field_profile_from_payload(profile))
        return table_schema, fields

    def _field_profile_from_payload(self, payload: dict[str, Any]) -> FieldProfile:
        sample_values = [str(value) for value in payload.get("sample_values", [])]
        values_for_type = dedupe_preserve_order(
            [
                *sample_values,
                *[
                    str(value)
                    for value in (payload.get("min_value"), payload.get("max_value"))
                    if value is not None and str(value).strip()
                ],
            ]
        )
        data_type = infer_profile_field_type(
            str(payload["name"]),
            str(payload.get("data_type") or ""),
            values_for_type,
        )
        return FieldProfile(
            field=str(payload["field"]),
            path=str(payload["path"]),
            table=str(payload["table"]) if payload.get("table") else None,
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            data_type=data_type,
            sample_values=sample_values[: self.sample_limit],
            non_null_count=int(payload.get("non_null_count") or 0),
            null_count=int(payload.get("null_count") or 0),
            min_value=coerce_profile_value(payload.get("min_value")),
            max_value=coerce_profile_value(payload.get("max_value")),
            is_id_like=is_id_like(str(payload["name"]), sample_values),
        )


def infer_field_type(name: str, values: list[str]) -> str:
    normalized_name = normalize_identifier(name)
    non_empty = [value for value in values if str(value).strip()]
    if not non_empty:
        return "empty"

    numeric_count = sum(parse_number(value) is not None for value in non_empty)
    integer_count = sum(parse_integer(value) is not None for value in non_empty)
    date_count = sum(parse_date(value) is not None or looks_like_year(value) for value in non_empty)
    lowered_values = {str(value).strip().lower() for value in non_empty[:100]}
    if lowered_values and lowered_values <= {"true", "false", "yes", "no", "y", "n", "0", "1"}:
        return "boolean"
    if numeric_count / len(non_empty) >= 0.85:
        return "integer" if integer_count / len(non_empty) >= 0.95 else "number"
    if date_count / len(non_empty) >= 0.65 or any(token in normalized_name for token in {"date", "year", "time"}):
        return "date"
    if len({normalize_value(value) for value in non_empty}) <= min(25, max(4, len(non_empty) // 5)):
        return "enum"
    return "text"


def infer_profile_field_type(name: str, duckdb_data_type: str, values: list[str]) -> str:
    normalized_type = str(duckdb_data_type).upper()
    if "BOOL" in normalized_type:
        return "boolean"
    if "DATE" in normalized_type or "TIME" in normalized_type:
        return "date"
    if any(
        token in normalized_type
        for token in {
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
        }
    ):
        return "integer"
    if any(token in normalized_type for token in {"FLOAT", "DOUBLE", "DECIMAL", "REAL"}):
        return "number"

    inferred = infer_field_type(name, values)
    if inferred != "empty":
        return inferred

    normalized_name = normalize_identifier(name)
    if any(token in normalized_name for token in {"date", "year", "month", "time"}):
        return "date"
    return inferred


def is_id_like(name: str, values: list[str]) -> bool:
    normalized_name = normalize_identifier(name)
    if normalized_name in {"id", "uid", "key"} or normalized_name.endswith("_id"):
        return True
    if normalized_name.startswith(("id_", "link_to_", "fk_", "ref_")):
        return True
    non_empty = [str(value).strip() for value in values if str(value).strip()]
    if not non_empty:
        return False
    if len(non_empty) >= 5 and len(set(non_empty)) / len(non_empty) > 0.95:
        return normalized_name in {"code", "number"} or normalized_name.endswith(("code", "number"))
    return False


def parse_integer(value: Any) -> int | None:
    text = str(value).strip().replace(",", "")
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    return None


def parse_number(value: Any) -> float | int | None:
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?\d+\.\d+", text):
        return float(text)
    return None


def parse_date(value: Any) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    candidates = [text, text.replace("/", "-")]
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m",
        "%Y/%m/%d",
        "%Y/%m",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %Y",
        "%b %Y",
        "%Y",
    ]
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            pass
        for fmt in formats:
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return None


def looks_like_year(value: Any) -> bool:
    return bool(re.fullmatch(r"(?:18|19|20|21)\d{2}", str(value).strip()))


def match_year_filter_value(raw_value: Any, profile: FieldProfile) -> FieldValueMatch | None:
    if not looks_like_year(raw_value):
        return None
    try:
        year = int(str(raw_value).strip())
    except ValueError:
        return None

    values = [
        str(value).strip()
        for value in [
            *profile.sample_values[:500],
            profile.min_value,
            profile.max_value,
        ]
        if value is not None and str(value).strip()
    ]
    date_format = infer_year_filter_format(values)
    if date_format is None:
        return None
    if profile.data_type != "date" and not profile_supports_year_filter(profile):
        return None
    if date_format == "YYYYMM":
        return FieldValueMatch(
            score=0.86,
            candidate_value={
                "operator": "between",
                "granularity": "year",
                "format": "YYYYMM",
                "year": year,
                "lower": year * 100 + 1,
                "upper": year * 100 + 12,
            },
            evidence=[
                f"year `{raw_value}` maps to YYYYMM range on date-like field `{profile.field}`"
            ],
            match_type="year_range",
        )
    if date_format == "YYYY-MM":
        return FieldValueMatch(
            score=0.86,
            candidate_value={
                "operator": "between",
                "granularity": "year",
                "format": "YYYY-MM",
                "year": year,
                "lower": f"{year:04d}-01",
                "upper": f"{year:04d}-12",
            },
            evidence=[
                f"year `{raw_value}` maps to YYYY-MM range on date-like field `{profile.field}`"
            ],
            match_type="year_range",
        )
    if date_format == "YYYYMMDD":
        return FieldValueMatch(
            score=0.86,
            candidate_value={
                "operator": "between",
                "granularity": "year",
                "format": "YYYYMMDD",
                "year": year,
                "lower": year * 10000 + 101,
                "upper": year * 10000 + 1231,
            },
            evidence=[
                f"year `{raw_value}` maps to YYYYMMDD range on date-like field `{profile.field}`"
            ],
            match_type="year_range",
        )
    if date_format == "YYYY-MM-DD":
        return FieldValueMatch(
            score=0.86,
            candidate_value={
                "operator": "between",
                "granularity": "year",
                "format": "YYYY-MM-DD",
                "year": year,
                "lower": f"{year:04d}-01-01",
                "upper": f"{year:04d}-12-31",
            },
            evidence=[
                f"year `{raw_value}` maps to ISO date range on date-like field `{profile.field}`"
            ],
            match_type="year_range",
        )
    if date_format == "YYYY":
        return FieldValueMatch(
            score=0.84,
            candidate_value={
                "operator": "=",
                "granularity": "year",
                "format": "YYYY",
                "year": year,
                "value": year,
            },
            evidence=[
                f"year `{raw_value}` maps to exact year value on date-like field `{profile.field}`"
            ],
            match_type="year_exact",
        )
    return None


def profile_supports_year_filter(profile: FieldProfile) -> bool:
    identifier_text = normalize_identifier(
        " ".join(
            [
                profile.field,
                profile.table or "",
                profile.name,
            ]
        )
    )
    identifier_tokens = set(identifier_text.split("_"))
    return bool(identifier_tokens & {"date", "year", "month", "yearmonth"})


def infer_year_filter_format(values: list[str]) -> str | None:
    for value in values:
        text = str(value).strip()
        if re.fullmatch(r"(?:18|19|20|21)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])", text):
            return "YYYYMMDD"
    for value in values:
        text = str(value).strip()
        if re.fullmatch(r"(?:18|19|20|21)\d{2}(?:0[1-9]|1[0-2])", text):
            return "YYYYMM"
    for value in values:
        text = str(value).strip()
        if re.fullmatch(r"(?:18|19|20|21)\d{2}-(?:0[1-9]|1[0-2])", text):
            return "YYYY-MM"
    for value in values:
        text = str(value).strip()
        if re.fullmatch(r"(?:18|19|20|21)\d{2}-\d{2}-\d{2}(?:[ T].*)?", text):
            return "YYYY-MM-DD"
    for value in values:
        text = str(value).strip()
        if re.fullmatch(r"(?:18|19|20|21)\d{2}", text):
            return "YYYY"
    return None


def parse_range_value(raw_value: str) -> dict[str, Any] | None:
    text = str(raw_value).strip().lower()
    if re.fullmatch(r"(?:18|19|20|21)\d{2}[-/](?:0?[1-9]|1[0-2])", text):
        return None
    if re.fullmatch(
        r"(?:18|19|20|21)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])(?:[ t].*)?",
        text,
    ):
        return None
    patterns = [
        r"\bbetween\s+([-+]?\d+(?:\.\d+)?)\s+(?:and|to)\s+([-+]?\d+(?:\.\d+)?)\b",
        r"\bfrom\s+([-+]?\d+(?:\.\d+)?)\s+to\s+([-+]?\d+(?:\.\d+)?)\b",
        r"\branging\s+from\s+([-+]?\d+(?:\.\d+)?)\s+to\s+([-+]?\d+(?:\.\d+)?)\b",
        r"\b([-+]?\d+(?:\.\d+)?)\s*(?:to|-)\s*([-+]?\d+(?:\.\d+)?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        lower = parse_number(match.group(1))
        upper = parse_number(match.group(2))
        if lower is None or upper is None:
            continue
        if float(lower) > float(upper):
            lower, upper = upper, lower
        return {"operator": "between", "lower": lower, "upper": upper}
    return None


def coerce_profile_value(value: Any) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text:
        return None
    return coerce_scalar(text)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
