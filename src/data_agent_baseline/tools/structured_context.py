from __future__ import annotations

import math
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from data_agent_baseline.benchmark.schema import PublicTask


_READ_ONLY_SQL_PREFIXES = ("select", "with", "pragma", "describe", "show")
_DOC_ANNOTATION_TABLES = {"doc_paragraphs", "doc_facts", "doc_relations"}
JSON_MAXIMUM_OBJECT_SIZE = 256 * 1024 * 1024
TIME_VALUE_SAMPLE_LIMIT = 200
_TIME_IDENTIFIER_TOKENS = {
    "date",
    "datetime",
    "month",
    "period",
    "time",
    "timestamp",
    "year",
    "yearmonth",
}
_MONTH_NUMBERS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_DOC_PARAGRAPH_COLUMNS = [
    ("paragraph_id", "TEXT"),
]

_DOC_FACT_COLUMNS = [
    ("paragraph_id", "TEXT"),
    ("record_anchor_name", "TEXT"),
    ("record_anchor_type", "TEXT"),
    ("entity_name_raw", "TEXT"),
    ("entity_value_raw", "TEXT"),
    ("value_type", "TEXT"),
    ("unit", "TEXT"),
    ("status", "TEXT"),
]

_DOC_RELATION_COLUMNS = [
    ("paragraph_id", "TEXT"),
    ("subject_name", "TEXT"),
    ("relation_type", "TEXT"),
    ("object_name_or_value", "TEXT"),
]


@dataclass(frozen=True, slots=True)
class StructuredTable:
    name: str
    path: str
    kind: str
    columns: list[str]
    row_count: int
    preview: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "columns": list(self.columns),
            "row_count": self.row_count,
            "preview": dict(self.preview) if isinstance(self.preview, dict) else self.preview,
        }


@dataclass(frozen=True, slots=True)
class StructuredField:
    field: str
    table: str
    path: str
    kind: str
    name: str
    data_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "table": self.table,
            "path": self.path,
            "kind": self.kind,
            "name": self.name,
            "data_type": self.data_type,
        }


class StructuredContextStore:
    """Per-task DuckDB store for structured context assets."""

    def __init__(
        self,
        *,
        task: PublicTask,
        conn: duckdb.DuckDBPyConnection,
        tables: list[StructuredTable],
    ) -> None:
        self.task = task
        self.conn = conn
        self._tables = list(tables)
        self._table_by_name = {_normalize_identifier(table.name): table for table in self._tables}
        self._closed = False

    @classmethod
    def from_task(cls, task: PublicTask) -> StructuredContextStore:
        conn = duckdb.connect(":memory:")
        tables: list[StructuredTable] = []
        used_names: set[str] = set()

        for path in sorted(task.context_dir.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(task.context_dir).as_posix()
            suffix = path.suffix.lower()
            if suffix == ".csv":
                table_name = _dedupe_name(_normalize_identifier(path.stem), used_names)
                tables.append(
                    _register_csv_table(
                        conn=conn,
                        path=path,
                        relative_path=relative_path,
                        table_name=table_name,
                    )
                )
            elif suffix == ".json":
                table_name = _dedupe_name(_normalize_identifier(path.stem), used_names)
                tables.append(
                    _register_json_records_table(
                        conn=conn,
                        path=path,
                        relative_path=relative_path,
                        table_name=table_name,
                    )
                )
            elif suffix in {".db", ".sqlite"}:
                tables.extend(
                    _register_sqlite_tables(
                        conn=conn,
                        path=path,
                        relative_path=relative_path,
                        used_names=used_names,
                    )
                )
        return cls(task=task, conn=conn, tables=tables)

    def tables(self) -> list[StructuredTable]:
        return list(self._tables)

    def table_dicts(self) -> list[dict[str, Any]]:
        return [table.to_dict() for table in self._tables]

    def register_doc_fact_tables(
        self,
        *,
        paragraphs: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> None:
        """Register fixed doc annotation tables in this task's in-memory DuckDB."""
        self._register_fixed_table(
            "doc_paragraphs",
            _DOC_PARAGRAPH_COLUMNS,
            paragraphs,
        )
        self._register_fixed_table(
            "doc_facts",
            _DOC_FACT_COLUMNS,
            facts,
        )
        self._register_fixed_table(
            "doc_relations",
            _DOC_RELATION_COLUMNS,
            relations,
        )

    def close(self) -> None:
        if self._closed:
            return
        self.conn.close()
        self._closed = True

    def __enter__(self) -> StructuredContextStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def fields(self) -> list[StructuredField]:
        fields: list[StructuredField] = []
        for table in self._tables:
            types = self._column_types(table.name)
            for column in table.columns:
                fields.append(
                    StructuredField(
                        field=f"{table.name}.{column}",
                        table=table.name,
                        path=table.path,
                        kind=table.kind,
                        name=column,
                        data_type=types.get(column, "UNKNOWN"),
                    )
                )
        return fields

    def inspect_schema(self) -> dict[str, Any]:
        return {
            "tables": [
                {
                    **table.to_dict(),
                    "columns": [
                        {
                            "name": field.name,
                            "type": field.data_type,
                        }
                        for field in self.fields()
                        if field.table == table.name
                    ],
                }
                for table in self._tables
            ]
        }

    def table_for_field(self, field_name: str) -> StructuredTable | None:
        table_name, _column = _split_field(field_name)
        if not table_name:
            return None
        return self._table_by_name.get(_normalize_identifier(table_name))

    def row_count(self, table_name: str) -> int:
        table = self._table_by_name.get(_normalize_identifier(table_name))
        if table is None:
            return 0
        quoted_table = _quote_identifier(table.name)
        return int(self.conn.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()[0])

    def field_profile(self, field_name: str) -> dict[str, Any] | None:
        table_name, column = _split_field(field_name)
        if not table_name or not column:
            return None
        table = self._table_by_name.get(_normalize_identifier(table_name))
        if table is None or column not in table.columns:
            return None
        quoted_table = _quote_identifier(table.name)
        quoted_column = _quote_identifier(column)
        stats = self.conn.execute(
            f"""
            SELECT
              COUNT(*) AS row_count,
              COUNT({quoted_column}) AS non_null_count,
              COUNT(*) - COUNT({quoted_column}) AS null_count,
              MIN({quoted_column}) AS min_value,
              MAX({quoted_column}) AS max_value
            FROM {quoted_table}
            """
        ).fetchone()
        return {
            "field": f"{table.name}.{column}",
            "table": table.name,
            "path": table.path,
            "kind": table.kind,
            "name": column,
            "data_type": self._column_types(table.name).get(column, "UNKNOWN"),
            "row_count": int(stats[0] if stats else 0),
            "non_null_count": int(stats[1] if stats else 0),
            "null_count": int(stats[2] if stats else 0),
            "min_value": _json_value(stats[3] if stats else None),
            "max_value": _json_value(stats[4] if stats else None),
            "sample_values": self.sample_values(f"{table.name}.{column}"),
        }

    def sample_values(
        self,
        field_name: str,
        *,
        limit: int = 10,
    ) -> list[str]:
        return self.distinct_values(
            field_name,
            limit=limit,
        )

    def distinct_values(
        self,
        field_name: str,
        *,
        limit: int = 5000,
    ) -> list[str]:
        table_name, column = _split_field(field_name)
        if not table_name or not column:
            return []
        table = self._table_by_name.get(_normalize_identifier(table_name))
        if table is None or column not in table.columns:
            return []
        quoted_table = _quote_identifier(table.name)
        quoted_column = _quote_identifier(column)
        rows = self.conn.execute(
            f"""
            WITH source AS (
              SELECT
                row_number() OVER () AS profile_row_number,
                CAST({quoted_column} AS VARCHAR) AS candidate_value
              FROM {quoted_table}
              WHERE {quoted_column} IS NOT NULL
                AND TRIM(CAST({quoted_column} AS VARCHAR)) <> ''
            )
            SELECT candidate_value
            FROM source
            GROUP BY candidate_value
            ORDER BY MIN(profile_row_number)
            LIMIT {max(1, int(limit))}
            """
        ).fetchall()
        return [str(row[0]) for row in rows]

    def match_value(
        self,
        field_name: str,
        raw_value: Any,
        *,
        limit: int = 20,
    ) -> dict[str, Any]:
        table_name, column = _split_field(field_name)
        if not table_name or not column:
            return {"matches": []}
        table = self._table_by_name.get(_normalize_identifier(table_name))
        if table is None or column not in table.columns:
            return {"matches": []}
        normalized_raw = _normalize_value(raw_value)
        if not normalized_raw:
            return {"matches": []}
        quoted_table = _quote_identifier(table.name)
        quoted_column = _quote_identifier(column)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT CAST({quoted_column} AS VARCHAR) AS candidate_value
            FROM {quoted_table}
            WHERE {quoted_column} IS NOT NULL
              AND trim(regexp_replace(
                    lower(CAST({quoted_column} AS VARCHAR)),
                    '[^a-z0-9]+',
                    ' ',
                    'g'
                  )) = {self.sql_literal(normalized_raw)}
            LIMIT {max(1, int(limit))}
            """
        ).fetchall()
        return {
            "field": field_name,
            "raw_value": raw_value,
            "matches": [str(row[0]) for row in rows],
        }

    def value_overlap(
        self,
        left_field: str,
        right_field: str,
        *,
        limit: int = 20,
        compare_as_number: bool | None = None,
    ) -> dict[str, Any]:
        left_table_name, left_column = _split_field(left_field)
        right_table_name, right_column = _split_field(right_field)
        if not left_table_name or not left_column or not right_table_name or not right_column:
            return {"overlap_count": 0, "sample_values": []}
        left_table = self._table_by_name.get(_normalize_identifier(left_table_name))
        right_table = self._table_by_name.get(_normalize_identifier(right_table_name))
        if (
            left_table is None
            or right_table is None
            or left_column not in left_table.columns
            or right_column not in right_table.columns
        ):
            return {"overlap_count": 0, "sample_values": []}

        left_types = self._column_types(left_table.name)
        right_types = self._column_types(right_table.name)
        if compare_as_number is None:
            compare_as_number = _duckdb_type_is_numeric(
                left_types.get(left_column, "")
            ) and _duckdb_type_is_numeric(right_types.get(right_column, ""))
        left_expr = self._normalized_value_expr(left_column, as_number=compare_as_number)
        right_expr = self._normalized_value_expr(right_column, as_number=compare_as_number)
        left_query = (
            f"SELECT DISTINCT {left_expr} AS value FROM {_quote_identifier(left_table.name)} "
            f"WHERE {_quote_identifier(left_column)} IS NOT NULL "
            f"AND TRIM(CAST({_quote_identifier(left_column)} AS VARCHAR)) <> ''"
        )
        right_query = (
            f"SELECT DISTINCT {right_expr} AS value FROM {_quote_identifier(right_table.name)} "
            f"WHERE {_quote_identifier(right_column)} IS NOT NULL "
            f"AND TRIM(CAST({_quote_identifier(right_column)} AS VARCHAR)) <> ''"
        )
        overlap_query = f"({left_query}) INTERSECT ({right_query})"
        overlap_rows = self.conn.execute(
            f"SELECT value FROM ({overlap_query}) LIMIT {max(1, int(limit))}"
        ).fetchall()
        overlap_count = self.conn.execute(
            f"SELECT COUNT(*) FROM ({overlap_query})"
        ).fetchone()[0]
        left_count = self.conn.execute(f"SELECT COUNT(*) FROM ({left_query})").fetchone()[0]
        right_count = self.conn.execute(f"SELECT COUNT(*) FROM ({right_query})").fetchone()[0]
        smaller_count = min(int(left_count), int(right_count))
        coverage = (int(overlap_count) / smaller_count) if smaller_count else 0.0
        left_distinct_count = int(left_count)
        right_distinct_count = int(right_count)
        overlap_count = int(overlap_count)
        left_coverage = (overlap_count / left_distinct_count) if left_distinct_count else 0.0
        right_coverage = (overlap_count / right_distinct_count) if right_distinct_count else 0.0
        union_count = left_distinct_count + right_distinct_count - overlap_count
        jaccard = (overlap_count / union_count) if union_count else 0.0
        return {
            "left": left_field,
            "right": right_field,
            "overlap_count": overlap_count,
            "left_distinct_count": left_distinct_count,
            "right_distinct_count": right_distinct_count,
            "coverage": round(coverage, 4),
            "left_coverage": round(left_coverage, 4),
            "right_coverage": round(right_coverage, 4),
            "containment": round(max(left_coverage, right_coverage), 4),
            "jaccard": round(jaccard, 4),
            "sample_values": [str(row[0]) for row in overlap_rows],
        }

    def records(
        self,
        table_name: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        table = self._table_by_name.get(_normalize_identifier(table_name))
        if table is None:
            return []
        sql = f"SELECT * FROM {_quote_identifier(table.name)}"
        if limit is not None:
            sql += f" LIMIT {max(1, int(limit))}"
        cursor = self.conn.execute(sql)
        columns = [str(item[0]) for item in cursor.description or []]
        return [
            {
                column: _json_value(value)
                for column, value in zip(columns, row, strict=False)
            }
            for row in cursor.fetchall()
        ]

    def query(self, sql: str, *, limit: int = 200) -> dict[str, Any]:
        normalized = sql.lstrip().lower()
        if not normalized.startswith(_READ_ONLY_SQL_PREFIXES):
            raise ValueError("Only read-only DuckDB SQL statements are allowed.")
        cursor = self.conn.execute(sql)
        columns = [str(item[0]) for item in cursor.description or []]
        rows = cursor.fetchmany(max(1, int(limit)) + 1)
        truncated = len(rows) > limit
        limited_rows = rows[:limit]
        return {
            "columns": columns,
            "rows": [
                [_json_value(value) for value in row]
                for row in limited_rows
            ],
            "row_count": len(limited_rows),
            "truncated": truncated,
        }

    def sql_literal(self, value: Any) -> str:
        return _sql_literal(value)

    def _column_types(self, table_name: str) -> dict[str, str]:
        rows = self.conn.execute(
            f"DESCRIBE SELECT * FROM {_quote_identifier(table_name)}"
        ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    def _normalized_value_expr(self, column: str, *, as_number: bool = False) -> str:
        quoted_column = _quote_identifier(column)
        if as_number:
            return f"CAST(CAST({quoted_column} AS DOUBLE) AS VARCHAR)"
        return (
            "trim(regexp_replace(lower(CAST("
            f"{quoted_column} AS VARCHAR)), '[^a-z0-9]+', ' ', 'g'))"
        )

    def _register_fixed_table(
        self,
        table_name: str,
        columns: list[tuple[str, str]],
        rows: list[dict[str, Any]],
    ) -> None:
        quoted_table = _quote_identifier(table_name)
        column_sql = ", ".join(
            f"{_quote_identifier(name)} {data_type}" for name, data_type in columns
        )
        self.conn.execute(f"DROP TABLE IF EXISTS {quoted_table}")
        self.conn.execute(f"CREATE TABLE {quoted_table} ({column_sql})")
        if rows:
            column_names = [name for name, _ in columns]
            placeholders = ", ".join(["?"] * len(column_names))
            insert_sql = (
                f"INSERT INTO {quoted_table} "
                f"({', '.join(_quote_identifier(name) for name in column_names)}) "
                f"VALUES ({placeholders})"
            )
            self.conn.executemany(
                insert_sql,
                [
                    tuple(row.get(column_name) for column_name in column_names)
                    for row in rows
                ],
            )
        self._replace_table_info(
            _table_info(self.conn, table_name, table_name, "doc_annotation")
        )

    def _replace_table_info(self, table: StructuredTable) -> None:
        normalized_name = _normalize_identifier(table.name)
        self._tables = [
            existing
            for existing in self._tables
            if _normalize_identifier(existing.name) != normalized_name
        ]
        self._tables.append(table)
        self._tables.sort(key=lambda item: (item.kind, item.path, item.name))
        self._table_by_name[normalized_name] = table


def _register_csv_table(
    *,
    conn: duckdb.DuckDBPyConnection,
    path: Path,
    relative_path: str,
    table_name: str,
) -> StructuredTable:
    source_sql = f"SELECT * FROM read_csv_auto({_sql_literal(path.as_posix())})"
    columns = _describe_query_columns(conn, source_sql)
    select_sql = source_sql
    if "__row_index" not in columns:
        select_sql = (
            f"SELECT row_number() OVER () AS __row_index, source.* FROM ({source_sql}) AS source"
        )
    conn.execute(f"CREATE TABLE {_quote_identifier(table_name)} AS {select_sql}")
    _normalize_imported_time_columns(conn, table_name)
    return _table_info(conn, table_name, relative_path, "csv")


def _register_json_records_table(
    *,
    conn: duckdb.DuckDBPyConnection,
    path: Path,
    relative_path: str,
    table_name: str,
) -> StructuredTable:
    json_sql = (
        f"SELECT * FROM read_json_auto("
        f"{_sql_literal(path.as_posix())}, "
        f"maximum_object_size = {JSON_MAXIMUM_OBJECT_SIZE})"
    )
    top_columns = _describe_query_columns(conn, json_sql)
    if "records" not in top_columns:
        raise ValueError(
            f"Unsupported JSON context structure for {relative_path}: expected a records array."
        )
    source_sql = (
        f"SELECT row_record.* FROM read_json_auto("
        f"{_sql_literal(path.as_posix())}, "
        f"maximum_object_size = {JSON_MAXIMUM_OBJECT_SIZE}), "
        "unnest(records) AS records_table(row_record)"
    )
    columns = _describe_query_columns(conn, source_sql)
    if not columns:
        raise ValueError(
            f"Unsupported JSON context structure for {relative_path}: records must contain objects."
        )
    select_sql = source_sql
    if "__row_index" not in columns:
        select_sql = (
            f"SELECT row_number() OVER () AS __row_index, source.* FROM ({source_sql}) AS source"
        )
    conn.execute(f"CREATE TABLE {_quote_identifier(table_name)} AS {select_sql}")
    _normalize_imported_time_columns(conn, table_name)
    return _table_info(conn, table_name, relative_path, "json")


def _register_sqlite_tables(
    *,
    conn: duckdb.DuckDBPyConnection,
    path: Path,
    relative_path: str,
    used_names: set[str],
) -> list[StructuredTable]:
    tables: list[StructuredTable] = []
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as sqlite_conn:
        table_rows = sqlite_conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        for (source_table,) in table_rows:
            if not isinstance(source_table, str) or not source_table.strip():
                continue
            table_name = _dedupe_name(source_table, used_names)
            tables.append(
                _register_sqlite_table(
                    duckdb_conn=conn,
                    sqlite_conn=sqlite_conn,
                    source_table=source_table,
                    table_name=table_name,
                    relative_path=relative_path,
                )
            )
    return tables


def _register_sqlite_table(
    *,
    duckdb_conn: duckdb.DuckDBPyConnection,
    sqlite_conn: sqlite3.Connection,
    source_table: str,
    table_name: str,
    relative_path: str,
) -> StructuredTable:
    import pandas as pd

    quoted_source_table = _quote_identifier(source_table)
    frame = pd.read_sql_query(f"SELECT * FROM {quoted_source_table}", sqlite_conn)
    if "__row_index" not in frame.columns:
        frame.insert(0, "__row_index", range(1, len(frame) + 1))
    view_name = f"__sqlite_import_{uuid.uuid4().hex}"
    duckdb_conn.register(view_name, frame)
    try:
        duckdb_conn.execute(
            f"CREATE TABLE {_quote_identifier(table_name)} AS "
            f"SELECT * FROM {_quote_identifier(view_name)}"
        )
    finally:
        duckdb_conn.unregister(view_name)
    _normalize_imported_time_columns(duckdb_conn, table_name)
    return _table_info(duckdb_conn, table_name, relative_path, "sqlite")


def _normalize_imported_time_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
) -> None:
    columns = _describe_query_columns(
        conn,
        f"SELECT * FROM {_quote_identifier(table_name)}",
    )
    column_types = _table_column_types(conn, table_name)
    time_columns = [
        column
        for column in columns
        if _should_normalize_time_column(
            conn=conn,
            table_name=table_name,
            column=column,
            data_type=column_types.get(column, ""),
        )
    ]
    if not time_columns:
        return

    quoted_table = _quote_identifier(table_name)
    temp_table_name = f"__time_normalized_{uuid.uuid4().hex}"
    quoted_temp_table = _quote_identifier(temp_table_name)
    udf_name = f"__normalize_time_value_{uuid.uuid4().hex}"
    conn.create_function(
        udf_name,
        _normalize_time_value_for_duckdb,
        ["VARCHAR"],
        "VARCHAR",
        null_handling="special",
    )

    time_column_set = set(time_columns)
    select_items = []
    for column in columns:
        quoted_column = _quote_identifier(column)
        if column in time_column_set:
            select_items.append(
                f"{_quote_identifier(udf_name)}(CAST({quoted_column} AS VARCHAR)) "
                f"AS {quoted_column}"
            )
        else:
            select_items.append(quoted_column)
    select_sql = ", ".join(select_items)
    conn.execute(f"CREATE TABLE {quoted_temp_table} AS SELECT {select_sql} FROM {quoted_table}")
    conn.execute(f"DROP TABLE {quoted_table}")
    conn.execute(f"ALTER TABLE {quoted_temp_table} RENAME TO {quoted_table}")


def _should_normalize_time_column(
    *,
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    column: str,
    data_type: str,
) -> bool:
    if column == "__row_index":
        return False
    if not _duckdb_type_is_temporal(data_type) and not _identifier_suggests_time(column):
        return False
    values = _sample_column_values(
        conn=conn,
        table_name=table_name,
        column=column,
        limit=TIME_VALUE_SAMPLE_LIMIT,
    )
    if not values:
        return False
    return any(_normalize_time_value(value) is not None for value in values)


def _identifier_suggests_time(column: str) -> bool:
    normalized = _normalize_identifier(column)
    tokens = set(normalized.split("_"))
    return bool(tokens & _TIME_IDENTIFIER_TOKENS)


def _sample_column_values(
    *,
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    column: str,
    limit: int,
) -> list[Any]:
    quoted_table = _quote_identifier(table_name)
    quoted_column = _quote_identifier(column)
    rows = conn.execute(
        f"""
        SELECT {quoted_column}
        FROM {quoted_table}
        WHERE {quoted_column} IS NOT NULL
          AND TRIM(CAST({quoted_column} AS VARCHAR)) <> ''
        LIMIT {max(1, int(limit))}
        """
    ).fetchall()
    return [row[0] for row in rows]


def _normalize_time_value_for_duckdb(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_time_value(value)
    return normalized if normalized is not None else str(value)


def _normalize_time_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = str(value).strip().strip("\"'")
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    compact = text.replace(",", "")
    if re.fullmatch(r"(?:18|19|20|21)\d{2}", compact):
        return compact

    match = re.fullmatch(r"((?:18|19|20|21)\d{2})(0[1-9]|1[0-2])", compact)
    if match:
        return _format_year_month(match.group(1), match.group(2))

    match = re.fullmatch(
        r"((?:18|19|20|21)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])",
        compact,
    )
    if match:
        return _format_full_date(match.group(1), match.group(2), match.group(3))

    match = re.fullmatch(
        r"((?:18|19|20|21)\d{2})[-/. ](0?[1-9]|1[0-2])",
        compact,
    )
    if match:
        return _format_year_month(match.group(1), match.group(2))

    match = re.fullmatch(
        r"((?:18|19|20|21)\d{2})[-/. ](0?[1-9]|1[0-2])"
        r"[-/. ](0?[1-9]|[12]\d|3[01])(?:[ T].*)?",
        compact,
    )
    if match:
        return _format_full_date(match.group(1), match.group(2), match.group(3))

    match = re.fullmatch(
        r"(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/]((?:18|19|20|21)\d{2})",
        compact,
    )
    if match:
        return _format_full_date(match.group(3), match.group(1), match.group(2))

    match = re.fullmatch(
        r"(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/]((?:18|19|20|21)\d{2})",
        compact,
    )
    if match:
        return _format_full_date(match.group(3), match.group(2), match.group(1))

    month_date = _normalize_month_name_date(compact)
    if month_date is not None:
        return month_date

    return None


def _normalize_month_name_date(text: str) -> str | None:
    month_token = r"([A-Za-z]+)\.?"
    year_token = r"((?:18|19|20|21)\d{2})"
    day_token = r"(0?[1-9]|[12]\d|3[01])"

    match = re.fullmatch(rf"{month_token} {year_token}", text, re.IGNORECASE)
    if match:
        month = _month_number(match.group(1))
        return _format_year_month(match.group(2), month) if month is not None else None

    match = re.fullmatch(rf"{year_token} {month_token}", text, re.IGNORECASE)
    if match:
        month = _month_number(match.group(2))
        return _format_year_month(match.group(1), month) if month is not None else None

    match = re.fullmatch(
        rf"{month_token} {day_token},? {year_token}(?:[ T].*)?",
        text,
        re.IGNORECASE,
    )
    if match:
        month = _month_number(match.group(1))
        return (
            _format_full_date(match.group(3), month, match.group(2))
            if month is not None
            else None
        )

    match = re.fullmatch(
        rf"{day_token} {month_token} {year_token}(?:[ T].*)?",
        text,
        re.IGNORECASE,
    )
    if match:
        month = _month_number(match.group(2))
        return (
            _format_full_date(match.group(3), month, match.group(1))
            if month is not None
            else None
        )

    return None


def _month_number(month: str) -> int | None:
    return _MONTH_NUMBERS.get(month.strip().lower().rstrip("."))


def _format_year_month(year: str | int, month: str | int) -> str | None:
    try:
        year_int = int(year)
        month_int = int(month)
    except (TypeError, ValueError):
        return None
    if year_int < 1800 or year_int > 2199 or month_int < 1 or month_int > 12:
        return None
    return f"{year_int:04d}-{month_int:02d}"


def _format_full_date(year: str | int, month: str | int, day: str | int) -> str | None:
    try:
        parsed = date(int(year), int(month), int(day))
    except (TypeError, ValueError):
        return None
    if parsed.year < 1800 or parsed.year > 2199:
        return None
    return parsed.isoformat()


def _table_info(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    relative_path: str,
    kind: str,
) -> StructuredTable:
    columns = _describe_query_columns(
        conn,
        f"SELECT * FROM {_quote_identifier(table_name)}",
    )
    row_count = int(
        conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}").fetchone()[0]
    )
    preview = _preview_row(conn, table_name, columns)
    return StructuredTable(
        name=table_name,
        path=relative_path,
        kind=kind,
        columns=columns,
        row_count=row_count,
        preview=preview,
    )

def _describe_query_columns(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
) -> list[str]:
    rows = conn.execute(f"DESCRIBE {sql}").fetchall()
    return [str(row[0]) for row in rows]


def _table_column_types(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
) -> dict[str, str]:
    rows = conn.execute(
        f"DESCRIBE SELECT * FROM {_quote_identifier(table_name)}"
    ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def _preview_row(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: list[str],
) -> dict[str, Any] | None:
    if not columns:
        return None
    cursor = conn.execute(f"SELECT * FROM {_quote_identifier(table_name)} LIMIT 1")
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        column: _json_value(value)
        for column, value in zip(columns, row, strict=False)
    }


def _dedupe_name(raw_name: str, used_names: set[str]) -> str:
    base = raw_name or "table"
    candidate = base
    suffix = 2
    while _normalize_identifier(candidate) in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(_normalize_identifier(candidate))
    return candidate


def _split_field(field_name: str) -> tuple[str | None, str | None]:
    if "." not in field_name:
        return None, field_name if field_name else None
    table, column = field_name.rsplit(".", 1)
    return table or None, column or None


def _normalize_identifier(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).lower()
    return re.sub(r"_+", "_", text).strip("_")


def _normalize_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "NULL"
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _duckdb_type_is_numeric(data_type: str) -> bool:
    normalized = str(data_type).upper()
    return any(
        token in normalized
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
            "FLOAT",
            "DOUBLE",
            "DECIMAL",
            "REAL",
        }
    )


def _duckdb_type_is_temporal(data_type: str) -> bool:
    normalized = str(data_type).upper()
    return any(token in normalized for token in {"DATE", "TIME", "TIMESTAMP"})


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)
