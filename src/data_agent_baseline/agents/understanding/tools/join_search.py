from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.understanding.tools.candidate_store import (
    CandidateStore,
    FieldProfile,
    normalize_value,
)
from data_agent_baseline.agents.understanding.tools.normalization import normalize_identifier

KEY_LIKE_TOKENS = {"id", "key"}
CODE_LIKE_TOKENS = {"code"}
POSTAL_LIKE_TOKENS = {"zip", "postal", "postcode"}
NAME_LIKE_TOKENS = {"name", "title", "label"}
LOW_CARDINALITY_TOKENS = {"category", "status", "type", "gender", "sex", "state", "country"}
NUMERIC_DATA_TYPES = {"number", "integer"}
MEASURE_DATA_TYPES = {"number", "integer", "date"}
HIGH_UNIQUE_RATIO = 0.8
KEY_CONTAINMENT_THRESHOLD = 0.5
NAME_CONTAINMENT_THRESHOLD = 0.35
WEAK_CONTAINMENT_THRESHOLD = 0.65
MAX_SAMPLE_VALUES = 5
GENERIC_SHARED_KEY_NAMES = {"id", "key", "code"}
JOIN_TYPE_PRIORITY = {
    "foreign_key": 4,
    "shared_key": 3,
    "value_overlap": 2,
    "row_order": 1,
}


def _normalize(text: str) -> str:
    return normalize_identifier(text)


def _singularize(name: str) -> str:
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if name.endswith("ses") and len(name) > 3:
        return name[:-2]
    if name.endswith("s") and len(name) > 1:
        return name[:-1]
    return name


@dataclass(frozen=True, slots=True)
class _TableProfile:
    table: str
    fields: list[str]
    table_aliases: set[str]
    primary_keys: list[str]


@dataclass(frozen=True, slots=True)
class JoinCandidate:
    left: str
    right: str
    score: float
    evidence: list[str]
    metadata: dict[str, object]


class RuleBasedJoinSearcher:
    """Find likely joins from common foreign-key naming conventions."""

    def __init__(self, top_k: int = 20) -> None:
        self.top_k = top_k

    def search(
        self,
        candidate_store: CandidateStore,
    ) -> list[JoinCandidate]:
        tables = self._table_profiles(candidate_store)
        fields_by_name = {field.field: field for field in candidate_store.fields if field.table}
        structured_store = getattr(candidate_store, "structured_store", None)
        candidates: list[JoinCandidate] = []
        for source in tables:
            for field in source.fields:
                candidates.extend(self._foreign_key_candidates(source, field, tables))
                candidates.extend(
                    self._shared_key_candidates(
                        source,
                        field,
                        tables,
                        fields_by_name=fields_by_name,
                        structured_store=structured_store,
                    )
                )

        candidates.extend(self._value_overlap_candidates(candidate_store))
        strong_pairs = self._strong_table_pairs(candidates)
        candidates.extend(self._row_order_candidates(candidate_store, strong_pairs))

        deduped = self._dedupe(candidates)
        deduped.sort(key=lambda item: (-item.score, item.left, item.right))
        return deduped[: self.top_k]

    def _table_profiles(self, candidate_store: CandidateStore) -> list[_TableProfile]:
        profiles: list[_TableProfile] = []
        for table_schema in candidate_store.tables:
            table = table_schema.table
            fields = list(table_schema.columns)
            if not table.strip() or not fields:
                continue
            normalized_table = _normalize(table)
            aliases = self._table_aliases(normalized_table)
            normalized_fields = {_normalize(str(field)): str(field) for field in fields}
            primary_keys = self._primary_keys(aliases, normalized_fields)
            profiles.append(
                _TableProfile(
                    table=table,
                    fields=[str(field) for field in fields],
                    table_aliases=aliases,
                    primary_keys=primary_keys,
                )
            )
        return profiles

    def _foreign_key_candidates(
        self,
        source: _TableProfile,
        field: str,
        tables: list[_TableProfile],
    ) -> list[JoinCandidate]:
        normalized_field = _normalize(field)
        references = self._referenced_table_aliases(normalized_field)
        if not references:
            return []

        candidates: list[JoinCandidate] = []
        for target in tables:
            if target.table == source.table:
                continue
            if not references & target.table_aliases:
                continue
            for target_key in target.primary_keys:
                score = 0.94 if normalized_field.startswith("link_to_") else 0.9
                evidence = [
                    f"foreign-key naming pattern `{source.table}.{field}` references table `{target.table}`",
                    f"target table `{target.table}` has primary key candidate `{target_key}`",
                ]
                candidates.append(
                    JoinCandidate(
                        left=f"{source.table}.{field}",
                        right=f"{target.table}.{target_key}",
                        score=score,
                        evidence=evidence,
                        metadata={"join_type": "foreign_key"},
                    )
                )
        return candidates

    def _shared_key_candidates(
        self,
        source: _TableProfile,
        field: str,
        tables: list[_TableProfile],
        *,
        fields_by_name: dict[str, FieldProfile],
        structured_store: Any,
    ) -> list[JoinCandidate]:
        normalized_field = _normalize(field)
        known_shared_key = normalized_field in {
            "id",
            "code",
            "zip",
            "postal",
            "postal_code",
            "postcode",
            *self._primary_key_names(source.table_aliases),
        }
        same_name_business_key = self._is_same_name_business_key_name(normalized_field)
        if not known_shared_key and not same_name_business_key:
            return []

        candidates: list[JoinCandidate] = []
        for target in tables:
            if target.table <= source.table:
                continue
            for target_field in target.fields:
                if _normalize(target_field) != normalized_field:
                    continue
                left = f"{source.table}.{field}"
                right = f"{target.table}.{target_field}"
                metadata: dict[str, object] = {"join_type": "shared_key"}
                evidence = [
                    f"shared key field `{normalized_field}` appears in both tables",
                    "lower-confidence join because no directional foreign-key prefix was found",
                ]
                if same_name_business_key:
                    metadata["same_name_business_key"] = True
                    metadata.update(
                        self._same_name_business_key_overlap_metadata(
                            left,
                            right,
                            fields_by_name=fields_by_name,
                            structured_store=structured_store,
                        )
                    )
                    if "overlap_count" in metadata:
                        evidence.extend(
                            [
                                f"`{left}` and `{right}` share {metadata['overlap_count']} normalized values",
                                "shared-key containment is "
                                f"{float(metadata.get('containment') or 0.0):.2f}",
                            ]
                        )
                    evidence.extend(self._value_overlap_warnings(metadata))
                candidates.append(
                    JoinCandidate(
                        left=left,
                        right=right,
                        score=0.72,
                        evidence=evidence,
                        metadata=metadata,
                    )
                )
        return candidates

    def _value_overlap_candidates(self, candidate_store: CandidateStore) -> list[JoinCandidate]:
        fields = [field for field in candidate_store.fields if field.table]
        candidates: list[JoinCandidate] = []
        structured_store = getattr(candidate_store, "structured_store", None)
        for left_index, left in enumerate(fields):
            for right in fields[left_index + 1 :]:
                if left.table == right.table:
                    continue
                if not self._field_pair_can_join_by_overlap(left, right):
                    continue
                entity_compatible = self._overlap_entity_compatible(left, right)

                if (
                    structured_store is not None
                    and structured_store.table_for_field(left.field) is not None
                    and structured_store.table_for_field(right.field) is not None
                ):
                    overlap_stats = structured_store.value_overlap(
                        left.field,
                        right.field,
                        compare_as_number=self._compare_overlap_as_number(left, right),
                    )
                    overlap_count = int(overlap_stats.get("overlap_count") or 0)
                    if overlap_count <= 0:
                        continue
                    left_distinct_count = int(overlap_stats.get("left_distinct_count") or 0)
                    right_distinct_count = int(overlap_stats.get("right_distinct_count") or 0)
                    sample_values = [
                        str(value)
                        for value in overlap_stats.get("sample_values", [])
                        if str(value).strip()
                    ]
                    metadata = self._overlap_metadata(
                        left=left,
                        right=right,
                        overlap_count=overlap_count,
                        left_distinct_count=left_distinct_count,
                        right_distinct_count=right_distinct_count,
                        sample_values=sample_values,
                        entity_compatible=entity_compatible,
                    )
                    if not self._overlap_stats_are_meaningful(
                        left,
                        right,
                        metadata,
                    ):
                        continue
                    candidates.append(
                        JoinCandidate(
                            left=left.field,
                            right=right.field,
                            score=self._value_overlap_score(metadata),
                            evidence=[
                                f"`{left.field}` and `{right.field}` share {overlap_count} normalized values",
                                "value-overlap containment is "
                                f"{float(metadata['containment']):.2f}",
                                *self._value_overlap_warnings(metadata),
                            ],
                            metadata=metadata,
                        )
                    )
                    continue

                left_values = self._normalized_sample_values(left)
                right_values = self._normalized_sample_values(right)
                overlap = sorted(left_values & right_values)
                if not overlap:
                    continue

                metadata = self._overlap_metadata(
                    left=left,
                    right=right,
                    overlap_count=len(overlap),
                    left_distinct_count=len(left_values),
                    right_distinct_count=len(right_values),
                    sample_values=overlap,
                    entity_compatible=entity_compatible,
                )
                if not self._overlap_is_meaningful(
                    left,
                    right,
                    overlap,
                    metadata,
                ):
                    continue

                candidates.append(
                    JoinCandidate(
                        left=left.field,
                        right=right.field,
                        score=self._value_overlap_score(metadata),
                        evidence=[
                            f"`{left.field}` and `{right.field}` share {len(overlap)} normalized values",
                            f"value-overlap containment is {float(metadata['containment']):.2f}",
                            *self._value_overlap_warnings(metadata),
                        ],
                        metadata=metadata,
                    )
                )
        return candidates

    def _row_order_candidates(
        self,
        candidate_store: CandidateStore,
        strong_pairs: set[tuple[str, str]],
    ) -> list[JoinCandidate]:
        tables = candidate_store.tables
        candidates: list[JoinCandidate] = []
        for left_index, left in enumerate(tables):
            for right in tables[left_index + 1 :]:
                if left.table == right.table:
                    continue
                if self._table_pair_key(left.table, right.table) in strong_pairs:
                    continue
                if left.row_count < 2 or right.row_count < 2:
                    continue

                smaller = min(left.row_count, right.row_count)
                larger = max(left.row_count, right.row_count)
                ratio = smaller / larger
                if ratio == 1:
                    score = 0.42
                    first_evidence = (
                        "tables have identical record counts, enabling row-order alignment fallback"
                    )
                elif ratio >= 0.9 and smaller >= 5:
                    score = 0.32
                    first_evidence = (
                        "tables have near-identical record counts, enabling weak row-order alignment fallback"
                    )
                else:
                    continue

                candidates.append(
                    JoinCandidate(
                        left=f"{left.table}.__row_index",
                        right=f"{right.table}.__row_index",
                        score=score,
                        evidence=[
                            first_evidence,
                            "row-order joins are weak and must be verified against source evidence before use",
                        ],
                        metadata={
                            "join_type": "row_order",
                            "left_table": left.table,
                            "right_table": right.table,
                            "left_count": left.row_count,
                            "right_count": right.row_count,
                            "requires_verification": True,
                        },
                    )
                )
        return candidates

    def _referenced_table_aliases(self, normalized_field: str) -> set[str]:
        references: set[str] = set()
        for prefix in ("link_to_", "fk_", "ref_"):
            if normalized_field.startswith(prefix) and len(normalized_field) > len(prefix):
                references.update(self._table_aliases(normalized_field[len(prefix) :]))
        for suffix in ("_id", "_key", "_code"):
            if normalized_field.endswith(suffix) and len(normalized_field) > len(suffix):
                references.update(self._table_aliases(normalized_field[: -len(suffix)]))
        if self._is_postal_like_name(normalized_field):
            references.update(self._postal_aliases(normalized_field))
        return references

    def _primary_keys(self, table_aliases: set[str], normalized_fields: dict[str, str]) -> list[str]:
        keys: list[str] = []
        key_names = [
            "id",
            "code",
            "zip",
            "postal",
            "postal_code",
            "postcode",
            *self._primary_key_names(table_aliases),
        ]
        for normalized_name in key_names:
            field = normalized_fields.get(normalized_name)
            if field is not None:
                keys.append(field)
        return self._dedupe_strings(keys)

    def _primary_key_names(self, table_aliases: set[str]) -> set[str]:
        names: set[str] = set()
        for alias in table_aliases:
            if not alias:
                continue
            names.update({f"{alias}_id", f"{alias}_key", f"{alias}_code"})
        return names

    def _table_aliases(self, normalized_table: str) -> set[str]:
        aliases = {normalized_table, _singularize(normalized_table)}
        parts = [part for part in normalized_table.split("_") if part]
        if len(parts) > 1:
            aliases.add(parts[-1])
            aliases.add(_singularize(parts[-1]))
        return {alias for alias in aliases if alias}

    def _name_tokens(self, value: str) -> set[str]:
        return {token for token in _normalize(value).split("_") if token}

    def _postal_aliases(self, normalized_name: str) -> set[str]:
        tokens = self._name_tokens(normalized_name)
        aliases = set(tokens & POSTAL_LIKE_TOKENS)
        if "zip" in tokens or "postal" in tokens or "postcode" in tokens:
            aliases.update({"zip", "postal", "postcode"})
        return aliases

    def _is_postal_like_name(self, normalized_name: str) -> bool:
        return bool(self._name_tokens(normalized_name) & POSTAL_LIKE_TOKENS)

    def _is_same_name_business_key_name(self, normalized_name: str) -> bool:
        if normalized_name in GENERIC_SHARED_KEY_NAMES:
            return False
        tokens = {token for token in normalized_name.split("_") if token}
        if len(tokens) < 2:
            return False
        key_tokens = KEY_LIKE_TOKENS | CODE_LIKE_TOKENS | POSTAL_LIKE_TOKENS
        return bool(tokens & key_tokens) and bool(tokens - key_tokens)

    def _is_same_name_business_key(self, left: FieldProfile, right: FieldProfile) -> bool:
        normalized_name = _normalize(left.name)
        return normalized_name == _normalize(right.name) and self._is_same_name_business_key_name(
            normalized_name
        )

    def _is_key_code_postal_like(self, field: FieldProfile) -> bool:
        tokens = self._name_tokens(field.name)
        return bool(tokens & (KEY_LIKE_TOKENS | CODE_LIKE_TOKENS | POSTAL_LIKE_TOKENS))

    def _is_name_like(self, field: FieldProfile) -> bool:
        return bool(self._name_tokens(field.name) & NAME_LIKE_TOKENS)

    def _field_entity_aliases(self, field: FieldProfile) -> set[str]:
        normalized_name = _normalize(field.name)
        aliases: set[str] = set()
        for prefix in ("link_to_", "fk_", "ref_", "id_"):
            if normalized_name.startswith(prefix) and len(normalized_name) > len(prefix):
                aliases.update(self._table_aliases(normalized_name[len(prefix) :]))
        for suffix in ("_id", "_key", "_code"):
            if normalized_name.endswith(suffix) and len(normalized_name) > len(suffix):
                aliases.update(self._table_aliases(normalized_name[: -len(suffix)]))
        if self._is_postal_like_name(normalized_name):
            aliases.update(self._postal_aliases(normalized_name))
        return aliases

    def _field_table_aliases(self, field: FieldProfile) -> set[str]:
        return self._table_aliases(_normalize(field.table)) if field.table else set()

    def _overlap_entity_compatible(self, left: FieldProfile, right: FieldProfile) -> bool:
        if _normalize(left.name) == _normalize(right.name):
            return True
        left_entities = self._field_entity_aliases(left)
        right_entities = self._field_entity_aliases(right)
        if left_entities & (right_entities | self._field_table_aliases(right)):
            return True
        return bool(right_entities & (left_entities | self._field_table_aliases(left)))

    def _field_pair_can_join_by_overlap(self, left: FieldProfile, right: FieldProfile) -> bool:
        if left.data_type in {"empty", "boolean"} or right.data_type in {"empty", "boolean"}:
            return False
        if left.data_type == "date" or right.data_type == "date":
            return False
        if (
            left.data_type in NUMERIC_DATA_TYPES
            and not self._is_key_code_postal_like(left)
        ):
            return False
        if (
            right.data_type in NUMERIC_DATA_TYPES
            and not self._is_key_code_postal_like(right)
        ):
            return False
        if self._is_low_cardinality_semantic_field(left):
            return False
        return not self._is_low_cardinality_semantic_field(right)

    def _is_low_cardinality_semantic_field(self, field: FieldProfile) -> bool:
        normalized_name = _normalize(field.name)
        if field.is_id_like:
            return False
        if normalized_name in LOW_CARDINALITY_TOKENS:
            return True
        return 0 < len(field.sample_values) <= 2 and field.non_null_count > 5

    def _normalized_sample_values(self, field: FieldProfile) -> set[str]:
        values = field.sample_values
        normalized: set[str] = set()
        for value in values:
            text = normalize_value(value)
            if len(text) >= 2:
                normalized.add(text)
        return normalized

    def _compare_overlap_as_number(self, left: FieldProfile, right: FieldProfile) -> bool:
        if self._is_key_code_postal_like(left) or self._is_key_code_postal_like(right):
            return False
        return left.data_type in NUMERIC_DATA_TYPES and right.data_type in NUMERIC_DATA_TYPES

    def _same_name_business_key_overlap_metadata(
        self,
        left_field: str,
        right_field: str,
        *,
        fields_by_name: dict[str, FieldProfile],
        structured_store: Any,
    ) -> dict[str, object]:
        left = fields_by_name.get(left_field)
        right = fields_by_name.get(right_field)
        if left is None or right is None:
            return {}
        if structured_store is None:
            return {}
        if (
            structured_store.table_for_field(left_field) is None
            or structured_store.table_for_field(right_field) is None
        ):
            return {}
        overlap_stats = structured_store.value_overlap(
            left_field,
            right_field,
            compare_as_number=self._compare_overlap_as_number(left, right),
        )
        overlap_count = int(overlap_stats.get("overlap_count") or 0)
        if overlap_count <= 0:
            return {}
        sample_values = [
            str(value)
            for value in overlap_stats.get("sample_values", [])
            if str(value).strip()
        ]
        metadata = self._overlap_metadata(
            left=left,
            right=right,
            overlap_count=overlap_count,
            left_distinct_count=int(overlap_stats.get("left_distinct_count") or 0),
            right_distinct_count=int(overlap_stats.get("right_distinct_count") or 0),
            sample_values=sample_values,
            entity_compatible=True,
        )
        metadata["join_type"] = "shared_key"
        metadata["same_name_business_key"] = True
        if metadata.get("multiplicity") != "one_to_one":
            metadata.setdefault(
                "warning",
                "same-name business key is not one-to-one; verify row multiplicity before use",
            )
            metadata["requires_verification"] = True
        return metadata

    def _overlap_metadata(
        self,
        *,
        left: FieldProfile,
        right: FieldProfile,
        overlap_count: int,
        left_distinct_count: int,
        right_distinct_count: int,
        sample_values: list[str],
        entity_compatible: bool,
    ) -> dict[str, object]:
        left_distinct_count = max(0, int(left_distinct_count))
        right_distinct_count = max(0, int(right_distinct_count))
        overlap_count = max(0, int(overlap_count))
        left_coverage = overlap_count / left_distinct_count if left_distinct_count else 0.0
        right_coverage = overlap_count / right_distinct_count if right_distinct_count else 0.0
        union_count = left_distinct_count + right_distinct_count - overlap_count
        jaccard = overlap_count / union_count if union_count else 0.0
        containment = max(left_coverage, right_coverage)
        left_unique_ratio = self._unique_ratio(left, left_distinct_count)
        right_unique_ratio = self._unique_ratio(right, right_distinct_count)
        joined_rows_estimate = self._estimated_join_rows(
            left,
            right,
            overlap_count,
            left_distinct_count,
            right_distinct_count,
        )
        left_rows = max(1, int(left.non_null_count or 0))
        right_rows = max(1, int(right.non_null_count or 0))
        left_match_multiplier = joined_rows_estimate / left_rows
        right_match_multiplier = joined_rows_estimate / right_rows
        multiplicity = self._multiplicity(left_unique_ratio, right_unique_ratio)
        many_to_many_risk = self._many_to_many_risk(
            multiplicity,
            left_match_multiplier,
            right_match_multiplier,
        )
        fanout_risk = self._fanout_risk(multiplicity, many_to_many_risk)
        overlap_tier = self._overlap_tier(
            left,
            right,
            containment=containment,
            entity_compatible=entity_compatible,
            multiplicity=multiplicity,
            fanout_risk=fanout_risk,
            left_unique_ratio=left_unique_ratio,
            right_unique_ratio=right_unique_ratio,
        )
        return {
            "join_type": "value_overlap",
            "overlap_tier": overlap_tier,
            "overlap_count": overlap_count,
            "left_distinct_count": left_distinct_count,
            "right_distinct_count": right_distinct_count,
            "left_coverage": round(left_coverage, 3),
            "right_coverage": round(right_coverage, 3),
            "containment": round(containment, 3),
            "jaccard": round(jaccard, 3),
            "coverage": round(containment, 3),
            "left_unique_ratio": round(left_unique_ratio, 3),
            "right_unique_ratio": round(right_unique_ratio, 3),
            "left_match_multiplier": round(left_match_multiplier, 3),
            "right_match_multiplier": round(right_match_multiplier, 3),
            "multiplicity": multiplicity,
            "many_to_many_risk": many_to_many_risk,
            "fanout_risk": fanout_risk,
            "sample_values": sample_values[:MAX_SAMPLE_VALUES],
            "entity_compatible": entity_compatible,
            "weak_overlap": overlap_tier == "weak_value_overlap",
            **self._value_overlap_warning_metadata(
                left,
                right,
                entity_compatible,
                overlap_tier=overlap_tier,
                fanout_risk=fanout_risk,
            ),
        }

    def _unique_ratio(self, field: FieldProfile, distinct_count: int) -> float:
        denominator = max(1, int(field.non_null_count or 0))
        return min(1.0, max(0.0, distinct_count / denominator))

    def _estimated_join_rows(
        self,
        left: FieldProfile,
        right: FieldProfile,
        overlap_count: int,
        left_distinct_count: int,
        right_distinct_count: int,
    ) -> float:
        left_per_key = (left.non_null_count / left_distinct_count) if left_distinct_count else 0.0
        right_per_key = (right.non_null_count / right_distinct_count) if right_distinct_count else 0.0
        return max(0.0, overlap_count * left_per_key * right_per_key)

    def _multiplicity(self, left_unique_ratio: float, right_unique_ratio: float) -> str:
        left_unique = left_unique_ratio >= HIGH_UNIQUE_RATIO
        right_unique = right_unique_ratio >= HIGH_UNIQUE_RATIO
        if left_unique and right_unique:
            return "one_to_one"
        if not left_unique and right_unique:
            return "many_to_one"
        if left_unique and not right_unique:
            return "one_to_many"
        return "many_to_many"

    def _many_to_many_risk(
        self,
        multiplicity: str,
        left_match_multiplier: float,
        right_match_multiplier: float,
    ) -> str:
        if multiplicity != "many_to_many":
            return "low"
        if left_match_multiplier > 1.5 and right_match_multiplier > 1.5:
            return "high"
        return "medium"

    def _fanout_risk(self, multiplicity: str, many_to_many_risk: str) -> str:
        if multiplicity == "one_to_one":
            return "low"
        if multiplicity in {"many_to_one", "one_to_many"}:
            return "low"
        if many_to_many_risk == "high":
            return "high"
        return "medium"

    def _overlap_tier(
        self,
        left: FieldProfile,
        right: FieldProfile,
        *,
        containment: float,
        entity_compatible: bool,
        multiplicity: str,
        fanout_risk: str,
        left_unique_ratio: float,
        right_unique_ratio: float,
    ) -> str:
        both_key_like = self._is_key_code_postal_like(left) and self._is_key_code_postal_like(right)
        same_name_business_key = self._is_same_name_business_key(left, right)
        if (
            both_key_like
            and (entity_compatible or same_name_business_key)
            and self._overlap_types_compatible(left, right)
            and containment >= KEY_CONTAINMENT_THRESHOLD
            and (
                same_name_business_key
                or max(left_unique_ratio, right_unique_ratio) >= HIGH_UNIQUE_RATIO
                or multiplicity != "many_to_many"
            )
            and not self._ordinary_measure_overlap(left, right)
        ):
            return "key_overlap"
        if self._is_name_like(left) and self._is_name_like(right):
            if fanout_risk == "low" and max(left_unique_ratio, right_unique_ratio) >= HIGH_UNIQUE_RATIO:
                return "dimension_name_overlap"
            return "broad_name_overlap"
        return "weak_value_overlap"

    def _overlap_types_compatible(self, left: FieldProfile, right: FieldProfile) -> bool:
        if left.data_type == right.data_type:
            return True
        if self._is_key_code_postal_like(left) and self._is_key_code_postal_like(right):
            return left.data_type != "date" and right.data_type != "date"
        return False

    def _ordinary_measure_overlap(self, left: FieldProfile, right: FieldProfile) -> bool:
        if left.data_type not in MEASURE_DATA_TYPES and right.data_type not in MEASURE_DATA_TYPES:
            return False
        return not (self._is_key_code_postal_like(left) and self._is_key_code_postal_like(right))

    def _overlap_is_meaningful(
        self,
        left: FieldProfile,
        right: FieldProfile,
        overlap: list[str],
        metadata: dict[str, object],
    ) -> bool:
        return self._overlap_metadata_is_meaningful(left, right, len(overlap), metadata)

    def _overlap_stats_are_meaningful(
        self,
        left: FieldProfile,
        right: FieldProfile,
        metadata: dict[str, object],
    ) -> bool:
        overlap_count = int(metadata.get("overlap_count") or 0)
        return self._overlap_metadata_is_meaningful(left, right, overlap_count, metadata)

    def _overlap_metadata_is_meaningful(
        self,
        left: FieldProfile,
        right: FieldProfile,
        overlap_count: int,
        metadata: dict[str, object],
    ) -> bool:
        tier = str(metadata.get("overlap_tier") or "")
        containment = float(metadata.get("containment") or 0.0)
        if tier == "key_overlap":
            return overlap_count >= 1 and containment >= KEY_CONTAINMENT_THRESHOLD
        if tier in {"dimension_name_overlap", "broad_name_overlap"}:
            return overlap_count >= 2 and containment >= NAME_CONTAINMENT_THRESHOLD
        if left.data_type in NUMERIC_DATA_TYPES or right.data_type in NUMERIC_DATA_TYPES:
            return False
        return overlap_count >= 3 and containment >= WEAK_CONTAINMENT_THRESHOLD

    def _value_overlap_score(self, metadata: dict[str, object]) -> float:
        containment = float(metadata.get("containment") or 0.0)
        tier = str(metadata.get("overlap_tier") or "")
        if tier == "key_overlap":
            base = 0.58
            cap = 0.68
        elif tier == "dimension_name_overlap":
            base = 0.48
            cap = 0.58
        elif tier == "broad_name_overlap":
            base = 0.34
            cap = 0.44
        else:
            base = 0.24
            cap = 0.34
        return round(min(base + containment * 0.18, cap), 2)

    def _weak_id_overlap(
        self,
        left: FieldProfile,
        right: FieldProfile,
        entity_compatible: bool,
    ) -> bool:
        return (left.is_id_like or right.is_id_like) and not entity_compatible

    def _value_overlap_warnings(self, metadata: dict[str, object]) -> list[str]:
        warning = metadata.get("warning")
        if isinstance(warning, str) and warning.strip():
            return [warning.strip()]
        return []

    def _value_overlap_warning_metadata(
        self,
        left: FieldProfile,
        right: FieldProfile,
        entity_compatible: bool,
        *,
        overlap_tier: str,
        fanout_risk: str,
    ) -> dict[str, object]:
        if overlap_tier == "weak_value_overlap":
            return {
                "warning": "weak value-overlap evidence; values overlap without key/name structure",
                "requires_verification": True,
            }
        if overlap_tier == "broad_name_overlap":
            return {
                "warning": "broad name-overlap evidence; verify row multiplicity before use",
                "requires_verification": True,
            }
        if fanout_risk == "high":
            return {
                "warning": "high fanout risk; verify whether this broad-field join is intended",
                "requires_verification": True,
            }
        if not self._weak_id_overlap(left, right, entity_compatible):
            return {}
        return {
            "warning": "weak value-overlap evidence; id-like field names point to different entities",
            "requires_verification": True,
        }

    def _strong_table_pairs(self, candidates: list[JoinCandidate]) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for candidate in candidates:
            if candidate.score < 0.45:
                continue
            tables = sorted(self._candidate_tables(candidate))
            if len(tables) == 2:
                pairs.add((tables[0], tables[1]))
        return pairs

    def _candidate_tables(self, candidate: JoinCandidate) -> set[str]:
        tables: set[str] = set()
        for field in [candidate.left, candidate.right]:
            if "." in field:
                table, _ = field.split(".", 1)
                if table:
                    tables.add(table)
        path = candidate.metadata.get("path")
        if isinstance(path, list):
            for edge in path:
                if not isinstance(edge, dict):
                    continue
                for field in [edge.get("left"), edge.get("right")]:
                    if isinstance(field, str) and "." in field:
                        table, _ = field.split(".", 1)
                        if table:
                            tables.add(table)
        for key in ("left_table", "right_table", "target_table", "lookup_table"):
            value = candidate.metadata.get(key)
            if isinstance(value, str) and value:
                tables.add(value)
        return tables

    def _table_pair_key(self, left: str, right: str) -> tuple[str, str]:
        first, second = sorted([left, right])
        return first, second

    def _dedupe(self, candidates: list[JoinCandidate]) -> list[JoinCandidate]:
        by_key: dict[tuple[str, str], JoinCandidate] = {}
        for candidate in candidates:
            key = self._dedupe_key(candidate)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = candidate
                continue
            if self._dedupe_priority(candidate) > self._dedupe_priority(existing):
                by_key[key] = self._merge_candidates(candidate, existing)
            else:
                by_key[key] = self._merge_candidates(existing, candidate)
        return list(by_key.values())

    def _dedupe_key(self, candidate: JoinCandidate) -> tuple[str, str]:
        return tuple(sorted([candidate.left, candidate.right]))

    def _dedupe_priority(self, candidate: JoinCandidate) -> tuple[int, float]:
        join_type = str(candidate.metadata.get("join_type") or "")
        return (JOIN_TYPE_PRIORITY.get(join_type, 0), candidate.score)

    def _merge_candidates(
        self,
        primary: JoinCandidate,
        secondary: JoinCandidate,
    ) -> JoinCandidate:
        return JoinCandidate(
            left=primary.left,
            right=primary.right,
            score=primary.score,
            evidence=self._merge_evidence(primary.evidence, secondary.evidence),
            metadata=self._merge_metadata(primary.metadata, secondary.metadata),
        )

    def _merge_metadata(
        self,
        primary: dict[str, object],
        secondary: dict[str, object],
    ) -> dict[str, object]:
        merged = dict(primary)
        for key, value in secondary.items():
            if key == "join_type":
                continue
            if key == "sample_values":
                merged["sample_values"] = self._merge_sample_values(
                    merged.get("sample_values"),
                    value,
                )
                continue
            if key == "warning":
                merged["warning"] = self._merge_warning(merged.get("warning"), value)
                continue
            if key == "requires_verification":
                if primary.get("join_type") == "foreign_key":
                    merged["requires_verification"] = merged.get("requires_verification") is True
                else:
                    merged["requires_verification"] = (
                        merged.get("requires_verification") is True or value is True
                    )
                continue
            if key not in merged or merged[key] is None:
                merged[key] = value
        return merged

    def _merge_evidence(self, primary: list[str], secondary: list[str]) -> list[str]:
        return self._dedupe_strings([*primary, *secondary])

    def _merge_sample_values(self, primary: object, secondary: object) -> list[str]:
        values: list[str] = []
        for candidate_values in [primary, secondary]:
            if not isinstance(candidate_values, list):
                continue
            values.extend(str(value) for value in candidate_values if str(value).strip())
        return self._dedupe_strings(values)[:MAX_SAMPLE_VALUES]

    def _merge_warning(self, primary: object, secondary: object) -> str:
        warnings = [
            str(value).strip()
            for value in [primary, secondary]
            if isinstance(value, str) and value.strip()
        ]
        return "; ".join(self._dedupe_strings(warnings))

    def _dedupe_strings(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result
