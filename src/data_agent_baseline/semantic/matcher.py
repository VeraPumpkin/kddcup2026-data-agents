from __future__ import annotations

import re
from dataclasses import dataclass

from data_agent_baseline.semantic.models import (
    ConceptEntry,
    JoinCandidate,
    SchemaFieldProfile,
    SemanticLink,
    SourceSpan,
    ValueMapping,
)


@dataclass(frozen=True, slots=True)
class MatchingResult:
    concepts: list[ConceptEntry]
    semantic_links: list[SemanticLink]
    debug: dict[str, object]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class QuerySlot:
    name: str
    slot_type: str
    aliases: list[str]
    value_mappings: list[ValueMapping]
    evidence: list[str]
    preferred_tables: list[str]
    preferred_fields: list[str]
    confidence: float


QUESTION_STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "list",
    "me",
    "of",
    "on",
    "or",
    "please",
    "return",
    "show",
    "tell",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "to",
    "what",
    "which",
    "who",
    "with",
}

SYNONYM_EXPANSIONS: dict[str, list[str]] = {
    "sex": ["gender"],
    "gender": ["sex"],
    "patient id": ["id", "identifier"],
    "identifier": ["id"],
    "disease": ["diagnosis", "condition"],
    "diagnosed": ["diagnosis"],
    "diagnosed with": ["diagnosis"],
    "disease diagnosed with": ["diagnosis", "medical condition"],
    "severe": ["serious", "highest", "most serious"],
    "degree": ["level", "severity"],
}
TABLE_NAME_ALIASES: dict[str, list[str]] = {
    "patient": ["patient", "patients"],
    "examination": ["examination", "exam", "medical examination"],
    "laboratory": ["laboratory", "lab"],
}
CHUNK_FIELD_PATTERN = re.compile(r"\*\*(?P<field>[^*]+)\*\*:\s*(?P<body>.+)")
CHUNK_VALUE_PATTERN = re.compile(
    r"with\s+'?(?P<code>[+-]?\d+(?:\.\d+)?)'?\s+(?:indicating|meaning|for)\s+(?P<label>.+)",
    re.I,
)
ORDINAL_QUERY_TERMS: dict[str, set[str]] = {
    "highest": {"highest", "top", "maximum", "max", "largest", "most", "most serious", "most severe"},
    "high": {"high", "higher", "severe", "serious", "advanced", "major"},
    "middle": {"middle", "medium", "moderate", "intermediate"},
    "low": {"low", "lower", "mild", "minor", "light"},
}
ORDINAL_LABEL_TERMS: dict[str, set[str]] = {
    "highest": {"highest", "top", "maximum", "most", "most serious", "most severe", "critical", "worst"},
    "high": {"high", "higher", "severe", "serious", "advanced", "major"},
    "middle": {"middle", "medium", "moderate", "intermediate"},
    "low": {"low", "lower", "mild", "minor", "light"},
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokens(text: str) -> set[str]:
    return {token for token in _normalize(text).split() if len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _phrase_variants(text: str) -> list[str]:
    variants: list[str] = []
    queue = [_normalize(text)]
    seen: set[str] = set()
    while queue:
        candidate = queue.pop(0).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        variants.append(candidate)
        simplified = re.sub(
            r"\b(?:the|a|an|degree|level|patient|patients|record|records|their|all|with|who|that|is|are)\b",
            " ",
            candidate,
        )
        simplified = re.sub(r"\s+", " ", simplified).strip()
        if simplified and simplified != candidate:
            queue.append(simplified)
        for key, expansions in SYNONYM_EXPANSIONS.items():
            if key in candidate:
                for expansion in expansions:
                    queue.append(candidate.replace(key, expansion))
        tokens = [token for token in candidate.split() if token not in QUESTION_STOPWORDS]
        if tokens:
            queue.append(" ".join(tokens))
    return variants[:12]


def _field_terms(profile: SchemaFieldProfile) -> set[str]:
    return (
        _tokens(profile.field_name)
        | _tokens(profile.table_name)
        | {_normalize(tag) for tag in profile.semantic_tags if tag}
    )


def _table_aliases(table_name: str) -> set[str]:
    normalized = _normalize(table_name)
    aliases = {normalized}
    for base_name, candidates in TABLE_NAME_ALIASES.items():
        if base_name in normalized:
            aliases.update(_normalize(item) for item in candidates)
    return aliases


def _concept_terms(concept: ConceptEntry) -> set[str]:
    texts = [concept.canonical_name, *concept.aliases]
    if concept.definition:
        texts.append(concept.definition)
    for mapping in concept.value_mappings:
        texts.append(mapping.source_text)
        texts.extend(mapping.target_column_candidates)
        texts.extend(mapping.target_table_candidates)
    return _tokens(" ".join(texts))


def _chunk_text(chunk: object) -> str:
    text = getattr(chunk, "text", "") or ""
    heading_path = getattr(chunk, "heading_path", []) or []
    if isinstance(heading_path, list):
        text = f"{' '.join(str(item) for item in heading_path)} {text}"
    return str(text)


def _chunk_heading_tables(chunk: object) -> list[str]:
    heading_path = getattr(chunk, "heading_path", []) or []
    if not isinstance(heading_path, list):
        return []
    return [
        str(item)
        for item in heading_path
        if _normalize(str(item)) in TABLE_NAME_ALIASES or any(
            alias in _normalize(str(item))
            for aliases in TABLE_NAME_ALIASES.values()
            for alias in (_normalize(candidate) for candidate in aliases)
        )
    ]


def _extract_chunk_value_mappings(retrieved_chunks: list[object]) -> list[ValueMapping]:
    mappings: list[ValueMapping] = []
    for index, chunk in enumerate(retrieved_chunks):
        heading_tables = _chunk_heading_tables(chunk)
        chunk_text = getattr(chunk, "text", "") or ""
        for raw_line in str(chunk_text).splitlines():
            field_match = CHUNK_FIELD_PATTERN.search(raw_line)
            if field_match is None:
                continue
            field_name = field_match.group("field").strip()
            body = field_match.group("body").strip()
            value_match = CHUNK_VALUE_PATTERN.search(body)
            if value_match is None:
                continue
            mappings.append(
                ValueMapping(
                    source_text=_clean_phrase(value_match.group("label")),
                    normalized_value=value_match.group("code").strip(),
                    target_column_candidates=[field_name],
                    target_table_candidates=heading_tables[:2],
                    evidence=[f"retrieved_chunk_rank={index + 1}", raw_line.strip()[:180]],
                )
            )
    return mappings


def _best_chunk_evidence(
    slot_terms: set[str],
    profile: SchemaFieldProfile,
    retrieved_chunks: list[object],
) -> tuple[float, list[str]]:
    best_score = 0.0
    best_evidence: list[str] = []
    field_name = _normalize(profile.field_name)
    table_aliases = _table_aliases(profile.table_name)
    for chunk in retrieved_chunks:
        chunk_text = _chunk_text(chunk)
        chunk_terms = _tokens(chunk_text)
        overlap = _jaccard(slot_terms, chunk_terms)
        if overlap <= 0:
            continue
        score = overlap * 0.18
        evidence: list[str] = []
        if field_name and field_name in _normalize(chunk_text):
            score += 0.12
            evidence.append(f"retrieved chunk mentions field '{profile.field_name}'")
        if any(alias in _normalize(chunk_text) for alias in table_aliases):
            score += 0.08
            evidence.append(f"retrieved chunk points to table '{profile.table_name}'")
        if score > best_score:
            best_score = score
            best_evidence = evidence
    return best_score, best_evidence


def _mapping_sample_evidence(slot: QuerySlot, profile: SchemaFieldProfile) -> tuple[float, list[str], list[str]]:
    evidence: list[str] = []
    matched_values: list[str] = []
    score = 0.0
    sample_tokens = {_normalize(value) for value in profile.sample_values if value}
    normalized_field = _normalize(profile.field_name)
    normalized_table = _normalize(profile.table_name)
    for mapping in slot.value_mappings:
        if _normalize(mapping.source_text) in sample_tokens:
            score += 0.16
            evidence.append(f"sample value matches phrase '{mapping.source_text}'")
        if _normalize(mapping.normalized_value) in sample_tokens:
            score += 0.28
            evidence.append(f"sample values contain encoded value '{mapping.normalized_value}'")
        if normalized_field in {_normalize(item) for item in mapping.target_column_candidates}:
            score += 0.22
            evidence.append(f"mapping hints column '{profile.field_name}'")
        if normalized_table in {_normalize(item) for item in mapping.target_table_candidates}:
            score += 0.14
            evidence.append(f"mapping hints table '{profile.table_name}'")
        if mapping.normalized_value not in matched_values:
            matched_values.append(mapping.normalized_value)
    return score, evidence, matched_values


def _ordinal_bucket(text: str, *, for_query: bool) -> str | None:
    normalized = _normalize(text)
    if not normalized:
        return None
    mapping = ORDINAL_QUERY_TERMS if for_query else ORDINAL_LABEL_TERMS
    best_bucket: str | None = None
    best_size = 0
    for bucket, phrases in mapping.items():
        for phrase in phrases:
            normalized_phrase = _normalize(phrase)
            if normalized_phrase and normalized_phrase in normalized:
                phrase_size = len(normalized_phrase.split())
                if phrase_size > best_size:
                    best_bucket = bucket
                    best_size = phrase_size
    return best_bucket


def _ordinal_match_score(slot: QuerySlot, profile: SchemaFieldProfile) -> tuple[float, list[str]]:
    if not slot.value_mappings:
        return 0.0, []
    query_bucket = _ordinal_bucket(" ".join([slot.name, *slot.aliases]), for_query=True)
    if query_bucket is None:
        return 0.0, []

    exact_hits = 0
    near_hits = 0
    other_hits = 0
    for mapping in slot.value_mappings:
        label_bucket = _ordinal_bucket(mapping.source_text, for_query=False)
        if label_bucket is None:
            continue
        if label_bucket == query_bucket:
            exact_hits += 1
        elif {label_bucket, query_bucket} == {"high", "highest"}:
            near_hits += 1
        else:
            other_hits += 1

    if exact_hits > 0:
        return 0.24, [f"ordinal label exactly matches query intent '{query_bucket}'"]
    if near_hits > 0:
        return 0.08, [f"ordinal label is adjacent to query intent '{query_bucket}'"]
    if other_hits > 0:
        return -0.12, [f"ordinal label conflicts with query intent '{query_bucket}'"]
    return 0.0, []


def _value_specificity_score(slot: QuerySlot) -> tuple[float, list[str]]:
    if not slot.value_mappings:
        return 0.0, []
    query_bucket = _ordinal_bucket(" ".join([slot.name, *slot.aliases]), for_query=True)
    if query_bucket is None:
        return 0.0, []

    exact = any(_ordinal_bucket(mapping.source_text, for_query=False) == query_bucket for mapping in slot.value_mappings)
    if exact:
        return 0.12, ["slot keeps exact ordinal evidence instead of only stronger/weaker paraphrases"]
    return 0.0, []


def _slot_mapping_alignment(slot_name: str, mapping: ValueMapping) -> float:
    query_bucket = _ordinal_bucket(slot_name, for_query=True)
    label_bucket = _ordinal_bucket(mapping.source_text, for_query=False)
    if query_bucket is None or label_bucket is None:
        return 0.0
    if query_bucket == label_bucket:
        return 0.08
    if {query_bucket, label_bucket} == {"high", "highest"}:
        return 0.01
    return -0.06


def _slot_hint_score(slot: QuerySlot, profile: SchemaFieldProfile) -> tuple[float, list[str]]:
    normalized_field = _normalize(profile.field_name)
    tags = {_normalize(tag) for tag in profile.semantic_tags}
    score = 0.0
    evidence: list[str] = []
    slot_terms = _tokens(slot.name + " " + " ".join(slot.aliases))
    if {"id", "identifier"} & slot_terms:
        if normalized_field == "id" or "identifier" in tags:
            score += 0.32
            evidence.append("output slot looks identifier-like")
    if {"sex", "gender"} & slot_terms and any(token in normalized_field for token in ("sex", "gender")):
        score += 0.34
        evidence.append("output slot looks sex/gender-like")
    if {"disease", "diagnosis", "condition"} & slot_terms and any(
        token in normalized_field for token in ("diagnosis", "disease", "condition")
    ):
        score += 0.34
        evidence.append("output slot looks diagnosis-like")
    if slot.slot_type == "filter":
        if slot.value_mappings and ("coded" in tags or "categorical" in tags):
            score += 0.18
            evidence.append("filter slot prefers coded/categorical fields")
        if {"severity", "severe", "serious"} & slot_terms and any(
            token in normalized_field for token in ("severity", "thrombosis", "grade", "level")
        ):
            score += 0.22
            evidence.append("filter slot mentions severity/degree")
    if slot.preferred_fields and normalized_field in {_normalize(item) for item in slot.preferred_fields}:
        score += 0.26
        evidence.append(f"slot prefers field '{profile.field_name}'")
    if slot.preferred_tables and _normalize(profile.table_name) in {
        _normalize(item) for item in slot.preferred_tables
    }:
        score += 0.2
        evidence.append(f"slot prefers table '{profile.table_name}'")
    return score, evidence


def _dtype_score(slot: QuerySlot, profile: SchemaFieldProfile) -> tuple[float, list[str]]:
    if not slot.value_mappings:
        return 0.0, []
    if profile.dtype in {"int", "float", "bool", "string"}:
        return 0.08, [f"dtype '{profile.dtype}' can support explicit value mapping"]
    return 0.0, []


def _score_slot_profile(
    slot: QuerySlot,
    profile: SchemaFieldProfile,
    retrieved_chunks: list[object],
) -> tuple[float, list[str], list[str], str]:
    evidence: list[str] = []
    score = 0.0

    slot_variants = _phrase_variants(" ".join([slot.name, *slot.aliases]))
    slot_terms = set().union(*(_tokens(item) for item in slot_variants))
    field_terms = _field_terms(profile)

    alias_score = max((_jaccard(_tokens(item), field_terms) for item in slot_variants), default=0.0)
    if alias_score > 0:
        score += alias_score * 0.42
        evidence.append(f"slot/name overlap={alias_score:.2f}")

    contains_score = 0.0
    normalized_field = _normalize(profile.field_name)
    normalized_table = _normalize(profile.table_name)
    for variant in slot_variants:
        if variant == normalized_field:
            contains_score = max(contains_score, 0.32)
        elif variant and variant in normalized_field:
            contains_score = max(contains_score, 0.22)
        elif variant and variant in normalized_table:
            contains_score = max(contains_score, 0.16)
    if contains_score > 0:
        score += contains_score
        evidence.append("slot phrase directly matches schema name")

    hint_score, hint_evidence = _slot_hint_score(slot, profile)
    score += hint_score
    evidence.extend(hint_evidence)

    mapping_score, mapping_evidence, matched_values = _mapping_sample_evidence(slot, profile)
    score += mapping_score
    evidence.extend(mapping_evidence)

    ordinal_score, ordinal_evidence = _ordinal_match_score(slot, profile)
    score += ordinal_score
    evidence.extend(ordinal_evidence)

    specificity_score, specificity_evidence = _value_specificity_score(slot)
    score += specificity_score
    evidence.extend(specificity_evidence)

    dtype_score, dtype_evidence = _dtype_score(slot, profile)
    score += dtype_score
    evidence.extend(dtype_evidence)

    chunk_score, chunk_evidence = _best_chunk_evidence(slot_terms, profile, retrieved_chunks)
    score += chunk_score
    evidence.extend(chunk_evidence)

    if slot.slot_type == "output" and profile.null_ratio < 0.5:
        score += 0.05
        evidence.append("output field has acceptable null ratio")
    if slot.slot_type == "filter" and profile.unique_ratio < 0.4:
        score += 0.05
        evidence.append("filter field looks constraint-friendly")

    if normalized_field == "id" and slot.slot_type == "output":
        score += 0.08
        evidence.append("identifier output usually projects ID directly")

    link_type = "field_link"
    if slot.slot_type == "filter" and matched_values and mapping_score >= 0.18:
        link_type = "value_constraint"
    else:
        matched_values = []
    return min(score, 0.99), evidence[:8], matched_values[:8], link_type


def _extract_output_phrases(question: str) -> list[str]:
    compact = re.sub(r"\s+", " ", question.strip())
    patterns = [
        re.compile(
            r"(?:list|show|return|give|find|get|what\s+(?:is|are))\s+(?P<body>.+?)(?:\bfor\b|\bwith\b|\bwhere\b|\bwhose\b|\?$)",
            re.I,
        ),
        re.compile(r"(?P<body>id,?\s+sex.+?)(?:\bfor\b|\bwith\b|\bwhere\b|\?$)", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(compact)
        if match:
            body = match.group("body")
            parts = re.split(r",|\band\b", body)
            cleaned = [_clean_phrase(part) for part in parts]
            return [item for item in cleaned if item]
    return []


def _extract_filter_phrases(
    question: str,
    concepts: list[ConceptEntry],
    retrieved_chunks: list[object],
) -> list[QuerySlot]:
    slots: list[QuerySlot] = []
    compact = re.sub(r"\s+", " ", question.strip())
    filter_patterns = [
        re.compile(
            r"\bfor\s+(?:patients?|records?)\s+with\s+(?P<body>.+?)(?:,|\b(?:list|show|return|give|get)\b|\?$)",
            re.I,
        ),
        re.compile(r"\b(?:where|whose|having)\b\s+(?P<body>.+?)(?:,|\?$)", re.I),
        re.compile(r"\bwith\s+(?P<body>.+?)(?:,|\b(?:list|show|return|give|get)\b|\?$)", re.I),
    ]
    filter_matches: list[re.Match[str]] = []
    for pattern in filter_patterns:
        filter_matches.extend(pattern.finditer(compact))
    for match in filter_matches:
        body = _clean_phrase(match.group("body"))
        if not body:
            continue
        slots.append(
            QuerySlot(
                name=body,
                slot_type="filter",
                aliases=_phrase_variants(body)[1:5],
                value_mappings=[],
                evidence=[f"question filter phrase: {body}"],
                preferred_tables=[],
                preferred_fields=[],
                confidence=0.72,
            )
        )

    question_terms = _tokens(question)
    chunk_mappings = _extract_chunk_value_mappings(retrieved_chunks)
    base_filter_slots = [slot for slot in slots if slot.slot_type == "filter"]
    for slot in base_filter_slots:
        slot_terms = _tokens(slot.name)
        for mapping in chunk_mappings:
            mapping_terms = _tokens(
                mapping.source_text
                + " "
                + " ".join(mapping.target_column_candidates)
                + " "
                + " ".join(mapping.target_table_candidates)
            )
            if not (mapping_terms & slot_terms or ({"thrombosis"} <= (mapping_terms | slot_terms))):
                continue
            rank_boost = 0.04 if any("retrieved_chunk_rank=1" == item for item in mapping.evidence) else 0.0
            alignment_boost = _slot_mapping_alignment(slot.name, mapping)
            slots.append(
                QuerySlot(
                    name=slot.name,
                    slot_type="filter",
                    aliases=sorted({*slot.aliases, mapping.source_text})[:6],
                    value_mappings=[mapping],
                    evidence=slot.evidence + [f"slot backed off to chunk mapping '{mapping.source_text}'"],
                    preferred_tables=mapping.target_table_candidates[:3] or slot.preferred_tables,
                    preferred_fields=mapping.target_column_candidates[:4] or slot.preferred_fields,
                    confidence=0.84 + rank_boost + alignment_boost,
                )
            )
    for mapping in chunk_mappings:
        mapping_terms = _tokens(mapping.source_text + " " + " ".join(mapping.target_column_candidates))
        if not (mapping_terms & question_terms):
            continue
        evidence = [f"retrieved chunk mapping: {mapping.source_text} -> {mapping.normalized_value}"]
        if mapping.evidence:
            evidence.extend(mapping.evidence[:1])
        slots.append(
            QuerySlot(
                name=mapping.source_text,
                slot_type="filter",
                aliases=_phrase_variants(mapping.source_text)[1:5],
                value_mappings=[mapping],
                evidence=evidence,
                preferred_tables=mapping.target_table_candidates[:3],
                preferred_fields=mapping.target_column_candidates[:4],
                confidence=0.86,
            )
        )
    for concept in concepts:
        matched_mappings = [
            mapping
            for mapping in concept.value_mappings
            if _tokens(mapping.source_text) & question_terms or _normalize(mapping.source_text) in _normalize(question)
        ]
        if not matched_mappings:
            continue
        slots.append(
            QuerySlot(
                name=matched_mappings[0].source_text,
                slot_type="filter",
                aliases=sorted(
                    {
                        concept.canonical_name,
                        *concept.aliases,
                        *[mapping.source_text for mapping in matched_mappings],
                    }
                )[:6],
                value_mappings=matched_mappings,
                evidence=[f"knowledge mapping triggered by concept '{concept.canonical_name}'"],
                preferred_tables=sorted(
                    {
                        table
                        for mapping in matched_mappings
                        for table in mapping.target_table_candidates
                    }
                )[:4],
                preferred_fields=sorted(
                    {
                        field
                        for mapping in matched_mappings
                        for field in mapping.target_column_candidates
                    }
                )[:6],
                confidence=min(0.9, concept.confidence + 0.2),
            )
        )
    return slots


def _clean_phrase(text: str) -> str:
    cleaned = re.sub(r"^(?:their|the|all|each)\b", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\b(?:that|which|who)\b.*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,.")


def _output_slot_from_phrase(phrase: str) -> QuerySlot:
    aliases = _phrase_variants(phrase)[1:5]
    preferred_fields: list[str] = []
    if {"id", "identifier"} & _tokens(phrase):
        preferred_fields.append("ID")
    if {"sex", "gender"} & _tokens(phrase):
        preferred_fields.extend(["SEX", "Gender"])
    if {"diagnosis", "disease", "condition"} & _tokens(phrase):
        preferred_fields.extend(["Diagnosis", "Disease"])
    return QuerySlot(
        name=phrase,
        slot_type="output",
        aliases=aliases,
        value_mappings=[],
        evidence=[f"question output phrase: {phrase}"],
        preferred_tables=[],
        preferred_fields=preferred_fields,
        confidence=0.76,
    )


def _dedupe_slots(slots: list[QuerySlot]) -> list[QuerySlot]:
    deduped: dict[tuple[str, str], QuerySlot] = {}
    for slot in slots:
        key = (slot.slot_type, _normalize(slot.name))
        existing = deduped.get(key)
        if existing is None or (
            slot.confidence,
            len(slot.value_mappings),
            _value_specificity_score(slot)[0],
        ) > (
            existing.confidence,
            len(existing.value_mappings),
            _value_specificity_score(existing)[0],
        ):
            deduped[key] = slot
    return sorted(deduped.values(), key=lambda item: (-item.confidence, item.slot_type, item.name))


def _build_query_specific_concepts(base_concepts: list[ConceptEntry], slots: list[QuerySlot]) -> list[ConceptEntry]:
    query_concepts = [
        ConceptEntry(
            canonical_name=slot.name,
            aliases=slot.aliases,
            definition=f"query {slot.slot_type} slot extracted from the current question",
            source_span=SourceSpan(path="question", excerpt=slot.name),
            constraints=[],
            value_mappings=slot.value_mappings,
            confidence=slot.confidence,
        )
        for slot in slots
    ]
    seen = {_normalize(concept.canonical_name) for concept in query_concepts}
    for concept in base_concepts:
        if _normalize(concept.canonical_name) in seen:
            continue
        query_concepts.append(concept)
    return query_concepts


def _extract_query_slots(
    question: str,
    concepts: list[ConceptEntry],
    retrieved_chunks: list[object],
) -> list[QuerySlot]:
    slots: list[QuerySlot] = []
    question_terms = _tokens(question)
    output_table_hint = ["Patient"] if {"patient", "patients"} & question_terms else []
    for phrase in _extract_output_phrases(question):
        slot = _output_slot_from_phrase(phrase)
        if output_table_hint:
            slot = QuerySlot(
                name=slot.name,
                slot_type=slot.slot_type,
                aliases=slot.aliases,
                value_mappings=slot.value_mappings,
                evidence=slot.evidence,
                preferred_tables=output_table_hint,
                preferred_fields=slot.preferred_fields,
                confidence=slot.confidence,
            )
        slots.append(slot)
    slots.extend(_extract_filter_phrases(question, concepts, retrieved_chunks))

    for concept in concepts:
        concept_overlap = _jaccard(question_terms, _concept_terms(concept))
        if concept_overlap < 0.12:
            continue
        if concept.value_mappings:
            slots.append(
                QuerySlot(
                    name=concept.canonical_name,
                    slot_type="filter",
                    aliases=sorted(set(concept.aliases))[:5],
                    value_mappings=concept.value_mappings,
                    evidence=[f"question overlaps knowledge concept '{concept.canonical_name}'"],
                    preferred_tables=sorted(
                        {
                            table
                            for mapping in concept.value_mappings
                            for table in mapping.target_table_candidates
                        }
                    )[:4],
                    preferred_fields=sorted(
                        {
                            field
                            for mapping in concept.value_mappings
                            for field in mapping.target_column_candidates
                        }
                    )[:6],
                    confidence=min(0.88, concept.confidence + concept_overlap * 0.3),
                )
            )
        elif any(token in question_terms for token in _tokens(concept.canonical_name)):
            slots.append(
                QuerySlot(
                    name=concept.canonical_name,
                    slot_type="output",
                    aliases=sorted(set(concept.aliases))[:5],
                    value_mappings=[],
                    evidence=[f"question overlaps knowledge concept '{concept.canonical_name}'"],
                    preferred_tables=[],
                    preferred_fields=[],
                    confidence=min(0.72, concept.confidence + concept_overlap * 0.2),
                )
            )
    return _dedupe_slots(slots)


def _select_slot_links(
    slot: QuerySlot,
    schema_profiles: list[SchemaFieldProfile],
    retrieved_chunks: list[object],
) -> tuple[list[SemanticLink], list[dict[str, object]]]:
    candidates: list[tuple[float, SemanticLink, dict[str, object]]] = []
    for profile in schema_profiles:
        score, evidence, matched_values, link_type = _score_slot_profile(slot, profile, retrieved_chunks)
        candidate = SemanticLink(
            concept_name=slot.name,
            matched_table=profile.table_name,
            matched_field=profile.field_name,
            matched_values=matched_values,
            link_type=link_type,
            slot_name=slot.name,
            slot_type=slot.slot_type,
            confidence=round(score, 4),
            evidence=(slot.evidence + evidence)[:8],
        )
        candidates.append(
            (
                candidate.confidence,
                candidate,
                {
                    "slot_name": slot.name,
                    "slot_type": slot.slot_type,
                    "table_name": profile.table_name,
                    "field_name": profile.field_name,
                    "score": candidate.confidence,
                    "matched_values": matched_values,
                    "link_type": link_type,
                    "evidence": candidate.evidence,
                },
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1].matched_table, item[1].matched_field))
    threshold = 0.34 if slot.slot_type == "filter" else 0.28
    accepted = [candidate for score, candidate, _ in candidates if score >= threshold][:2]
    return accepted, [debug for _, _, debug in candidates[:10]]


def _build_join_links(
    semantic_links: list[SemanticLink],
    join_candidates: list[JoinCandidate],
) -> list[SemanticLink]:
    if not semantic_links or not join_candidates:
        return []
    filter_tables = {link.matched_table for link in semantic_links if link.slot_type == "filter"}
    output_tables = {link.matched_table for link in semantic_links if link.slot_type == "output"}
    if not filter_tables or not output_tables:
        return []

    join_links: list[SemanticLink] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()
    for filter_table in sorted(filter_tables):
        for output_table in sorted(output_tables):
            if filter_table == output_table:
                continue
            candidates = [
                join
                for join in join_candidates
                if {join.left_table, join.right_table} == {filter_table, output_table}
            ]
            if not candidates:
                continue
            best = sorted(candidates, key=lambda item: (-item.score, item.left_field, item.right_field))[0]
            join_key = (best.left_table, best.left_field, best.right_table, best.right_field)
            if join_key in seen_pairs:
                continue
            seen_pairs.add(join_key)
            join_links.append(
                SemanticLink(
                    concept_name=f"{filter_table} to {output_table}",
                    matched_table=f"{best.left_table}<->{best.right_table}",
                    matched_field=f"{best.left_field}={best.right_field}",
                    matched_values=[best.left_field, best.right_field],
                    link_type="join_path_hint",
                    slot_name="join path",
                    slot_type="join",
                    confidence=round(min(best.score + 0.04, 0.99), 4),
                    evidence=[best.reason, "join candidate connects filter/output tables"],
                )
            )
    return join_links


def match_concepts_to_schema(
    *,
    question: str,
    concepts: list[ConceptEntry],
    schema_profiles: list[SchemaFieldProfile],
    join_candidates: list[JoinCandidate],
    retrieved_chunks: list[object] | None = None,
) -> MatchingResult:
    effective_chunks = list(retrieved_chunks or [])
    warnings: list[str] = []
    candidate_debug: list[dict[str, object]] = []

    slots = _extract_query_slots(question, concepts, effective_chunks)
    query_concepts = _build_query_specific_concepts(concepts, slots)
    semantic_links: list[SemanticLink] = []

    for slot in slots:
        accepted, debug_rows = _select_slot_links(slot, schema_profiles, effective_chunks)
        candidate_debug.extend(debug_rows)
        if not accepted:
            warnings.append(f"No executable schema link found for {slot.slot_type} slot '{slot.name}'.")
            continue
        semantic_links.extend(accepted)

    semantic_links.extend(_build_join_links(semantic_links, join_candidates))
    semantic_links.sort(
        key=lambda item: (
            -item.confidence,
            item.slot_type or "",
            item.concept_name,
            item.matched_table,
            item.matched_field,
        )
    )

    return MatchingResult(
        concepts=query_concepts,
        semantic_links=semantic_links,
        debug={
            "query_slots": [
                {
                    "name": slot.name,
                    "slot_type": slot.slot_type,
                    "aliases": slot.aliases,
                    "preferred_tables": slot.preferred_tables,
                    "preferred_fields": slot.preferred_fields,
                    "value_mappings": [
                        {
                            "source_text": mapping.source_text,
                            "normalized_value": mapping.normalized_value,
                            "target_column_candidates": mapping.target_column_candidates,
                            "target_table_candidates": mapping.target_table_candidates,
                        }
                        for mapping in slot.value_mappings
                    ],
                    "evidence": slot.evidence,
                }
                for slot in slots
            ],
            "candidate_scores": candidate_debug,
        },
        warnings=warnings,
    )
