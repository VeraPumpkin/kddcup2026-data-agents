from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_chars: int = 900
    overlap_chars: int = 120


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    chunk_id: str
    text: str
    heading_path: list[str]
    start_line: int
    end_line: int
    token_count_estimate: int


def _estimate_token_count(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def _line_number_map(text: str) -> list[tuple[int, str]]:
    return [(index, line.rstrip("\n")) for index, line in enumerate(text.splitlines(), start=1)]


def _heading_level(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
    if match is None:
        return None
    return len(match.group(1)), match.group(2).strip()


def _build_sections(numbered_lines: list[tuple[int, str]]) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[tuple[int, str]] = []
    current_start = 1

    def flush_section(end_line: int) -> None:
        nonlocal current_lines, current_start
        if not current_lines:
            return
        text = "\n".join(line for _, line in current_lines).strip()
        if text:
            sections.append(
                {
                    "heading_path": [item[1] for item in heading_stack],
                    "start_line": current_start,
                    "end_line": end_line,
                    "text": text,
                }
            )
        current_lines = []

    for line_number, line in numbered_lines:
        heading = _heading_level(line)
        if heading is not None:
            flush_section(line_number - 1)
            level, title = heading
            heading_stack = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, title))
            current_start = line_number
            current_lines = [(line_number, line)]
            continue
        if not current_lines:
            current_start = line_number
        current_lines.append((line_number, line))

    flush_section(numbered_lines[-1][0] if numbered_lines else 1)
    return sections


def _split_section_text(
    text: str,
    *,
    target_chars: int,
    overlap_chars: int,
) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= target_chars:
        return [normalized]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= target_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            overlap_text = current[-overlap_chars:].strip()
            current = f"{overlap_text}\n\n{paragraph}".strip() if overlap_text else paragraph
        else:
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + target_chars)
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    break
                start = max(0, end - overlap_chars)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def chunk_markdown(
    knowledge_md_path: Path,
    *,
    config: ChunkingConfig,
) -> list[KnowledgeChunk]:
    text = knowledge_md_path.read_text(encoding="utf-8", errors="replace")
    numbered_lines = _line_number_map(text)
    sections = _build_sections(numbered_lines)

    chunks: list[KnowledgeChunk] = []
    chunk_index = 0
    for section in sections:
        section_text = str(section["text"])
        heading_path = [str(item) for item in section["heading_path"]]
        start_line = int(section["start_line"])
        end_line = int(section["end_line"])
        for piece in _split_section_text(
            section_text,
            target_chars=config.target_chars,
            overlap_chars=config.overlap_chars,
        ):
            chunk_index += 1
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"chunk_{chunk_index:04d}",
                    text=piece,
                    heading_path=heading_path,
                    start_line=start_line,
                    end_line=end_line,
                    token_count_estimate=_estimate_token_count(piece),
                )
            )
    return chunks
