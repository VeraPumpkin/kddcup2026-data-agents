from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR


@dataclass(frozen=True, slots=True)
class ParsedColonDuration:
    second_bucket: int
    second_prefix: str


def normalize_identifier(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return re.sub(r"_+", "_", lowered).strip("_")


def parse_colon_duration(value: object) -> ParsedColonDuration | None:
    text = str(value).strip()
    if not text or ":" not in text:
        return None

    parts = text.split(":")
    if len(parts) == 2:
        minutes_text, seconds_text = parts
        if not re.fullmatch(r"\d+", minutes_text):
            return None
        seconds = _parse_seconds_component(seconds_text)
        if seconds is None:
            return None
        total_seconds = Decimal(int(minutes_text)) * Decimal(60) + seconds
    elif len(parts) == 3:
        hours_text, minutes_text, seconds_text = parts
        if not re.fullmatch(r"\d+", hours_text) or not re.fullmatch(r"\d+", minutes_text):
            return None
        minutes = int(minutes_text)
        if minutes > 59:
            return None
        seconds = _parse_seconds_component(seconds_text)
        if seconds is None:
            return None
        total_seconds = (
            Decimal(int(hours_text)) * Decimal(3600)
            + Decimal(minutes) * Decimal(60)
            + seconds
        )
    else:
        return None

    return ParsedColonDuration(
        second_bucket=int(total_seconds.to_integral_value(rounding=ROUND_FLOOR)),
        second_prefix=text.split(".", 1)[0],
    )


def _parse_seconds_component(value: str) -> Decimal | None:
    if not re.fullmatch(r"\d+(?:\.\d+)?", value):
        return None
    seconds = Decimal(value)
    if seconds < 0 or seconds >= 60:
        return None
    return seconds
