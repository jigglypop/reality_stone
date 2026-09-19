"""Fail-closed local canonical receipt for the E1 metadata-only contract.

This module deliberately has no AllenSDK, network, dataframe, or file I/O
dependency.  It accepts already materialised mappings and emits only the two
canonical byte streams allowed by the frozen provenance contract.  In
particular, it cannot request units, channels, probes, NWB, LFP, spikes, or
any scientific endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
import unicodedata


SPLIT_SALT = b"CE-E1-NEUROPIXELS-V1"
_REQUIRED_FIELDS = (
    "ecephys_session_id",
    "specimen_id",
    "session_type",
    "date_of_acquisition",
)
_RFC3339_WITH_OFFSET = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
_RFC3339_NAIVE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?$")


class E1MetadataReceiptError(ValueError):
    """A named, fail-closed metadata-apparatus rejection."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True)
class ReceiptStatus:
    """Validation state, kept separate from canonical hashed byte streams."""

    code: str
    accepted: bool
    detail: str
    forbidden_resources_opened: bool = False


@dataclass(frozen=True)
class E1MetadataReceipt:
    """The complete local output permitted by the metadata-only contract."""

    status: ReceiptStatus
    canonical_table_jsonl: bytes
    assignment_jsonl: bytes
    canonical_table_sha256: str
    assignment_sha256: str
    input_row_count: int
    canonical_row_count: int
    byte_identical_duplicate_count: int
    specimen_count: int
    split_counts: tuple[tuple[str, int], ...]


def _fail(code: str, detail: str = "") -> None:
    raise E1MetadataReceiptError(code, detail)


def canonical_decimal_id(value: object, *, field: str) -> str:
    """Return an ASCII decimal identifier from a built-in nonnegative ``int``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("E1_FIELD_TYPE_INVALID", field)
    return str(value)


def canonical_session_type(value: object) -> str:
    """NFC-normalise an allowed session type without silently repairing controls."""
    if not isinstance(value, str):
        _fail("E1_FIELD_TYPE_INVALID", "session_type")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or any(unicodedata.category(char) == "Cc" for char in normalized):
        _fail("E1_FIELD_TYPE_INVALID", "session_type")
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as error:
        raise E1MetadataReceiptError("E1_FIELD_TYPE_INVALID", "session_type") from error
    return normalized


def canonical_utc_timestamp(value: object) -> str:
    """Parse the frozen RFC3339 subset and emit a UTC microsecond instant."""
    if not isinstance(value, str):
        _fail("E1_FIELD_TYPE_INVALID", "date_of_acquisition")
    if _RFC3339_NAIVE.fullmatch(value):
        _fail("E1_TIMEZONE_UNRESOLVED", "date_of_acquisition")
    if not _RFC3339_WITH_OFFSET.fullmatch(value):
        _fail("E1_FIELD_TYPE_INVALID", "date_of_acquisition")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise E1MetadataReceiptError("E1_FIELD_TYPE_INVALID", "date_of_acquisition") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail("E1_TIMEZONE_UNRESOLVED", "date_of_acquisition")
    utc = parsed.astimezone(timezone.utc)
    return f"{utc:%Y-%m-%dT%H:%M:%S}.{utc.microsecond:06d}Z"


def canonical_core_row(row: Mapping[str, Any]) -> dict[str, str]:
    """Validate one response row and retain exactly the four allowed keys."""
    if not isinstance(row, Mapping):
        _fail("E1_FIELD_TYPE_INVALID", "row")
    missing = [field for field in _REQUIRED_FIELDS if field not in row]
    if missing:
        _fail("E1_REQUIRED_FIELD_MISSING", ",".join(missing))
    return {
        "date_of_acquisition": canonical_utc_timestamp(row["date_of_acquisition"]),
        "ecephys_session_id": canonical_decimal_id(
            row["ecephys_session_id"], field="ecephys_session_id"
        ),
        "session_type": canonical_session_type(row["session_type"]),
        "specimen_id": canonical_decimal_id(row["specimen_id"], field="specimen_id"),
    }


def _compact_json_line(record: Mapping[str, str]) -> bytes:
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"


def specimen_uint64(specimen_id: object) -> int:
    """First eight SHA-256 bytes of the exact salted ASCII ID, in big-endian order."""
    encoded = canonical_decimal_id(specimen_id, field="specimen_id").encode("ascii")
    digest = hashlib.sha256(SPLIT_SALT + b":" + encoded).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def specimen_split(specimen_id: object) -> str:
    """Allocate a specimen using only the exact integer contract thresholds."""
    value = specimen_uint64(specimen_id)
    if value < (1 << 63):
        return "calibration"
    if value < (3 << 62):
        return "development"
    return "held_out"


def build_local_metadata_receipt(rows: Iterable[Mapping[str, Any]]) -> E1MetadataReceipt:
    """Build the local receipt, rejecting every unsupported schema variation.

    Byte-identical *canonical* repeats are deduplicated.  Any same-session
    conflict, including a changed specimen, timestamp, or session type, stops
    immediately with ``E1_INCONSISTENT_DUPLICATE_SESSION``.
    """
    by_session: dict[str, bytes] = {}
    input_row_count = 0
    duplicate_count = 0
    for row in rows:
        input_row_count += 1
        canonical = canonical_core_row(row)
        line = _compact_json_line(canonical)
        session_id = canonical["ecephys_session_id"]
        previous = by_session.get(session_id)
        if previous is None:
            by_session[session_id] = line
        elif previous == line:
            duplicate_count += 1
        else:
            _fail("E1_INCONSISTENT_DUPLICATE_SESSION", session_id)

    ordered_ids = sorted(by_session, key=int)
    table = b"".join(by_session[session_id] for session_id in ordered_ids)
    assignments: list[bytes] = []
    split_counts = {"calibration": 0, "development": 0, "held_out": 0}
    specimens: set[str] = set()
    for session_id in ordered_ids:
        canonical = json.loads(by_session[session_id])
        specimen_id = canonical["specimen_id"]
        split = specimen_split(int(specimen_id))
        assignments.append(
            _compact_json_line(
                {
                    "ecephys_session_id": session_id,
                    "specimen_id": specimen_id,
                    "split": split,
                }
            )
        )
        specimens.add(specimen_id)

    # Count specimens rather than sessions: multiple sessions from one specimen
    # must not inflate an allocation count.
    for specimen_id in specimens:
        split_counts[specimen_split(int(specimen_id))] += 1
    assignment = b"".join(assignments)
    return E1MetadataReceipt(
        status=ReceiptStatus(
            code="E1_LOCAL_METADATA_RECEIPT_VALID",
            accepted=True,
            detail=(
                "local canonicalization only; no unit/channel/probe/NWB/LFP/spike/"
                "covariance/dimension/model-score resource can be opened by this core"
            ),
            forbidden_resources_opened=False,
        ),
        canonical_table_jsonl=table,
        assignment_jsonl=assignment,
        canonical_table_sha256=hashlib.sha256(table).hexdigest(),
        assignment_sha256=hashlib.sha256(assignment).hexdigest(),
        input_row_count=input_row_count,
        canonical_row_count=len(ordered_ids),
        byte_identical_duplicate_count=duplicate_count,
        specimen_count=len(specimens),
        split_counts=tuple((name, split_counts[name]) for name in ("calibration", "development", "held_out")),
    )
