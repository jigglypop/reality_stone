"""Deterministic, structural-only C. elegans connectome replay utilities."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


FORMAT = "clarus.connectome_structural_replay.v1"
_HEADER = ["Source", "Target", "Weight", "Type"]
_CLASSES = {"chemical", "electrical"}
_IDENTIFIER = re.compile(r"[A-Za-z0-9]+\Z")
_WEIGHT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_METRIC_KEYS = {
    "source_byte_length", "source_observation_count", "normalized_endpoint_identifier_count",
    "self_loop_observation_count", "chemical_observation_count", "chemical_released_weight_sum",
    "electrical_observation_count", "electrical_released_weight_sum", "total_released_weight_sum",
    "electrical_unordered_pair_count", "electrical_reciprocal_two_row_pair_count",
    "electrical_unequal_reciprocal_weight_pair_count", "electrical_max_observations_per_pair",
    "exact_duplicate_released_row_count",
}


class ReplayValidationError(ValueError):
    """Raised when authority bytes, manifest, or replay values are invalid."""


@dataclass(frozen=True)
class Observation:
    connection_class: str
    source_id: str
    target_id: str
    released_weight: int
    source_record_ordinal: int

    def __post_init__(self) -> None:
        if self.connection_class not in _CLASSES:
            raise ReplayValidationError("connection_class must be chemical or electrical")
        _validate_identifier(self.source_id)
        _validate_identifier(self.target_id)
        _validate_exact_nonnegative_int(self.released_weight, "released_weight")
        _validate_exact_nonnegative_int(self.source_record_ordinal, "source_record_ordinal")


def _validate_exact_nonnegative_int(value: Any, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ReplayValidationError(f"{name} must be a nonnegative int (not bool)")


def _normalize_identifier(value: str) -> str:
    if type(value) is not str:
        raise ReplayValidationError("identifier must be a string")
    # Deliberately remove only ASCII 0x20, never Unicode whitespace.
    normalized = value.strip(" ")
    _validate_identifier(normalized)
    return normalized


def _validate_identifier(value: str) -> None:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ReplayValidationError("identifier must be nonempty ASCII alphanumeric")


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ReplayValidationError(f"{label} keys must be exactly {sorted(expected)}")


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the v1 preregistration manifest."""
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ReplayValidationError("manifest BOM is forbidden")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayValidationError("manifest must be strict UTF-8 JSON") from exc
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: Any) -> None:
    root_keys = {"schema_version", "dataset_id", "scope", "source", "population", "parser",
                 "electrical_weight_semantics", "expected_source_metrics"}
    _require_keys(manifest, root_keys, "manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ReplayValidationError("schema_version must be exact integer 1")
    exact_strings = {
        "dataset_id": "openworm_celegansneuroml_herm_full_edgelist",
        "scope": "adult_hermaphrodite_structural_graph_only",
        "population": "all_normalized_endpoint_identifiers_in_frozen_file_including_non_neuron_cells",
        "electrical_weight_semantics": "sum_of_released_row_weights_not_physical_gap_junction_count",
    }
    for key, expected in exact_strings.items():
        if manifest[key] != expected:
            raise ReplayValidationError(f"manifest {key} is not the registered value")
    source = manifest["source"]
    _require_keys(source, {"url", "commit", "path", "byte_length", "sha256", "redistribution_permission", "repository_handling"}, "source")
    if type(source["byte_length"]) is not int or source["byte_length"] < 0:
        raise ReplayValidationError("source byte_length must be a nonnegative exact integer")
    if type(source["sha256"]) is not str or _SHA256.fullmatch(source["sha256"]) is None:
        raise ReplayValidationError("source sha256 must be lowercase hexadecimal")
    if source["redistribution_permission"] != "not_established" or source["repository_handling"] != "manifest_only_raw_and_full_output_run_local":
        raise ReplayValidationError("source handling status is not registered")
    if not all(type(source[key]) is str and source[key] for key in ("url", "commit", "path")):
        raise ReplayValidationError("source url, commit, and path must be nonempty strings")
    parser = manifest["parser"]
    _require_keys(parser, {"header", "utf8", "bom", "classes", "endpoint_identifier", "weight_grammar", "self_loops", "ordinal_policy", "duplicate_identity"}, "parser")
    if parser != {
        "header": _HEADER, "utf8": "strict", "bom": "forbidden", "classes": ["chemical", "electrical"],
        "endpoint_identifier": "ascii_alphanumeric_after_leading_trailing_ascii_space_removal",
        "weight_grammar": "0|[1-9][0-9]*", "self_loops": "accepted_and_preserved",
        "ordinal_policy": "r1_zero_based_csv_data_record_position_preserved_and_complete",
        "duplicate_identity": "source_record_ordinal",
    }:
        raise ReplayValidationError("parser profile is not the registered strict profile")
    metrics = manifest["expected_source_metrics"]
    _require_keys(metrics, _METRIC_KEYS, "expected_source_metrics")
    for key, value in metrics.items():
        _validate_exact_nonnegative_int(value, f"expected_source_metrics.{key}")
    if metrics["source_byte_length"] != source["byte_length"]:
        raise ReplayValidationError("source byte length disagrees with expected metrics")


def verify_source_bytes(raw: bytes, manifest: Mapping[str, Any]) -> None:
    """Verify raw authority bytes before decoding or CSV parsing."""
    if type(raw) is not bytes:
        raise ReplayValidationError("source must be bytes")
    source = manifest["source"]
    if len(raw) != source["byte_length"]:
        raise ReplayValidationError("source byte length mismatch")
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ReplayValidationError("source sha256 mismatch")


def parse_source_bytes(raw: bytes, manifest: Mapping[str, Any]) -> list[Observation]:
    """Hash-check, then strictly decode and parse a frozen source CSV."""
    verify_source_bytes(raw, manifest)
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ReplayValidationError("source BOM is forbidden")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ReplayValidationError("source is not strict UTF-8") from exc
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise ReplayValidationError("source CSV is malformed") from exc
    if not rows or rows[0] != _HEADER:
        raise ReplayValidationError("source header must exactly match registered header")
    observations: list[Observation] = []
    for ordinal, row in enumerate(rows[1:]):
        if len(row) != 4:
            raise ReplayValidationError("every source CSV row must have exactly four fields")
        source_id, target_id, weight_text, connection_class = row
        source_id = _normalize_identifier(source_id)
        target_id = _normalize_identifier(target_id)
        if connection_class not in _CLASSES:
            raise ReplayValidationError("source connection class is not allowed")
        if _WEIGHT.fullmatch(weight_text) is None:
            raise ReplayValidationError("source weight is not a canonical nonnegative decimal")
        observations.append(Observation(connection_class, source_id, target_id, int(weight_text), ordinal))
    validate_observations(observations)
    return observations


def validate_observations(observations: Iterable[Observation]) -> list[Observation]:
    """Validate the R1 complete ordinal domain without renumbering records."""
    result = list(observations)
    for observation in result:
        if not isinstance(observation, Observation):
            raise ReplayValidationError("observations must be Observation values")
    ordinals = [item.source_record_ordinal for item in result]
    if set(ordinals) != set(range(len(result))):
        raise ReplayValidationError("source record ordinals must be complete, unique R1 ordinals")
    return result


def _endpoints(observation: Observation) -> tuple[str, str]:
    if observation.connection_class == "chemical":
        return observation.source_id, observation.target_id
    return min(observation.source_id, observation.target_id), max(observation.source_id, observation.target_id)


def build_artifact(observations: Iterable[Observation], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize validated observations into the structural-only v1 artifact."""
    _validate_manifest(dict(manifest))
    records = validate_observations(observations)
    observation_values = []
    for item in records:
        endpoint_a, endpoint_b = _endpoints(item)
        observation_values.append({"connection_class": item.connection_class, "endpoint_a": endpoint_a,
            "endpoint_b": endpoint_b, "source_id": item.source_id, "target_id": item.target_id,
            "released_weight": item.released_weight, "source_record_ordinal": item.source_record_ordinal})
    observation_values.sort(key=lambda item: (item["connection_class"], item["endpoint_a"], item["endpoint_b"], item["source_record_ordinal"]))
    nodes = sorted({item.source_id for item in records} | {item.target_id for item in records})
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for value in observation_values:
        groups[(value["connection_class"], value["endpoint_a"], value["endpoint_b"])].append(value)
    connections = []
    for key in sorted(groups):
        values = groups[key]
        connections.append({"connection_class": key[0], "endpoint_a": key[1], "endpoint_b": key[2],
            "released_observation_count": len(values), "released_weight_sum": sum(value["released_weight"] for value in values),
            "source_record_ordinals": [value["source_record_ordinal"] for value in values]})
    summary = _summary(records, observation_values, connections)
    source = manifest["source"]
    return {"format": FORMAT, "metadata": {"dataset_id": manifest["dataset_id"], "scope": "structural_graph_only",
        "source_url": source["url"], "source_commit": source["commit"], "source_path": source["path"],
        "source_byte_length": source["byte_length"], "source_sha256": source["sha256"],
        "redistribution_permission": "not_established", "population": manifest["population"],
        "electrical_weight_semantics": manifest["electrical_weight_semantics"]},
        "nodes": [{"id": node} for node in nodes], "observations": observation_values,
        "connections": connections, "summary": summary}


def _summary(records: list[Observation], observations: list[dict[str, Any]], connections: list[dict[str, Any]]) -> dict[str, int]:
    chemical = [item for item in records if item.connection_class == "chemical"]
    electrical = [item for item in records if item.connection_class == "electrical"]
    electrical_groups = [item for item in connections if item["connection_class"] == "electrical"]
    reciprocal = 0
    unequal = 0
    for group in electrical_groups:
        ordinal_set = set(group["source_record_ordinals"])
        group_rows = [item for item in observations if item["source_record_ordinal"] in ordinal_set]
        if len(group_rows) == 2 and group_rows[0]["source_id"] == group_rows[1]["target_id"] and group_rows[0]["target_id"] == group_rows[1]["source_id"]:
            reciprocal += 1
            if group_rows[0]["released_weight"] != group_rows[1]["released_weight"]:
                unequal += 1
    released_rows = [(item.connection_class, item.source_id, item.target_id, item.released_weight) for item in records]
    return {"node_count": len({item.source_id for item in records} | {item.target_id for item in records}),
        "canonical_observation_count": len(records), "aggregate_connection_count": len(connections),
        "chemical_observation_count": len(chemical), "chemical_released_weight_sum": sum(item.released_weight for item in chemical),
        "electrical_observation_count": len(electrical), "electrical_released_weight_sum": sum(item.released_weight for item in electrical),
        "total_released_weight_sum": sum(item.released_weight for item in records),
        "self_loop_observation_count": sum(item.source_id == item.target_id for item in records),
        "electrical_unordered_pair_count": len(electrical_groups), "electrical_reciprocal_two_row_pair_count": reciprocal,
        "electrical_unequal_reciprocal_weight_pair_count": unequal,
        "electrical_max_observations_per_pair": max((item["released_observation_count"] for item in electrical_groups), default=0),
        "exact_duplicate_released_row_count": len(released_rows) - len(set(released_rows))}


def canonical_bytes(artifact: Mapping[str, Any]) -> bytes:
    """Serialize canonical JSON with recursive key sorting and one terminal LF."""
    try:
        text = json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ReplayValidationError("artifact is not finite canonical JSON") from exc
    return text.encode("utf-8") + b"\n"


def artifact_sha256(artifact: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(artifact)).hexdigest()


def validate_expected_metrics(artifact: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Compare all registered source metrics against a replay artifact exactly."""
    summary = artifact["summary"]
    expected = manifest["expected_source_metrics"]
    actual = {"source_byte_length": artifact["metadata"]["source_byte_length"],
        "source_observation_count": summary["canonical_observation_count"],
        "normalized_endpoint_identifier_count": summary["node_count"]}
    actual.update({key: summary[key] for key in _METRIC_KEYS - set(actual)})
    for key in _METRIC_KEYS:
        if actual[key] != expected[key]:
            raise ReplayValidationError(f"registered metric mismatch: {key}")


def replay_source_file(manifest_path: str | Path, source_path: str | Path) -> tuple[dict[str, Any], str]:
    manifest = load_manifest(manifest_path)
    observations = parse_source_bytes(Path(source_path).read_bytes(), manifest)
    artifact = build_artifact(observations, manifest)
    validate_expected_metrics(artifact, manifest)
    return artifact, artifact_sha256(artifact)
