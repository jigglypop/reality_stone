"""Reproducible gates for the minimum-life/protocell research track.

The empirical layer deliberately separates three questions: autocatalytic
amplification, compartment-dependent persistence, and heritable lineage
change.  Passing them supports a minimal *protocell-like experimental system*;
it does not identify the historical first organism or prove that every living
system must use the same chemistry.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import statistics
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence


_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_index(cell_reference: str) -> int:
    match = re.match(r"([A-Z]+)", cell_reference.upper())
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {cell_reference}")
    value = 0
    for letter in match.group(1):
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t"))
        for item in root.iter(f"{{{_MAIN_NS}}}si")
    ]


def xlsx_sheet_names(path: str | Path) -> tuple[str, ...]:
    """Return workbook sheet names using only the Python standard library."""

    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    return tuple(
        sheet.attrib["name"]
        for sheet in workbook.iter(f"{{{_MAIN_NS}}}sheet")
    )


def read_xlsx_sheet(path: str | Path, sheet_name: str) -> tuple[tuple[Any, ...], ...]:
    """Read cached values from one XLSX worksheet without optional dependencies."""

    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships.iter(f"{{{_PKG_REL_NS}}}Relationship")
        }
        sheet_target: str | None = None
        for sheet in workbook.iter(f"{{{_MAIN_NS}}}sheet"):
            if sheet.attrib["name"] == sheet_name:
                relation_id = sheet.attrib[f"{{{_REL_NS}}}id"]
                sheet_target = targets[relation_id]
                break
        if sheet_target is None:
            raise KeyError(f"workbook has no sheet {sheet_name!r}")
        if sheet_target.startswith("/"):
            worksheet_path = sheet_target.lstrip("/")
        else:
            worksheet_path = f"xl/{sheet_target.lstrip('/')}"
        worksheet_path = str(Path(worksheet_path).as_posix())
        root = ET.fromstring(archive.read(worksheet_path))
        strings = _shared_strings(archive)

        rows: list[tuple[Any, ...]] = []
        for row in root.iter(f"{{{_MAIN_NS}}}row"):
            values: dict[int, Any] = {}
            for cell in row.iter(f"{{{_MAIN_NS}}}c"):
                reference = cell.attrib.get("r")
                if reference is None:
                    continue
                column = _column_index(reference)
                value_node = cell.find(f"{{{_MAIN_NS}}}v")
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or ""
                        for node in cell.iter(f"{{{_MAIN_NS}}}t")
                    )
                elif value_node is None:
                    value = None
                elif cell_type == "s":
                    value = strings[int(value_node.text or "0")]
                elif cell_type in {"str", "e"}:
                    value = value_node.text or ""
                elif cell_type == "b":
                    value = value_node.text == "1"
                else:
                    raw = value_node.text or ""
                    try:
                        numeric = float(raw)
                        value = int(numeric) if numeric.is_integer() else numeric
                    except ValueError:
                        value = raw
                values[column] = value
            if values:
                width = max(values) + 1
                rows.append(tuple(values.get(column) for column in range(width)))
        return tuple(rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected numeric source-data cell, got {value!r}")
    return float(value)


def _slice_numbers(row: Sequence[Any], start: int, stop: int) -> tuple[float, ...]:
    return tuple(_number(value) for value in row[start:stop])


def _exact_median_permutation_p(
    control: Sequence[float], treatment: Sequence[float]
) -> float:
    """One-sided exact permutation p-value for a median difference."""

    control = tuple(float(value) for value in control)
    treatment = tuple(float(value) for value in treatment)
    pooled = control + treatment
    observed = statistics.median(treatment) - statistics.median(control)
    extreme = 0
    total = 0
    for treatment_indices in itertools.combinations(range(len(pooled)), len(treatment)):
        treatment_set = set(treatment_indices)
        permuted_treatment = [pooled[index] for index in treatment_indices]
        permuted_control = [
            value for index, value in enumerate(pooled) if index not in treatment_set
        ]
        difference = statistics.median(permuted_treatment) - statistics.median(
            permuted_control
        )
        extreme += int(difference >= observed - 1e-12)
        total += 1
    return extreme / total


def _minimum_leave_one_each_out_ratio(
    numerator: Sequence[float], denominator: Sequence[float]
) -> float:
    ratios = []
    for numerator_index in range(len(numerator)):
        kept_numerator = [
            float(value)
            for index, value in enumerate(numerator)
            if index != numerator_index
        ]
        for denominator_index in range(len(denominator)):
            kept_denominator = [
                float(value)
                for index, value in enumerate(denominator)
                if index != denominator_index
            ]
            ratios.append(
                statistics.median(kept_numerator)
                / statistics.median(kept_denominator)
            )
    return min(ratios)


def extract_salibi_autocatalysis(
    source_xlsx: str | Path,
) -> dict[int, dict[str, tuple[float, ...]]]:
    """Extract Fig. 2b FT-cycle ligation yields from Salibi et al. 2023."""

    rows = read_xlsx_sheet(source_xlsx, "Figure 2")
    result: dict[int, dict[str, tuple[float, ...]]] = {}
    for row in rows:
        if len(row) < 11 or not isinstance(row[1], (int, float)):
            continue
        cycle = int(row[1])
        # The source sheet repeats cycle labels in a later summary table
        # (mean, standard deviation, N).  Keep the first occurrence, which is
        # the three biological replicate values used by the gate.
        if cycle not in {0, 1, 2, 3} or cycle in result:
            continue
        result[cycle] = {
            "negative": _slice_numbers(row, 2, 5),
            "seed_1uM": _slice_numbers(row, 5, 8),
            "seed_2uM": _slice_numbers(row, 8, 11),
        }
    if set(result) != {0, 1, 2, 3}:
        raise ValueError("Salibi Figure 2 source layout changed")
    return result


def extract_salibi_serial_transfer(
    source_xlsx: str | Path,
) -> dict[int, dict[str, tuple[float, ...]]]:
    """Extract Fig. 3b seven-generation serial-transfer ligation yields."""

    rows = read_xlsx_sheet(source_xlsx, "Figure 3")
    result: dict[int, dict[str, tuple[float, ...]]] = {}
    for row in rows:
        if len(row) < 11 or not isinstance(row[1], (int, float)):
            continue
        generation = int(row[1])
        if generation not in set(range(8)) or generation in result:
            continue
        result[generation] = {
            "negative": _slice_numbers(row, 2, 5),
            "seed_1uM": _slice_numbers(row, 5, 8),
            "seed_2uM": _slice_numbers(row, 8, 11),
        }
    if set(result) != set(range(8)):
        raise ValueError("Salibi Figure 3 source layout changed")
    return result


def evaluate_autocatalysis(
    cycle_data: Mapping[int, Mapping[str, Sequence[float]]],
    serial_data: Mapping[int, Mapping[str, Sequence[float]]],
) -> dict[str, object]:
    """Apply effect-size and complete-separation gates to the RNA experiment."""

    cycle3 = cycle_data[3]
    negative = tuple(float(value) for value in cycle3["negative"])
    seed1 = tuple(float(value) for value in cycle3["seed_1uM"])
    seed2 = tuple(float(value) for value in cycle3["seed_2uM"])
    negative_median = statistics.median(negative)
    seed1_ratio = statistics.median(seed1) / negative_median
    seed2_ratio = statistics.median(seed2) / negative_median
    seed1_loo_ratio = _minimum_leave_one_each_out_ratio(seed1, negative)
    seed2_loo_ratio = _minimum_leave_one_each_out_ratio(seed2, negative)
    complete_separation = min(seed1 + seed2) > max(negative)
    dose_ordered = statistics.median(seed2) > statistics.median(seed1)
    autocatalysis_passed = (
        complete_separation
        and dose_ordered
        and seed1_ratio >= 4.0
        and seed2_ratio >= 4.0
    )

    first = serial_data[1]
    last = serial_data[7]
    serial_rows: dict[str, dict[str, float]] = {}
    serial_passes: list[bool] = []
    for condition in ("seed_1uM", "seed_2uM"):
        first_median = statistics.median(float(value) for value in first[condition])
        last_values = tuple(float(value) for value in last[condition])
        last_median = statistics.median(last_values)
        negative_last = statistics.median(
            float(value) for value in last["negative"]
        )
        retention = last_median / first_median
        negative_ratio = last_median / negative_last
        separated = min(last_values) > max(float(value) for value in last["negative"])
        passed = retention >= 0.5 and negative_ratio >= 2.0 and separated
        serial_passes.append(passed)
        serial_rows[condition] = {
            "generation1_median": first_median,
            "generation7_median": last_median,
            "generation7_over_generation1": retention,
            "generation7_over_negative": negative_ratio,
            "complete_separation_at_generation7": separated,
            "minimum_leave_one_each_out_generation7_over_negative": (
                _minimum_leave_one_each_out_ratio(last_values, last["negative"])
            ),
        }
    serial_passed = all(serial_passes)
    return {
        "passed": autocatalysis_passed and serial_passed,
        "autocatalysis": {
            "passed": autocatalysis_passed,
            "cycle3_negative": list(negative),
            "cycle3_seed_1uM": list(seed1),
            "cycle3_seed_2uM": list(seed2),
            "seed_1uM_median_ratio_over_negative": seed1_ratio,
            "seed_2uM_median_ratio_over_negative": seed2_ratio,
            "seed_1uM_minimum_leave_one_each_out_ratio": seed1_loo_ratio,
            "seed_2uM_minimum_leave_one_each_out_ratio": seed2_loo_ratio,
            "seed_1uM_exact_one_sided_median_permutation_p": (
                _exact_median_permutation_p(negative, seed1)
            ),
            "seed_2uM_exact_one_sided_median_permutation_p": (
                _exact_median_permutation_p(negative, seed2)
            ),
            "complete_seed_vs_negative_separation": complete_separation,
            "dose_ordered": dose_ordered,
            "small_sample_caveat": (
                "with n=3 per condition, the one-sided exact median-permutation "
                "p-value is 0.10 despite complete separation; the effect-size gate "
                "is retrospective support, not an alpha=0.05 confirmation"
            ),
        },
        "serial_transfer": {
            "passed": serial_passed,
            "generations": 7,
            "conditions": serial_rows,
        },
    }


def _time_pairs(rows: Sequence[Sequence[Any]]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        if len(row) < 4 or row[1] not in {"0h", "16h"}:
            continue
        round_index = int(_number(row[2]))
        result.setdefault(round_index, {})[str(row[1])] = _number(row[3])
    return result


def extract_abil_boundary(fig3_xlsx: str | Path) -> dict[str, object]:
    """Extract liposome and bulk serial-transfer measurements from Fig. 3."""

    ratios: dict[int, dict[str, float]] = {}
    for row in read_xlsx_sheet(fig3_xlsx, "Fig 3g"):
        if (
            len(row) < 4
            or not isinstance(row[1], (int, float))
            or not isinstance(row[2], (int, float))
            or not isinstance(row[3], (int, float))
        ):
            continue
        ratios[int(row[1])] = {
            "liposome": _number(row[2]),
            "bulk": _number(row[3]),
        }
    liposome = _time_pairs(read_xlsx_sheet(fig3_xlsx, "Fig 3b"))
    bulk = _time_pairs(read_xlsx_sheet(fig3_xlsx, "Fig 3e"))
    if set(ratios) != set(range(1, 6)) or set(bulk) != set(range(1, 7)):
        raise ValueError("Abil Figure 3 source layout changed")
    return {"amplification_ratios": ratios, "liposome": liposome, "bulk": bulk}


def evaluate_boundary(boundary_data: Mapping[str, object]) -> dict[str, object]:
    """Test compartment-associated persistence without claiming strict necessity."""

    ratios = boundary_data["amplification_ratios"]
    liposome = boundary_data["liposome"]
    bulk = boundary_data["bulk"]
    late_rounds = (3, 4, 5)
    late_dominance = all(
        ratios[round_index]["liposome"] > ratios[round_index]["bulk"]
        for round_index in late_rounds
    )
    round5_ratio = ratios[5]["liposome"] / ratios[5]["bulk"]
    liposome_final_retention = liposome[5]["16h"] / liposome[1]["16h"]
    bulk_final_retention = bulk[6]["16h"] / bulk[1]["16h"]
    passed = (
        late_dominance
        and round5_ratio >= 2.0
        and liposome_final_retention >= 1.0
        and bulk_final_retention <= 0.1
    )
    return {
        "passed": passed,
        "late_rounds_liposome_over_bulk": late_dominance,
        "round5_amplification_ratio_liposome_over_bulk": round5_ratio,
        "liposome_round5_over_round1_final_DNA": liposome_final_retention,
        "bulk_round6_over_round1_final_DNA": bulk_final_retention,
        "important_caveat": (
            "the liposome and bulk campaigns use different transfer schemes, and "
            "the semi-continuous liposome condition permits some extra-vesicular "
            "amplification; this supports compartment-associated persistence, not "
            "a clean factorial proof that a boundary is universally necessary"
        ),
    }


def extract_mutation_campaign(
    supplementary_xlsx: str | Path,
    sheet_name: str,
) -> tuple[dict[str, object], ...]:
    rows = read_xlsx_sheet(supplementary_xlsx, sheet_name)
    header_index = next(
        index for index, row in enumerate(rows) if row and row[0] == "Mutation"
    )
    header = rows[header_index]
    round_columns = [
        index
        for index, value in enumerate(header)
        if isinstance(value, str) and re.fullmatch(r"[RM]\d+", value)
    ]
    result: list[dict[str, object]] = []
    for row in rows[header_index + 1 :]:
        if len(row) <= max(round_columns) or not isinstance(row[0], str):
            continue
        try:
            trajectory = tuple(_number(row[index]) for index in round_columns)
        except TypeError:
            continue
        note = row[max(round_columns) + 1] if len(row) > max(round_columns) + 1 else None
        result.append(
            {
                "mutation": row[0],
                "region": row[1],
                "rounds": [header[index] for index in round_columns],
                "trajectory": list(trajectory),
                "low_depth_warning": isinstance(note, str) and "low read depth" in note,
            }
        )
    return tuple(result)


def evaluate_heredity(
    campaigns: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    """Require independently repeated enrichment of initially rare variants."""

    campaign_results: dict[str, object] = {}
    pass_count = 0
    for name, mutations in campaigns.items():
        candidates = []
        for mutation in mutations:
            trajectory = tuple(float(value) for value in mutation["trajectory"])
            if (
                not mutation["low_depth_warning"]
                and trajectory[0] <= 0.001
                and trajectory[-1] >= 0.25
                and max(trajectory[1:]) >= 0.25
            ):
                candidates.append(
                    {
                        "mutation": mutation["mutation"],
                        "region": mutation["region"],
                        "initial_frequency": trajectory[0],
                        "final_frequency": trajectory[-1],
                        "fold_enrichment": trajectory[-1] / max(trajectory[0], 1e-12),
                        "trajectory": list(trajectory),
                    }
                )
        passed = bool(candidates)
        pass_count += int(passed)
        campaign_results[name] = {"passed": passed, "qualifying_variants": candidates}
    required = len(campaigns)
    return {
        "passed": pass_count == required and required >= 2,
        "campaigns_passed": pass_count,
        "campaigns_required": required,
        "campaigns": campaign_results,
        "important_caveat": (
            "the intermittent campaigns include experimenter-mediated extraction, "
            "PCR, and re-encapsulation; they establish heritable molecular change "
            "under compartmentalized selection, not autonomous organismal heredity"
        ),
    }


def build_empirical_protocell_artifact(
    salibi_xlsx: str | Path,
    abil_source_zip: str | Path,
    abil_fig3_xlsx: str | Path,
    abil_supplementary_xlsx: str | Path,
) -> dict[str, object]:
    salibi_hash = sha256_file(salibi_xlsx)
    abil_zip_hash = sha256_file(abil_source_zip)
    abil_fig3_hash = sha256_file(abil_fig3_xlsx)
    abil_supplementary_hash = sha256_file(abil_supplementary_xlsx)
    expected_hashes = {
        "salibi_2023": "d39d7b0b9a30091fe848e1bb12737030da043025a5b96ab54b577444f37b7fc7",
        "abil_2024_source_zip": "9ba3b9bbca80be0371bb7efb54f2ac954731cd07f316700cce4d090e1d98bf4a",
        "abil_2024_fig3": "3f61c11bae6d92f63ef9b18c86829683fd82f11634b992f285f3febdf0c97aa9",
        "abil_2024_supplementary": "2488340879755750114099e38f511666c1704f07be9f6893cadf9fafb1b3c5f1",
    }
    hash_checks = {
        "salibi_2023": salibi_hash == expected_hashes["salibi_2023"],
        "abil_2024_source_zip": abil_zip_hash
        == expected_hashes["abil_2024_source_zip"],
        "abil_2024_fig3": abil_fig3_hash == expected_hashes["abil_2024_fig3"],
        "abil_2024_supplementary": abil_supplementary_hash
        == expected_hashes["abil_2024_supplementary"],
    }
    autocatalysis = evaluate_autocatalysis(
        extract_salibi_autocatalysis(salibi_xlsx),
        extract_salibi_serial_transfer(salibi_xlsx),
    )
    boundary = evaluate_boundary(extract_abil_boundary(abil_fig3_xlsx))
    heredity = evaluate_heredity(
        {
            "Int-WT(1)": extract_mutation_campaign(
                abil_supplementary_xlsx, "Int-WT(1)"
            ),
            "Int-WT(2)": extract_mutation_campaign(
                abil_supplementary_xlsx, "Int-WT(2)"
            ),
        }
    )
    existence_passed = (
        all(hash_checks.values())
        and bool(autocatalysis["passed"])
        and bool(boundary["passed"])
        and bool(heredity["passed"])
    )
    return {
        "artifact_type": "clarus_origin_life_component_evidence_gate",
        "artifact_version": 2,
        "phase": "retrospective_external_source_data",
        "empirical_component_support_gate_passed": existence_passed,
        "confirmatory_statistical_gate_at_alpha_0_05_passed": False,
        "single_system_joint_cycle_proven": False,
        "autonomous_growth_division_heredity_proven": False,
        "universal_minimality_or_necessity_proven": False,
        "claim_supported": (
            "across two engineered systems, source data support seeded autocatalytic "
            "replication, serial-transfer persistence, compartment-associated "
            "persistence, and heritable sequence change at the component level"
        ),
        "claim_not_supported": (
            "no single system here jointly demonstrates autonomous growth, division, "
            "and heredity; this is not proof of the historical first organism or "
            "the universal necessity/sufficiency of exactly three terms"
        ),
        "equation_scope": (
            "X[n+1] = Pi_C[X[n] + A_auto(X,E) + B_compartment(X) + "
            "C_copy(X) - L_leak(X)]"
        ),
        "sources": {
            "salibi_2023": {
                "doi": "10.1038/s41467-023-36940-z",
                "source_data_sha256": salibi_hash,
                "expected_sha256": expected_hashes["salibi_2023"],
                "verified": hash_checks["salibi_2023"],
            },
            "abil_2024": {
                "doi": "10.1038/s41467-024-53226-0",
                "source_zip_sha256": abil_zip_hash,
                "source_zip_expected_sha256": expected_hashes[
                    "abil_2024_source_zip"
                ],
                "fig3_sha256": abil_fig3_hash,
                "fig3_expected_sha256": expected_hashes["abil_2024_fig3"],
                "supplementary_sha256": abil_supplementary_hash,
                "supplementary_expected_sha256": expected_hashes[
                    "abil_2024_supplementary"
                ],
                "verified": hash_checks["abil_2024_source_zip"]
                and hash_checks["abil_2024_fig3"]
                and hash_checks["abil_2024_supplementary"],
            },
        },
        "gates": {
            "autocatalysis_and_serial_transfer": autocatalysis,
            "compartment_associated_persistence": boundary,
            "heritable_sequence_change": heredity,
        },
        "evidence_grade": (
            "cross-study component-level empirical support, not a joint single-system "
            "existence proof; necessity remains open because there is no clean 2^3 "
            "factorial ablation or autonomous growth-division-heredity cycle"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salibi-xlsx", required=True)
    parser.add_argument("--abil-source-zip", required=True)
    parser.add_argument("--abil-fig3-xlsx", required=True)
    parser.add_argument("--abil-supplementary-xlsx", required=True)
    parser.add_argument("--output")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    artifact = build_empirical_protocell_artifact(
        args.salibi_xlsx,
        args.abil_source_zip,
        args.abil_fig3_xlsx,
        args.abil_supplementary_xlsx,
    )
    payload = json.dumps(artifact, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return int(
        args.require_pass
        and not artifact["empirical_component_support_gate_passed"]
    )


__all__ = [
    "build_empirical_protocell_artifact",
    "evaluate_autocatalysis",
    "evaluate_boundary",
    "evaluate_heredity",
    "extract_abil_boundary",
    "extract_mutation_campaign",
    "extract_salibi_autocatalysis",
    "extract_salibi_serial_transfer",
    "main",
    "read_xlsx_sheet",
    "sha256_file",
    "xlsx_sheet_names",
]


if __name__ == "__main__":
    raise SystemExit(main())
