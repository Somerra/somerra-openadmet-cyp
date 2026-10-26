"""CPU-only molecular split generation and auditing; does not train or submit models."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_sources.common import parent_smiles, write_json
from .download import sha256_file
from .molecular_similarity import (
    OBJECTIVES,
    PARTITIONS,
    MatchingProblem,
    fingerprints,
    kernel,
    load_assignments,
    provenance,
    read_molecules,
    search,
    summary,
)


def supplemental_audit(specifications, problem, assignment):
    """Diagnostics only. These structures never enter the split optimization."""
    result, hashes = {}, {}
    for spec in specifications:
        name, separator, filename = spec.partition("=")
        if not separator or not name or name in result:
            raise ValueError("Supplements require unique NAME=CSV specifications")
        path = Path(filename)
        columns = pd.read_csv(path, nrows=0).columns
        structure_column = next((c for c in ("SMILES", "smiles", "parent") if c in columns), None)
        if structure_column is None:
            raise ValueError("Supplement needs a SMILES, smiles, or parent column")
        frame = pd.read_csv(path, usecols=[structure_column])
        if frame.empty or frame[structure_column].isna().any():
            raise ValueError("Supplement must contain nonmissing structures")
        parents = sorted({parent_smiles(s) for s in frame[structure_column].unique()})
        heldout = set(problem.parents[assignment != 0]) | set(problem.blind_parents)
        eligible = sorted(set(parents) - heldout)
        source = {
            "unique_parents": len(parents),
            "eligible_training_parents": len(eligible),
            "excluded_exact_parents": len(set(parents) & heldout),
            "exact_overlap": {
                part: len(set(parents) & set(problem.parents[assignment == i]))
                for i, part in enumerate(PARTITIONS)
            },
        }
        source["exact_overlap"]["official_test"] = len(set(parents) & set(problem.blind_parents))
        # Process references in chunks to avoid allocating queries x an external million-row source.
        refs = fingerprints(eligible)
        for part, query in [
            *((p, problem.parents[assignment == i]) for i, p in enumerate(PARTITIONS) if i),
            ("official_test", problem.blind_parents),
        ]:
            if not refs:
                source[part] = {"n": len(query), "reason": "No eligible supplemental parents"}
                continue
            best = np.zeros(len(query), dtype=np.float32)
            queries = fingerprints(query)
            for start in range(0, len(refs), 2048):
                best = np.maximum(best, kernel(queries, refs[start : start + 2048]).max(axis=1))
            source[part] = summary(1.0 - best)
        result[name] = source
        hashes[name] = sha256_file(path)
    return result, hashes


def render_report(output, audits, status):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    names = list(audits)
    for name, audit in audits.items():
        for part, style in [("validation", "-"), ("internal_test", "--")]:
            x = np.sort(audit["partitions"][part]["distance_values"])
            axes[0].plot(x, np.arange(1, len(x) + 1) / len(x), style, label=f"{name}: {part}")
    target = np.sort(audits[names[0]]["official_test"]["target_distance_values"])
    axes[0].plot(
        target,
        np.arange(1, len(target) + 1) / len(target),
        color="black",
        linewidth=2.5,
        label="Official test → all labeled training",
    )
    axes[0].set(xlabel="Nearest-training Tanimoto distance", ylabel="Cumulative fraction")
    axes[0].legend(fontsize=6)
    positions = np.arange(len(OBJECTIVES))
    for i, name in enumerate(names):
        axes[1].bar(
            positions + i * 0.8 / len(names),
            [audits[name]["objectives"][o] for o in OBJECTIVES],
            width=0.8 / len(names),
            label=name,
        )
    axes[1].set_xticks(
        positions + 0.4,
        [
            "Valid\ncomposition",
            "Valid\ndistance",
            "Internal test\ncomposition",
            "Internal test\ndistance",
        ],
    )
    axes[1].set_ylabel("Discrepancy (lower is better; MMD / Wasserstein)")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "matching-diagnostics.svg")
    fig.savefig(output / "matching-diagnostics.png", dpi=150)
    plt.close(fig)
    rows = []
    for name, audit in audits.items():
        for part in PARTITIONS:
            item = audit["partitions"][part]
            rows.append(
                f"<tr><td>{html.escape(name)}</td><td>{part}</td>"
                f"<td>{item['parents']}</td><td>{item['records']}</td>"
                f"<td>{', '.join(map(str, item['labels'].values()))}</td></tr>"
            )
    (output / "report.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Molecular split audit</title><style>body{font:17px system-ui;max-width:1000px;"
        "margin:30px auto;padding:16px;line-height:1.5}img{width:100%}"
        "table{border-collapse:collapse}"
        "td,th{padding:8px;border-bottom:1px solid #ddd}.scroll{overflow-x:auto}"
        "</style></head><body>"
        f"<h1>Molecular split audit: {html.escape(status)}</h1>"
        "<p>Structures select the split. Activity values and model scores are not used. "
        "Internal test ranks strategies; it is not an independent final evaluation.</p>"
        '<img src="matching-diagnostics.svg" alt="Distance distributions and matching objectives">'
        '<div class="scroll"><table><tr><th>Design</th><th>Partition</th><th>Parents</th>'
        "<th>Records</th><th>Endpoint label counts (canonical order)</th></tr>"
        + "".join(rows)
        + "</table></div><p>Scaffold baseline targets 60/20/20 but cannot break scaffold groups; "
        "actual counts can differ. Supplemental-source overlap/proximity, properties, "
        "record-weighted distances, and provenance are in audit.json and manifest.json.</p>"
        "<p>Similarity matching does not establish official score or ranking correlation. "
        "These molecules appeared in earlier research; the new holdouts are prospective only.</p>"
        "</body></html>"
    )


def run(args):
    if args.min_labels < 1:
        raise ValueError("Minimum label coverage must be positive")
    if args.output.exists():
        raise FileExistsError("Use a new output directory; previous evidence is immutable")
    official = read_molecules(args.training, labeled=True)
    blind = read_molecules(args.official_test)
    problem = MatchingProblem(official, blind)
    if args.command == "generate":
        assignment, baseline, search_result = search(
            problem,
            seed=args.seed,
            candidates=args.candidates,
            swaps=args.swaps,
            minimum=args.min_labels,
        )
        status = search_result["status"]
    else:
        assignment = load_assignments(args.assignments, problem)
        baseline = problem.random_partition(np.random.default_rng(args.seed))
        search_result = None
        status = "audited_only_not_approved"
    audits = {
        "selected": problem.audit(assignment, args.min_labels),
        "random": problem.audit(baseline, args.min_labels),
    }
    scaffold = problem.scaffold_partition()
    if scaffold is not None:
        audits["scaffold"] = problem.audit(scaffold, args.min_labels)
    supplements, supplement_hashes = supplemental_audit(args.supplement, problem, assignment)
    args.output.mkdir(parents=True)
    chosen_file = "assignments.csv" if status == "approved" else "candidate-assignments.csv"
    problem.assignments_frame(assignment).to_csv(args.output / chosen_file, index=False)
    problem.assignments_frame(baseline).to_csv(args.output / "random-assignments.csv", index=False)
    if scaffold is not None:
        problem.assignments_frame(scaffold).to_csv(
            args.output / "scaffold-assignments.csv", index=False
        )
    write_json(
        args.output / "audit.json",
        {
            "designs": audits,
            "supplemental_sources": supplements,
            "scaffold_baseline_available": scaffold is not None,
        },
    )
    if search_result is not None:
        write_json(args.output / "search.json", search_result)
    manifest = {
        **provenance(),
        "status": status,
        "seed": args.seed,
        "ratios": [0.6, 0.2, 0.2],
        "min_holdout_labels_per_endpoint": args.min_labels,
        "search": {"candidates": args.candidates, "swap_proposals": args.swaps}
        if args.command == "generate"
        else None,
        "input_sha256": {
            "training": sha256_file(args.training),
            "official_test": sha256_file(args.official_test),
            "supplements": supplement_hashes,
        },
        "implementation_sha256": {
            p.name: sha256_file(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("molecular_similarity.py"),
                Path(__file__).parent / "data_sources/common.py",
            ]
        },
        "assignments_file": chosen_file,
        "assignments_sha256": sha256_file(args.output / chosen_file),
        "exclusions": "exact validation, internal-test, and official-test parents from supplements",
        "evaluation": "validation selects checkpoint; same checkpoint scores internal test",
        "internal_test_role": "strategy selection, not independent final evaluation",
        "adopted_by_training": False,
        "historically_untouched": False,
        "supplemental_sources_affect_optimization": False,
    }
    if args.command == "audit":
        manifest["input_sha256"]["assignments"] = sha256_file(args.assignments)
    write_json(args.output / "manifest.json", manifest)
    render_report(args.output, audits, status)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "selected_objectives": audits["selected"]["objectives"],
            }
        )
    )
    return 2 if status == "unapproved" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["generate", "audit"]:
        command = sub.add_parser(name)
        command.add_argument("--training", type=Path, required=True)
        command.add_argument("--official-test", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--min-labels", type=int, default=100)
        command.add_argument("--supplement", action="append", default=[], metavar="NAME=CSV")
        if name == "generate":
            command.add_argument("--candidates", type=int, default=128)
            command.add_argument("--swaps", type=int, default=512)
        else:
            command.add_argument("--assignments", type=Path, required=True)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
