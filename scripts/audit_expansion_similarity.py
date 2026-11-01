import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from somerra_openadmet.expansion_evaluation import structure_key


def main():
    parser = argparse.ArgumentParser(
        description="Audit hit-expansion and official-test structural similarity"
    )
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workbench-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, source, out = args.split, args.source, args.output
    if out.exists():
        raise FileExistsError("Audit output already exists")
    frozen = json.loads((base / "manifest.json").read_text())
    for path in [source / "official.csv", source / "test.csv", args.workbench_csv]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != frozen["input_sha256"][str(path)]:
            raise ValueError(f"Changed source: {path}")
    for name in ["assignments.csv", "seed-hits.csv"]:
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != frozen["file_sha256"][name]:
            raise ValueError(f"Changed split: {name}")
    out.mkdir(parents=True)
    official = pd.read_csv(source / "official.csv")
    blind = pd.read_csv(source / "test.csv")
    assigned = pd.read_csv(base / "assignments.csv")
    for df in [official, blind]:
        df["expansion_parent"] = df.SMILES.map(structure_key)
    seeds = pd.read_csv(base / "seed-hits.csv")
    endpoints = [f"{c}_pIC50_direct_inhibition" for c in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]]
    reconstructed = pd.concat(
        [
            official.dropna(subset=[ep])
            .sort_values([ep, "Molecule_Name"], ascending=[False, True])
            .drop_duplicates("expansion_parent")
            .head(25)
            .assign(selection_endpoint=ep)
            for ep in endpoints
            if not ep.startswith("CYP2D6")
        ]
    ).drop_duplicates("expansion_parent")
    reconstructed.to_csv(out / "reconstructed-official-seeds.csv", index=False)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    parents = sorted(set(official.expansion_parent) | set(blind.expansion_parent))
    fps = {p: gen.GetFingerprint(Chem.MolFromSmiles(p)) for p in parents}
    rows = []
    summaries = []

    def compare(name, query, reference):
        refs = sorted(set(reference))
        ref_fps = [fps[p] for p in refs]
        values = []
        for _, r in query.iterrows():
            sim = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[r.expansion_parent], ref_fps))
            i = int(sim.argmax())
            values.append(float(sim[i]))
            rows.append(
                dict(
                    comparison=name,
                    Molecule_Name=r.Molecule_Name,
                    parent=r.expansion_parent,
                    nearest_parent=refs[i],
                    similarity=float(sim[i]),
                    neighbors_ge_04=int((sim >= 0.4).sum()),
                    neighbors_ge_06=int((sim >= 0.6).sum()),
                )
            )
        v = np.array(values)
        summary = dict(
            comparison=name,
            n=len(v),
            query_parents=query.expansion_parent.nunique(),
            reference_parents=len(refs),
            mean=float(v.mean()),
            p10=float(np.quantile(v, 0.1)),
            median=float(np.median(v)),
            p90=float(np.quantile(v, 0.9)),
            ge_04=float((v >= 0.4).mean()),
            ge_06=float((v >= 0.6).mean()),
            ge_08=float((v >= 0.8).mean()),
            exact=float((v == 1).mean()),
        )
        summaries.append(summary)
        print(json.dumps(summary), flush=True)

    proxy_train = assigned[assigned.partition == "train"]
    proxy_test = assigned[assigned.partition == "test"]
    compare("official_blind_to_official_training", blind, official.expansion_parent)
    compare("proxy_test_to_official_training_partition", proxy_test, proxy_train.expansion_parent)
    for role in ["validation", "calibration"]:
        compare(
            f"proxy_{role}_to_official_training_partition",
            assigned[assigned.partition == role],
            proxy_train.expansion_parent,
        )
    compare("official_blind_to_reconstructed_top25_seeds", blind, reconstructed.expansion_parent)
    compare("proxy_test_to_selected_seeds", proxy_test, seeds.seed_parent)
    wb = pd.read_csv(args.workbench_csv)
    wb["expansion_parent"] = wb.smiles.map(structure_key)
    wb["Molecule_Name"] = wb.molecule_name
    compare(
        "workbench_fold2_to_train_folds134",
        wb[wb.fold == 2],
        wb[wb.fold.isin([1, 3, 4])].expansion_parent,
    )
    for ep in endpoints:
        target = ep.split("_")[0]
        compare(
            f"official_blind_to_{target}_labeled_training",
            blind,
            official.loc[official[ep].notna(), "expansion_parent"],
        )
        compare(
            f"proxy_test_{target}_labeled_to_labeled_training",
            proxy_test[proxy_test[ep].notna()],
            proxy_train.loc[proxy_train[ep].notna(), "expansion_parent"],
        )
    pd.DataFrame(rows).to_csv(out / "nearest-neighbors.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    manifest = dict(
        rdkit=rdBase.rdkitVersion,
        fingerprint="Morgan radius 2, 2048 bits; standardized stereo-insensitive parent",
        weighting="molecule rows; unique parents in reference",
        official_blind_training_parent_overlap=len(
            set(blind.expansion_parent) & set(official.expansion_parent)
        ),
        proxy_test_training_parent_overlap=len(
            set(proxy_test.expansion_parent) & set(proxy_train.expansion_parent)
        ),
        seed_warning=(
            "Official seeds reconstructed from top 25 potency rows per CYP1A2, CYP2C9, CYP3A4; "
            "actual organizer seed IDs not verified"
        ),
        inputs={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                source / "official.csv",
                source / "test.csv",
                base / "assignments.csv",
                base / "seed-hits.csv",
                args.workbench_csv,
            ]
        },
        summaries=summaries,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        import matplotlib
    except ImportError:
        matplotlib = None
    if matplotlib is not None:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        allrows = pd.DataFrame(rows)
        for name, label in [
            ("official_blind_to_official_training", "Official blind (750)"),
            ("proxy_test_to_official_training_partition", "Expansion proxy test (162)"),
            ("workbench_fold2_to_train_folds134", "Workbench internal test"),
        ]:
            v = np.sort(allrows.loc[allrows.comparison == name, "similarity"])
            axs[0].plot(v, np.arange(1, len(v) + 1) / len(v), label=label)
        for name, label in [
            ("official_blind_to_reconstructed_top25_seeds", "Official blind → reconstructed seeds"),
            ("proxy_test_to_selected_seeds", "Proxy test → selected seeds"),
        ]:
            v = np.sort(allrows.loc[allrows.comparison == name, "similarity"])
            axs[1].plot(v, np.arange(1, len(v) + 1) / len(v), label=label)
        for ax, title in zip(
            axs,
            [
                "Nearest available training molecule",
                "Nearest seed hit (official seeds approximate)",
            ],
            strict=True,
        ):
            ax.set(
                xlabel="Morgan radius-2 Tanimoto",
                ylabel="Cumulative fraction",
                title=title,
                xlim=(0, 1),
                ylim=(0, 1),
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.savefig(out / "similarity-comparison.png", dpi=170)
        fig.savefig(out / "similarity-comparison.pdf")
    (out / "reproduce.py").write_text(Path(__file__).read_text())
    print("OUTPUT", out, "OVERLAP", manifest["official_blind_training_parent_overlap"], flush=True)


if __name__ == "__main__":
    main()
