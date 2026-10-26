"""Structure-only distribution matching for parent-grouped 60/20/20 partitions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import wasserstein_distance

from .constants import ACTIVITY_ENDPOINTS
from .data_sources.common import parent_smiles

PARTITIONS = ("train", "validation", "internal_test")
OBJECTIVES = (
    "validation_composition",
    "validation_distance",
    "internal_test_composition",
    "internal_test_distance",
)
FINGERPRINT = dict(
    radius=2,
    fpSize=2048,
    includeChirality=False,
    useBondTypes=True,
    countSimulation=False,
    includeRingMembership=True,
)
STANDARDIZATION = "Cleanup; largest organic fragment; uncharge; non-isomeric canonical parent"


def read_molecules(path: Path, *, labeled=False):
    """Return identifiers, standardized structures and label availability, never label values."""
    columns = ["Molecule_Name", "SMILES"] + (list(ACTIVITY_ENDPOINTS) if labeled else [])
    frame = pd.read_csv(path, usecols=columns, dtype={"Molecule_Name": str, "SMILES": str})
    if frame.empty or frame["Molecule_Name"].isna().any() or frame.Molecule_Name.duplicated().any():
        raise ValueError("Molecule IDs must be nonmissing and unique; input must be nonempty")
    if frame.SMILES.isna().any():
        raise ValueError("Missing molecular structure")
    structures = {s: parent_smiles(s) for s in sorted(frame.SMILES.unique())}
    frame["parent"] = frame.SMILES.map(structures)
    for ep in ACTIVITY_ENDPOINTS if labeled else []:
        frame[ep] = frame[ep].notna()
    return frame.sort_values("Molecule_Name").reset_index(drop=True)


def fingerprints(parents):
    generator = rdFingerprintGenerator.GetMorganGenerator(**FINGERPRINT)
    return [generator.GetFingerprint(Chem.MolFromSmiles(p)) for p in parents]


def kernel(left, right):
    """Tanimoto similarity, not distance; rows and columns retain supplied ordering."""
    return np.asarray(
        [DataStructs.BulkTanimotoSimilarity(fp, right) for fp in left], dtype=np.float32
    )


def nearest_distances(similarities, query_ids=None, reference_ids=None):
    """Exclude identical parent identities, including an accidental self-neighbor."""
    values = np.asarray(similarities, dtype=np.float32).copy()
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("Nearest-neighbor calculation needs a nonempty reference")
    if query_ids is not None:
        if reference_ids is None:
            raise ValueError("Reference identities required for self exclusion")
        values[np.asarray(query_ids)[:, None] == np.asarray(reference_ids)[None, :]] = -np.inf
    result = 1.0 - values.max(axis=1)
    if not np.isfinite(result).all():
        raise ValueError("No non-self neighbor available")
    return result


def composition_mmd(holdout_kernel, cross_mean, official_mean):
    """Biased empirical kernel MMD; diagonals included, unique parents equally weighted."""
    square = float(np.mean(holdout_kernel, dtype=np.float64) + official_mean - 2 * cross_mean)
    return float(np.sqrt(max(0.0, square)))


def summary(values):
    values = np.asarray(values)
    if not len(values):
        return {"n": 0}
    return dict(
        n=len(values),
        mean=float(values.mean()),
        **dict(zip(("p10", "median", "p90"), np.quantile(values, [0.1, 0.5, 0.9]), strict=True)),
    )


def scaffold(parent):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(parent))


def properties(parents):
    mols = [Chem.MolFromSmiles(p) for p in parents]
    return {
        name: summary([fn(m) for m in mols])
        for name, fn in [
            ("molecular_weight", Descriptors.MolWt),
            ("logp", Descriptors.MolLogP),
            ("tpsa", Descriptors.TPSA),
            ("heavy_atoms", Descriptors.HeavyAtomCount),
            ("rings", Descriptors.RingCount),
        ]
    }


@dataclass
class MatchingProblem:
    official: pd.DataFrame
    blind: pd.DataFrame

    def __post_init__(self):
        self.parents = np.asarray(sorted(self.official.parent.unique()))
        self.blind_parents = np.asarray(sorted(self.blind.parent.unique()))
        if set(self.parents) & set(self.blind_parents):
            raise ValueError("Official labeled training and blind test share standardized parents")
        if len(self.parents) < 5:
            raise ValueError("Need at least five labeled molecular parents")
        self.counts = (
            self.official.groupby("parent")[list(ACTIVITY_ENDPOINTS)]
            .sum()
            .loc[self.parents]
            .to_numpy(dtype=int)
        )
        self.row_counts = self.official.groupby("parent").size().loc[self.parents].to_numpy()
        self.fps = fingerprints(self.parents)
        self.blind_fps = fingerprints(self.blind_parents)
        self.similarity = kernel(self.fps, self.fps)
        self.cross = kernel(self.fps, self.blind_fps)
        self.cross_means = self.cross.mean(axis=1, dtype=np.float64)
        self.official_kernel_mean = float(
            kernel(self.blind_fps, self.blind_fps).mean(dtype=np.float64)
        )
        self.target = nearest_distances(self.cross.T)
        heldout = len(self.parents) // 5
        self.sizes = np.array([len(self.parents) - 2 * heldout, heldout, heldout])

    def random_partition(self, rng):
        assignment = np.repeat(np.arange(3), self.sizes)
        return rng.permutation(assignment)

    def coverage(self, assignment, minimum):
        return all((self.counts[assignment == p].sum(axis=0) >= minimum).all() for p in (1, 2))

    def objectives(self, assignment):
        train = np.flatnonzero(assignment == 0)
        values = []
        for p in (1, 2):
            hold = np.flatnonzero(assignment == p)
            if not len(hold) or not len(train):
                raise ValueError("Every partition must contain molecular parents")
            values.extend(
                [
                    composition_mmd(
                        self.similarity[np.ix_(hold, hold)],
                        self.cross_means[hold].mean(),
                        self.official_kernel_mean,
                    ),
                    wasserstein_distance(
                        nearest_distances(self.similarity[np.ix_(hold, train)]), self.target
                    ),
                ]
            )
        return np.asarray(values)

    def scaffold_partition(self):
        groups = {}
        for i, parent in enumerate(self.parents):
            groups.setdefault(scaffold(parent), []).append(i)
        if len(groups) < 3:
            return None
        assignment = np.zeros(len(self.parents), dtype=int)
        loads = np.zeros(3)
        for position, (_, indices) in enumerate(
            sorted(groups.items(), key=lambda x: (-len(x[1]), x[0]))
        ):
            # Seed all three partitions, then minimize proportional size error.
            choices = [position] if position < 3 else range(3)
            costs = []
            for p in choices:
                proposed = loads.copy()
                proposed[p] += len(indices)
                costs.append((np.sum((proposed - self.sizes) ** 2 / self.sizes), p))
            p = min(costs)[1]
            assignment[indices] = p
            loads[p] += len(indices)
        return assignment

    def assignments_frame(self, assignment):
        lookup = dict(zip(self.parents, [PARTITIONS[i] for i in assignment], strict=True))
        result = self.official[["Molecule_Name", "parent"]].copy()
        result["partition"] = result.parent.map(lookup)
        return result

    def feasibility(self):
        """Optimistic support bound: other holdouts are allowed as neighbors here."""
        optimistic = nearest_distances(self.similarity, self.parents, self.parents)
        total_holdout = int(self.sizes[1:].sum())
        return {
            "reference": "all other labeled parents; optimistic, not an achievable split",
            "all_labeled_nonself_distance": summary(optimistic),
            "best_total_holdout_nonself_distance": summary(np.sort(optimistic)[:total_holdout]),
            "thresholds": [
                {
                    "distance_at_most": threshold,
                    "official_fraction": float(np.mean(self.target <= threshold)),
                    "available_parents": int(np.sum(optimistic <= threshold)),
                    "required_across_holdouts": float(
                        total_holdout * np.mean(self.target <= threshold)
                    ),
                }
                for threshold in (0.3, 0.4, 0.5, 0.6)
            ],
        }

    def endpoint_audit(self, assignment):
        result = {}
        for column, endpoint in enumerate(ACTIVITY_ENDPOINTS):
            labeled = self.counts[:, column] > 0
            retained = np.flatnonzero(labeled & (assignment == 0))
            all_refs = np.flatnonzero(labeled)
            item = {"all_training_parents": len(all_refs), "retained_parents": len(retained)}
            for name, refs in [("official_full", all_refs), ("official_retained", retained)]:
                item[name] = (
                    summary(nearest_distances(self.cross[refs].T))
                    if len(refs)
                    else {"n": 0, "reason": "No endpoint-labeled reference parents"}
                )
            for part, name in [(1, "validation"), (2, "internal_test")]:
                query = np.flatnonzero(labeled & (assignment == part))
                item[name] = (
                    summary(nearest_distances(self.similarity[np.ix_(query, retained)]))
                    if len(query) and len(retained)
                    else {"n": 0, "reason": "No endpoint-labeled queries or references"}
                )
            result[endpoint] = item
        return result

    def audit(self, assignment, minimum):
        train = np.flatnonzero(assignment == 0)
        train_scaffolds = {scaffold(p) for p in self.parents[train]}
        report = {
            "objectives": dict(zip(OBJECTIVES, self.objectives(assignment), strict=True)),
            "coverage_ok": self.coverage(assignment, minimum),
            "endpoint_distances": self.endpoint_audit(assignment),
            "target_parent_counts": dict(zip(PARTITIONS, self.sizes.tolist(), strict=True)),
            "partitions": {},
        }
        for p, name in enumerate(PARTITIONS):
            indices = np.flatnonzero(assignment == p)
            parents = self.parents[indices]
            part = {
                "parents": len(indices),
                "records": int(self.row_counts[indices].sum()),
                "labels": dict(
                    zip(ACTIVITY_ENDPOINTS, self.counts[indices].sum(axis=0).tolist(), strict=True)
                ),
                "properties": properties(parents),
            }
            if p:
                distances = nearest_distances(self.similarity[np.ix_(indices, train)])
                part.update(
                    nearest_training_distance=summary(distances),
                    nearest_training_distance_record_weighted=summary(
                        np.repeat(distances, self.row_counts[indices])
                    ),
                    distance_values=distances.tolist(),
                    parent_fraction_with_training_scaffold=float(
                        np.mean([scaffold(s) in train_scaffolds for s in parents])
                    ),
                )
            report["partitions"][name] = part
        blind_to_retained = nearest_distances(self.cross[train].T)
        blind_lookup = dict(zip(self.blind_parents, self.target, strict=True))
        report["official_test"] = {
            "parents": len(self.blind_parents),
            "records": len(self.blind),
            "nearest_all_official_training": summary(self.target),
            "nearest_all_official_training_record_weighted": summary(
                self.blind.parent.map(blind_lookup).to_numpy()
            ),
            "nearest_retained_training": summary(blind_to_retained),
            "target_distance_values": self.target.tolist(),
            "properties": properties(self.blind_parents),
            "parent_fraction_with_training_scaffold": float(
                np.mean([scaffold(p) in train_scaffolds for p in self.blind_parents])
            ),
        }
        return report


def acceptable(values, baseline, scales):
    return bool(
        np.all(values <= baseline + 1e-12)
        and np.mean(values / scales) < np.mean(baseline / scales) - 1e-12
    )


def search(problem, *, seed=42, candidates=128, swaps=8192, minimum=100, patience=1024):
    if candidates < 1 or swaps < 0 or minimum < 1 or patience < 1:
        raise ValueError("Invalid search counts or minimum label coverage")
    rng = np.random.default_rng(seed)
    randoms = []
    trace = []
    for i in range(candidates):
        assignment = problem.random_partition(rng)
        if not problem.coverage(assignment, minimum):
            trace.append(dict(stage="random", step=i, feasible=False))
            continue
        randoms.append((i, assignment, problem.objectives(assignment)))
    if not randoms:
        raise ValueError("No random candidate meets minimum holdout coverage")
    base_values = np.median([x[2] for x in randoms], axis=0)
    scales = np.maximum(base_values, 0.01)
    # A real partition for visualization only; the gate uses the componentwise median.
    baseline = min(randoms, key=lambda x: np.mean(np.abs(x[2] - base_values) / scales))[1]
    best = baseline.copy()
    best_values = problem.objectives(best)
    accepted = None

    def consider(assignment, values, stage, step):
        nonlocal best, best_values, accepted
        loss = float(np.mean(values / scales))
        improved = bool(loss < np.mean(best_values / scales) - 1e-12)
        trace.append(
            dict(
                stage=stage,
                step=step,
                feasible=True,
                loss=loss,
                objectives=values.tolist(),
                improved=improved,
            )
        )
        if improved:
            best, best_values = assignment.copy(), values.copy()
        if acceptable(values, base_values, scales):
            if accepted is None or loss < accepted[0]:
                accepted = loss, assignment.copy(), values.copy()
        return improved

    for i, assignment, values in randoms:
        consider(assignment, values, "random", i)
    stale = 0
    evaluated = 0
    for i in range(swaps):
        proposal = best.copy()
        left, right = rng.choice(3, size=2, replace=False)
        a = rng.choice(np.flatnonzero(proposal == left))
        b = rng.choice(np.flatnonzero(proposal == right))
        proposal[a], proposal[b] = proposal[b], proposal[a]
        evaluated += 1
        if not problem.coverage(proposal, minimum):
            trace.append(dict(stage="swap", step=i, feasible=False))
            improved = False
        else:
            improved = consider(proposal, problem.objectives(proposal), "swap", i)
        stale = 0 if improved else stale + 1
        if stale >= patience:
            break
    chosen = accepted[1] if accepted is not None else best
    return (
        chosen,
        baseline,
        dict(
            status="approved" if accepted is not None else "unapproved",
            baseline_definition="componentwise median of feasible random candidate objectives",
            feasible_random_candidates=len(randoms),
            random_objective_quantiles={
                str(q): np.quantile([x[2] for x in randoms], q, axis=0).tolist()
                for q in (0.1, 0.5, 0.9)
            },
            normalization_floor=0.01,
            normalization_scales=scales.tolist(),
            baseline_objectives=base_values.tolist(),
            swap_proposals_evaluated=evaluated,
            stopping_reason="no_improvement_patience" if stale >= patience else "proposal_budget",
            proposals_since_improvement=stale,
            patience=patience,
            trace=trace,
            reason="All four objectives non-worsening versus random median and aggregate improved"
            if accepted is not None
            else "No candidate met the random-median gate; do not adopt",
        ),
    )


def load_assignments(path, problem):
    frame = pd.read_csv(path, dtype=str)
    required = {"Molecule_Name", "parent", "partition"}
    if not required.issubset(frame) or frame.Molecule_Name.duplicated().any():
        raise ValueError("Invalid assignment schema or duplicate IDs")
    expected = problem.official.set_index("Molecule_Name").parent
    if set(frame.Molecule_Name) != set(expected.index):
        raise ValueError("Assignments must cover exactly the official labeled training IDs")
    if not frame.partition.isin(PARTITIONS).all() or frame.parent.isna().any():
        raise ValueError("Invalid partition or parent")
    if not (frame.Molecule_Name.map(expected) == frame.parent).all():
        raise ValueError("Assignment parent does not match standardized input")
    if (frame.groupby("parent").partition.nunique() != 1).any():
        raise ValueError("Parent crosses partition boundaries")
    lookup = frame.drop_duplicates("parent").set_index("parent").partition
    return np.array([PARTITIONS.index(lookup[p]) for p in problem.parents])


def provenance():
    return dict(
        rdkit_version=rdBase.rdkitVersion,
        fingerprint=FINGERPRINT,
        standardization=STANDARDIZATION,
        distance="1 - bit-vector Tanimoto similarity",
        composition="sqrt biased empirical Tanimoto-kernel MMD (unique parents)",
        difficulty="Wasserstein-1 of nearest-training distances; unique parents",
        target="official blind test versus ALL official labeled training parents",
    )
