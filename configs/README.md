# Maintained recipes and data profiles

Only two model recipes are maintained: `models/incumbent.json` (nested MA-ST-RAE
0.743511) and `models/frozen-challenger.json` (0.739029, not promoted). Embeddings
are model features, not an additional assay source. See `../program.md` for the
single-axis research workflow. Trial recipes and results belong under ignored
`runs/`, not here.

| Data profile | Prepared inputs | Adapter in `src/somerra_openadmet/data_sources/` |
| --- | --- | --- |
| `official-direct` | `data/prepared/official-direct` | `official.py` |
| `official-with-auxiliary` | Direct + `data/prepared/official-auxiliary` | `official.py` |
| `official-with-external` | Above + `data/external/chembl37`, `data/external/pubchem1851` | `chembl.py`, `pubchem.py` |

Profiles live in `configs/data/`. All JSON paths are relative to the invocation
directory; run from the repository root. Profiles point to immutable prepared
snapshots, not downloaded content in Git. Existing `runs/regression-*` snapshots
are not moved. A local profile can reference those original paths without copying
or refetching. Use `regression prepare`, `auxiliary`, `fetch-chembl`, and
`fetch-pubchem` to create new snapshots explicitly. CPU recipes accept only direct
labels. Auxiliary profiles require a separately established neural baseline.

Historical per-trial configs were removed from the maintained catalog, not from
run artifacts or Git history. Exact old flat JSON remains at commit
`9b43c07c2c9c3afc79a445634097b4481b79d0ac`. Restore only the needed historical
file into ignored `runs/` with `git show COMMIT:configs/NAME`. Legacy `--config`
remains supported. Do not substitute a current recipe for a historical recovery.
