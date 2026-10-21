# Model recipes and the shared data catalog

See [the data-source guide](../docs/data-sources.md) for each source's targets,
potential modeling value, preparation rules, limitations and measured evidence.

Only two model recipes are maintained: `models/incumbent.json` (nested MA-ST-RAE
0.743511) and `models/frozen-challenger.json` (0.739029, not promoted). Embeddings
are model features, not an additional assay source. See `../program.md` for the
single-axis research workflow. Trial recipes and results belong under ignored
`runs/`, not here.

`configs/data.json` registers the official bundle’s direct and auxiliary locations. Each model recipe
chooses which to consume through `data_sources`; merely registering a source does
not load it. Official direct labels are required for training and evaluation.

| Source name | Prepared inputs | Adapter in `src/somerra_openadmet/data_sources/` |
| --- | --- | --- |
| `official_direct` | `data/prepared/official-direct` | `official.py` |
| `official_auxiliary` | `data/prepared/official-auxiliary` | `official.py` |

For example, a neural recipe can specify
`"data_sources": ["official_direct", "official_auxiliary"]`.
The two maintained CPU recipes select only `official_direct`; selecting auxiliary
sources with a CPU recipe raises an error rather than silently ignoring them.
Unknown, duplicate or unavailable selections are rejected. Unselected sources
need not exist on disk for a run. Provenance records both the catalog hash and
the selected source names/paths; evaluators hash the inputs they actually consume.

For a fresh installation, `make prepare-regression-data` downloads the official
files and prepares the official direct and auxiliary measurements as one OpenADMET bundle.
It downloads official challenge files, not external datasets. Snapshot directories must be new; the command stops
instead of replacing existing evidence. If setup is interrupted, resume the missing
stages individually using the commands in the Makefile. Training never downloads
or refreshes a source implicitly.

All JSON paths are relative to the repository root. Existing `runs/regression-*`
snapshots are not moved; update the catalog paths to reuse them without refetching.
`python autoresearch.py plan --output-dir runs/regression-trial-001` previews the
recipe; replace `plan` with `run` to execute it. Both use the shared catalog
by default; `--data-config` can override its location for another installation.
There is no separate data config per experiment. ChEMBL and PubChem integrations
are deferred to [#3](https://github.com/Somerra/somerra-openadmet-cyp/issues/3) and
[#4](https://github.com/Somerra/somerra-openadmet-cyp/issues/4); their names and
catalog paths are rejected by the current configuration resolver.

Historical per-trial configs were removed from the maintained catalog, not from
run artifacts or Git history. Exact old flat JSON remains at commit
`9b43c07c2c9c3afc79a445634097b4481b79d0ac`. Restore only the needed historical
file into ignored `runs/` with `git show COMMIT:configs/NAME`. Legacy `--config`
remains supported. Do not substitute a current recipe for a historical recovery.
