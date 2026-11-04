# CYP inhibition method report

[Read the two-page paper](method-report.pdf).

The paper describes the research campaigns, representations, hit-expansion
evaluation, final Chemprop ensemble, and leaderboard-informed calibration.
The reported internal scores are development results. No official score for
the final candidate is claimed.

The LaTeX source is [method-report.tex](method-report.tex). Figures can be
regenerated from [figure-data.json](figure-data.json) using Python and Matplotlib:

```sh
cd report
python render_figures.py
pdflatex -interaction=nonstopmode -halt-on-error method-report.tex
pdflatex -interaction=nonstopmode -halt-on-error method-report.tex
```

The PDF requires a standard LaTeX installation with Latin Modern, geometry,
microtype, amsmath, graphicx, booktabs, caption, and hyperref (TeX Live base and
recommended packages). The figure script does not load training data or private
run directories. Aggregate values are frozen from the completed campaign audit.

The final prediction CSV is identified by SHA-256
`88bdb3dbe8801a4d442de7eec0c0c5007146130d74d49418dce75dccf4d36d2e`.
Predictions, checkpoints, organizer correspondence, and local run artifacts are
not included in this report directory.
