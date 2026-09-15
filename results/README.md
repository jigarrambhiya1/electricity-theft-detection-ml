# Results

Generated artifacts go here when you run the notebooks (all save paths already
point to `../results/`):

- `stacking_ensemble_results.csv`, `full_summary_*.csv` — metrics tables
- `cm_*.png`, `roc_*.png`, `calibration_*.png`, `cv_boxplots.png`, `radar_*.png`, `accuracy_comparison.png` — figures
- `*.pkl` — trained stacking/base models and scalers (git-ignored by default)

Small CSV summaries are committed; large PNG/PKL files are up to you
(see `.gitignore` — `*.pkl` is ignored, `results/*.csv` is allowed).
