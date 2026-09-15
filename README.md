# Electricity Theft Detection in Smart Grids using Machine Learning

A stacking-ensemble machine learning system for detecting electricity theft from smart meter consumption data, built as a Semester 2 group research project (M.Sc. Statistics and Data Science, NMIMS). The work reproduces the soft-voting ensemble proposed by Mohammad et al. (2023) and extends it with a **statistically validated stacking ensemble**, outperforming the baseline on accuracy and AUC-ROC.

> Full [Project Report](report/Report_Group_11.pdf) · Full [Presentation](presentation/PPT_Group_11.pdf)

---

## Problem

Electricity theft causes an estimated **$96B in annual non-technical losses (NTL)** worldwide. This project builds an automated decision-support system to classify smart meter consumption profiles as normal or one of six synthetic theft patterns, using the **TDD2022** benchmark dataset (560,640 instances).

## Approach

1. **Baseline replication** — Soft-voting ensemble (Random Forest + XGBoost + MLP), following Mohammad et al. (2023)
2. **Extension: Stacking ensemble** — Replaces fixed-weight voting with a meta-learner trained on out-of-fold (OOF) base-model predictions, learning class-specific combination weights instead of simple averaging
3. **Meta-learner selection** — 4 candidates (Logistic Regression, Ridge Classifier, Decision Tree, Extra Trees) compared via 3-fold CV across 5 criteria, with **Friedman + Nemenyi statistical tests** to confirm selection is not arbitrary

Two scenarios were evaluated:

- **P7C** — 7 classes (Normal + Theft 1–6)
- **P6C** — 6 classes (Normal + Theft 1–5, excluding the hardest-to-detect Theft 6)

## Key Results

| Metric          | Soft Voting (Baseline) | Stacking (Ours)            |
| --------------- | ---------------------- | -------------------------- |
| Accuracy — P7C | 88.00%                 | **90.19%** (+2.19pp) |
| Accuracy — P6C | 94.75%                 | **95.74%** (+0.99pp) |
| AUC-ROC — P7C  | N/R                    | **96.80%**           |
| AUC-ROC — P6C  | N/R                    | **99.52%**           |

- Meta-learner selection statistically validated: **Friedman test p < 0.05** for both scenarios (P7C: p=0.029, P6C: p=0.042)
- Selected meta-learners: **Extra Trees** (P7C), **Ridge Classifier** (P6C)
- Stacking wins on accuracy, precision, and AUC-ROC; soft voting retains an edge on macro recall — driven almost entirely by Theft 6, a class whose time-reversed consumption profile closely mimics normal usage (recall = 14% under stacking)

### Baseline reproduction (accuracy, ours vs paper)

Reproduced with `python -m src.knn_model` / `src.rf_model` / `src.xgb_model` on the full dataset (560,655 rows, 80/20 stratified split, `random_state=42`):

| Scenario | KNN (ours / paper) | RF (ours / paper) | XGB (ours / paper) |
| -------- | ------------------ | ----------------- | ------------------ |
| P7C | 84.74% / 84.34% | 84.88% / 85.72% | 84.51% / 85.67% |
| P7U | 84.52% / 84.10% | 84.85% / 85.65% | 84.30% / 85.70% |
| P6C | 90.87% / 90.50% | 94.71% / 94.70% | 90.65% / 91.21% |
| P6U | 90.63% / 89.65% | 94.65% / 94.69% | 90.20% / 90.79% |

KNN matches-or-beats the paper everywhere; RF matches (P6C/P6U near-identical); XGB trails by ~0.5–1pp, consistent with library-version variance (XGBoost emits a harmless `scale_pos_weight` unused warning for multiclass — safe to ignore).

See the [full report](report/Report_Group_11.pdf) for confusion matrices, ROC curves, calibration (ECE) analysis, and per-class breakdowns.

## Repository Structure

```
├── notebooks/          # 01 soft-voting → 02 stacking → 03 meta-learner (multiclass) → 04 meta-learner (binary)
├── src/                # Reusable scripts: preprocess + KNN/RF/XGBoost baselines + metrics
├── report/             # Full project report (PDF)
├── presentation/       # Slide deck (PDF)
├── results/            # Generated CSVs/figures/models go here (see results/README.md)
├── data/               # Dataset access instructions (CSV not committed — see below)
├── requirements.txt    # Pinned environment (Python 3.13)
└── .gitignore
```

## Dataset

This project uses **TDD2022** (Theft Detection Dataset 2022) by Zidi et al. (2022). The dataset is **not** committed (61 MB) — see [`data/README.md`](data/README.md) for the download link.

- **Source:** Mendeley Data — DOI: [10.17632/c3c7329tjj.1](https://doi.org/10.17632/c3c7329tjj.1)
- Place the downloaded file at `data/Dataset_actual.csv` (actual columns: 10 consumption features + `Class` + `theft` = `Normal`, `Theft1`…`Theft6`).

## Setup & Run Order

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (use source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
# put Dataset_actual.csv into data/, then:
jupyter notebook
```

1. `notebooks/01_base_models_soft_voting.ipynb` — baselines + soft voting
2. `notebooks/02_stacking_ensemble.ipynb` — stacking extension (P7C/P6C)
3. `notebooks/03_meta_learner_selection_multiclass.ipynb` — Friedman/Nemenyi selection
4. `notebooks/04_meta_learner_selection_binary.ipynb` — binary (Normal vs Theft) variant

Or run the script baselines headlessly:

```bash
python -m src.knn_model
python -m src.rf_model
python -m src.xgb_model
```

All notebook outputs (CSVs, PNGs, PKLs) save into `results/`. Notebook outputs are stripped in this repo — they regenerate on run.

## Tech Stack

`Python 3.13` · `scikit-learn 1.9` · `XGBoost 3.3` · `pandas 3.0` · `NumPy 2.4` · `SciPy 1.18` (Friedman/Nemenyi) · `Matplotlib` / `Seaborn` · `scikit-posthocs` (optional, Nemenyi)

## Team — Group 11

| Name                | Roll No. |
| ------------------- | -------- |
| Vedant Chaugule     | A007     |
| Rajeshwari Majumdar | A033     |
| Sarrah Pittalwala   | A044     |
| Jigar Rambhiya      | A046     |
| Symprose Remedios   | A049     |

**Mentor:** Dr. Pradnya Khandeparker
**Institution:** Nilkamal School of Mathematics and Applied Statistics (NSoMASA), SVKM's NMIMS

## References

Mohammad, N., et al. (2023). *Ensemble Learning-Based Decision Support System for Energy Theft Detection in Smart Grid Environments.*
Zidi, S., et al. (2022). *Theft Detection Dataset for Benchmarking and ML Classification in a Smart Grid Environment.* J. King Saud Univ. DOI: 10.17632/c3c7329tjj.1
