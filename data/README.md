# Dataset

This project uses the **TDD2022 (Theft Detection Dataset 2022)**, introduced by Zidi et al. (2022).

- **Download:** [Mendeley Data — DOI: 10.17632/c3c7329tjj.1](https://doi.org/10.17632/c3c7329tjj.1)
- **Size:** ~65MB (not committed to this repo)
- **Instances:** 560,640
- **Consumer types:** 17 values in the `Class` column (e.g. FullServiceRestaurant, Hospital, …)
- **Features:** 10 numerical hourly consumption columns + `Class`
- **Target:** `theft` column with 7 values: `Normal`, `Theft1` … `Theft6`
- **Dropped:** the leading `0` index column

Original consumption records were sourced from the [Open Energy Data Initiative (OEDI)](https://data.openei.org/) platform maintained by the U.S. Department of Energy.

## Setup

1. Download the dataset from the link above.
2. Place the file in this `data/` folder as **`Dataset_actual.csv`**
   (all notebooks and `src/` default to `data/Dataset_actual.csv`).
3. Run notebooks in order: `01` → `02` → `03` → `04`.
