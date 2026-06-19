# Financial Intelligence and Human Trafficking Enforcement
## MIS581 Capstone Analysis | Christy Bryant | Colorado State University Global

This repository contains the Python analysis script for my MIS581 capstone project, which examines the relationship between FinCEN human trafficking Suspicious Activity Report (SAR) filings and federal and local enforcement outcomes at the state-year level.

The full paper is titled *Financial Intelligence and Human Trafficking Enforcement: An Empirical Analysis of SAR Filing Correlations With Federal and Local Outcomes* and was submitted in partial fulfillment of the requirements for the M.S. in Data Analytics program at Colorado State University Global.

---

## What the Script Does

`capstone_analysis.py` integrates four publicly available datasets to test four formal hypotheses and two supplementary analyses:

- **H1** — Contemporaneous correlation between SAR filings and federal sex trafficking prosecution counts (raw counts)
- **H2** — T+1 lag analysis: SAR filings in year T as a predictor of prosecution counts in year T+1
- **H3** — Population-normalized analysis: per-capita SAR rate vs. per-capita prosecution rate
- **H4** — Per-capita SAR rate vs. per-capita FBI commercial sex act arrest rate
- **Extended lag analysis** — SAR-to-arrest correlation at T+0 through T+4
- **Multivariate OLS regression** — SAR rate, FBI arrest rate, and population as simultaneous predictors of per-capita prosecution rate

It also produces a regional analysis, a Nevada sensitivity analysis, and an involuntary servitude correlation. All results are printed to the console and saved as CSV and PNG files.

---

## Data Sources

The following datasets are required to run the script. All are publicly available at no cost.

| File | Source | URL |
|---|---|---|
| `SARStats.csv` | FinCEN SAR Stats (2020-2023) | https://www.fincen.gov/reports/sar-stats |
| `SARStats2024.csv` | FinCEN SAR Stats (2024) | https://www.fincen.gov/reports/sar-stats |
| `HIT_New_Cases.csv` | Human Trafficking Institute annual federal reports | https://traffickinginstitute.org/federal-human-trafficking-report/ |
| `HT_2013_2024.csv` | FBI Crime Data Explorer human trafficking data | https://cde.fbi.gov/dataexplorer |
| `census_state_pop_2020_2023.csv` | U.S. Census Bureau state population estimates | https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html |

### Data preparation notes

**FinCEN SAR data** — download the human trafficking category from the SAR Stats tool. Filter to [Total] on all four breakdown dimensions (Instrument, Regulator, Relationship, Product) before exporting. The script applies this filter programmatically, but the raw export must include those columns.

**HTI data** — aggregated manually from the district-level tables in the annual federal human trafficking reports (2020, 2021, 2022, 2023). The script maps districts to states using a built-in lookup table.

**FBI data** — download the human trafficking offense data from the Crime Data Explorer. The file should include `STATE_NAME`, `DATA_YEAR`, `OFFENSE_SUBCAT_NAME`, and `ACTUAL_COUNT` columns.

**Census data** — the file should be structured with one row per state and columns named `State`, `Pop2020`, `Pop2021`, `Pop2022`, `Pop2023`.

---

## Setup

Install dependencies (run once):

```bash
pip install pandas scipy matplotlib numpy
```

Place all five data files in the same directory as the script, then run:

```bash
python capstone_analysis.py
```

Output files are saved to an `output` folder in the same directory. The script creates the folder automatically if it does not exist.

---

## Output Files

**CSVs:**
- `sar_annual_summary.csv` — SAR descriptive statistics by year
- `sar_top10_states.csv` — top 10 states by cumulative SAR count
- `hti_national_totals.csv` — HTI prosecution counts by year
- `hti_state_year.csv` — HTI data aggregated to state-year
- `merged_sar_hti.csv` — merged SAR and HTI dataset
- `lag_analysis_t1.csv` — T+1 lag dataset (H2)
- `h3_normalized_data.csv` — per-capita data for H3
- `h4_sar_fbi_data.csv` — per-capita data for H4
- `lag_analysis_fbi_extended.csv` — extended lag results (T+0 through T+4)
- `regression_results.csv` — OLS regression coefficients
- `regional_correlations.csv` — correlation results by Census region
- `regional_summary.csv` — mean rates by region

**PNGs (12 figures):**
- `fig1_sar_trend.png` — national SAR trend (2020-2024)
- `fig2_hti_trend.png` — HTI prosecution trend
- `fig3_top10_states.png` — top 10 states by SAR count
- `fig4_scatter_h1_raw.png` — H1 scatter (raw counts)
- `fig5_scatter_h3_normalized.png` — H3 scatter (per-capita)
- `fig6_scatter_h4_fbi.png` — H4 scatter (FBI arrests)
- `fig7_lag_trend.png` — extended lag trend
- `fig8_regional_rates.png` — regional SAR vs. arrest rates
- `fig9_regional_correlations.png` — regional correlation bar chart
- `fig10_scatter_h2_lag.png` — H2 scatter (T+1 lag)
- `fig11_nevada_sensitivity.png` — Nevada sensitivity analysis
- `fig12_scatter_involuntary_servitude.png` — involuntary servitude scatter

---

## Tableau Dashboard

An interactive visualization of the findings is published at:
https://public.tableau.com/views/CBryant_HT/Dashboard1

---

## Citation

Bryant, C. (2026). *Financial intelligence and human trafficking enforcement dashboard* [Tableau dashboard]. Tableau Public. https://public.tableau.com/views/CBryant_HT/Dashboard1
