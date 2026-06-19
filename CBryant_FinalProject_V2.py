"""
capstone_analysis.py
Christy Bryant | MIS581 Capstone | Colorado State University Global

This script analyzes the relationship between FinCEN human trafficking
Suspicious Activity Report (SAR) filings and federal and local enforcement
outcomes at the state-year level. It integrates four datasets:

    - FinCEN SAR Statistics (2020-2024)
    - Human Trafficking Institute federal prosecution data (2020-2023)
    - FBI Uniform Crime Reporting human trafficking arrest data (2013-2024)
    - U.S. Census Bureau state population estimates (2020-2023)

HOW TO USE
**********
1. Update the file paths in the FILE PATHS section below to match
   wherever your data files are stored on your machine.

2. Install dependencies if you haven't already (run once in terminal):
       pip install pandas scipy matplotlib numpy

3. Run the script:
       python capstone_analysis.py

4. All output files (CSVs and PNGs) are saved to OUTPUT_DIR.
   The script prints a summary of results to the console as it runs.

OUTPUT FILES
************
CSVs:
    sar_annual_summary.csv          - SAR descriptive stats by year
    sar_top10_states.csv            - Top 10 states by cumulative SAR count
    hti_national_totals.csv         - HTI prosecution counts by year
    hti_state_year.csv              - HTI data aggregated to state-year
    merged_sar_hti.csv              - Merged SAR + HTI dataset
    lag_analysis_t1.csv             - T+1 lag dataset (H2)
    h3_normalized_data.csv          - Per-capita data for H3
    h4_sar_fbi_data.csv             - Per-capita data for H4
    lag_analysis_fbi_extended.csv   - Extended lag results (T+0 through T+4)
    regression_results.csv          - OLS regression coefficients
    regional_correlations.csv       - Correlation results by Census region
    regional_summary.csv            - Mean rates by region

PNGs (12 figures):
    fig1_sar_trend.png                      - National SAR trend (2020-2024)
    fig2_hti_trend.png                      - HTI prosecution trend
    fig3_top10_states.png                   - Top 10 states by SAR count
    fig4_scatter_h1_raw.png                 - H1 scatter (raw counts)
    fig5_scatter_h3_normalized.png          - H3 scatter (per-capita)
    fig6_scatter_h4_fbi.png                 - H4 scatter (FBI arrests)
    fig7_lag_trend.png                      - Extended lag trend
    fig8_regional_rates.png                 - Regional SAR vs. arrest rates
    fig9_regional_correlations.png          - Regional correlation bar chart
    fig10_scatter_h2_lag.png                - H2 scatter (T+1 lag)
    fig11_nevada_sensitivity.png            - Nevada sensitivity analysis
    fig12_scatter_involuntary_servitude.png - Involuntary servitude scatter
"""

# ************************************************************
# FILE PATHS - update these to match your local directory
# ************************************************************
SAR_CSV_PATH       = "SARStats.csv"
HTI_NEW_CASES_PATH = "HIT_New_Cases.csv"
FBI_CSV_PATH       = "HT_2013_2024.csv"
CENSUS_POP_PATH    = "census_state_pop_2020_2023.csv"
SAR_2024_PATH      = "SARStats2024.csv"
OUTPUT_DIR         = "output"
# ************************************************************

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ************************************************************
# 1. LOAD & CLEAN SAR DATA (2020-2023)
# ************************************************************
print("Loading SAR data...")
sar_raw = pd.read_csv(SAR_CSV_PATH, dtype=str)
sar_raw.columns = [c.strip() for c in sar_raw.columns]
sar_raw.rename(columns={"Year Month": "Year"}, inplace=True)

sar_raw["Count"] = (
    sar_raw["Count"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
sar_raw["Count"] = pd.to_numeric(sar_raw["Count"], errors="coerce")

# Filter to one total row per state-year across all four breakdown dimensions.
# Requiring [Total] on all four avoids double-counting subtotals.
sar = sar_raw[
    (sar_raw["Instrument"]   == "[Total]") &
    (sar_raw["Regulator"]    == "[Total]") &
    (sar_raw["Relationship"] == "[Total]") &
    (sar_raw["Product"]      == "[Total]") &
    (~sar_raw["State"].isin(["[Total]", "Unknown"])) &
    (sar_raw["Year"]         != "All Years [Total]")
][["Year", "State", "Count"]].copy()

sar["Year"] = sar["Year"].astype(int)
sar = sar.rename(columns={"Year": "year", "State": "state", "Count": "sar_count"})
print(f"  SAR state-year observations: {len(sar)}")


# ************************************************************
# 2. LOAD HTI PROSECUTION DATA
# ************************************************************
print("Loading HTI prosecution data...")
hti_raw = pd.read_csv(HTI_NEW_CASES_PATH)

DISTRICT_TO_STATE = {
    "Alabama Middle": "Alabama", "Alabama Northern": "Alabama",
    "Alabama Southern": "Alabama", "Alaska": "Alaska",
    "Arizona": "Arizona", "Arkansas Eastern": "Arkansas",
    "Arkansas Western": "Arkansas", "California Central": "California",
    "California Eastern": "California", "California Northern": "California",
    "California Southern": "California", "Colorado": "Colorado",
    "Connecticut": "Connecticut", "Delaware": "Delaware",
    "District of Columbia": "District of Columbia",
    "Florida Middle": "Florida", "Florida Northern": "Florida",
    "Florida Southern": "Florida", "Georgia Middle": "Georgia",
    "Georgia Northern": "Georgia", "Georgia Southern": "Georgia",
    "Guam": "Guam", "Hawaii": "Hawaii", "Idaho": "Idaho",
    "Illinois Central": "Illinois", "Illinois Northern": "Illinois",
    "Illinois Southern": "Illinois", "Indiana Northern": "Indiana",
    "Indiana Southern": "Indiana", "Iowa Northern": "Iowa",
    "Iowa Southern": "Iowa", "Kansas": "Kansas",
    "Kentucky Eastern": "Kentucky", "Kentucky Western": "Kentucky",
    "Louisiana Eastern": "Louisiana", "Louisiana Middle": "Louisiana",
    "Louisiana Western": "Louisiana", "Maine": "Maine",
    "Maryland": "Maryland", "Massachusetts": "Massachusetts",
    "Michigan Eastern": "Michigan", "Michigan Western": "Michigan",
    "Minnesota": "Minnesota", "Mississippi Northern": "Mississippi",
    "Mississippi Southern": "Mississippi", "Missouri Eastern": "Missouri",
    "Missouri Western": "Missouri", "Montana": "Montana",
    "Nebraska": "Nebraska", "Nevada": "Nevada",
    "New Hampshire": "New Hampshire", "New Jersey": "New Jersey",
    "New Mexico": "New Mexico", "New York Eastern": "New York",
    "New York Northern": "New York", "New York Southern": "New York",
    "New York Western": "New York",
    "North Carolina Eastern": "North Carolina",
    "North Carolina Middle": "North Carolina",
    "North Carolina Western": "North Carolina",
    "North Dakota": "North Dakota",
    "Northern Mariana Islands": "Northern Mariana Islands",
    "Ohio Northern": "Ohio", "Ohio Southern": "Ohio",
    "Oklahoma Eastern": "Oklahoma", "Oklahoma Northern": "Oklahoma",
    "Oklahoma Western": "Oklahoma", "Oregon": "Oregon",
    "Pennsylvania Eastern": "Pennsylvania",
    "Pennsylvania Middle": "Pennsylvania",
    "Pennsylvania Western": "Pennsylvania",
    "Puerto Rico": "Puerto Rico", "Rhode Island": "Rhode Island",
    "South Carolina": "South Carolina", "South Dakota": "South Dakota",
    "Tennessee Eastern": "Tennessee", "Tennessee Middle": "Tennessee",
    "Tennessee Western": "Tennessee", "Texas Eastern": "Texas",
    "Texas Northern": "Texas", "Texas Southern": "Texas",
    "Texas Western": "Texas", "Utah": "Utah", "Vermont": "Vermont",
    "Virginia Eastern": "Virginia", "Virginia Western": "Virginia",
    "Virgin Islands": "Virgin Islands",
    "Washington Eastern": "Washington", "Washington Western": "Washington",
    "West Virginia Northern": "West Virginia",
    "West Virginia Southern": "West Virginia",
    "Wisconsin Eastern": "Wisconsin", "Wisconsin Western": "Wisconsin",
    "Wyoming": "Wyoming",
}

hti_raw["state"] = hti_raw["district"].map(DISTRICT_TO_STATE)

hti = (
    hti_raw
    .groupby(["state", "year"])[
        ["new_sex_trafficking_cases", "new_sex_trafficking_defendants",
         "new_forced_labor_cases",    "new_forced_labor_defendants",
         "new_cases_outside_ch77"]
    ]
    .sum()
    .reset_index()
)
print(f"  HTI state-year observations: {len(hti)}")


# ************************************************************
# 3. MERGE SAR + HTI
# ************************************************************
merged = sar.merge(hti, on=["state", "year"], how="inner")
print(f"  Merged observations (SAR + HTI): {len(merged)}")


# ************************************************************
# 4. DESCRIPTIVE STATISTICS - SAR
# ************************************************************
print("\n" + "*" * 60)
print("DESCRIPTIVE STATISTICS - SAR FILINGS")
print("*" * 60)

sar_annual = (
    sar.groupby("year")["sar_count"]
    .agg(total="sum", mean="mean", median="median",
         std="std", min="min", max="max", n_states="count")
    .reset_index()
)
print("\nAnnual SAR Summary (state-level distributions):")
print(sar_annual.round(1).to_string(index=False))

totals = sar.groupby("year")["sar_count"].sum()
print("\nYear-over-Year Change:")
for i in range(1, len(totals)):
    y0, y1 = totals.index[i-1], totals.index[i]
    pct = (totals[y1] - totals[y0]) / totals[y0] * 100
    print(f"  {y0} -> {y1}: {pct:+.1f}%")

top10 = (
    sar.groupby("state")["sar_count"].sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top10.columns = ["State", "Total SARs"]
print("\nTop 10 States - Cumulative SAR Count (2020-2023):")
print(top10.to_string(index=False))

sar_annual.to_csv(os.path.join(OUTPUT_DIR, "sar_annual_summary.csv"), index=False)
top10.to_csv(os.path.join(OUTPUT_DIR, "sar_top10_states.csv"), index=False)


# ************************************************************
# 5. DESCRIPTIVE STATISTICS - HTI PROSECUTIONS
# ************************************************************
print("\n" + "*" * 60)
print("DESCRIPTIVE STATISTICS - HTI FEDERAL PROSECUTIONS")
print("*" * 60)

hti_natl = (
    hti_raw.groupby("year")[
        ["new_sex_trafficking_cases", "new_sex_trafficking_defendants",
         "new_forced_labor_cases",    "new_forced_labor_defendants"]
    ]
    .sum()
    .reset_index()
)
print("\nNational Totals by Year:")
print(hti_natl.to_string(index=False))

hti_natl.to_csv(os.path.join(OUTPUT_DIR, "hti_national_totals.csv"), index=False)
hti.to_csv(os.path.join(OUTPUT_DIR, "hti_state_year.csv"), index=False)
merged.to_csv(os.path.join(OUTPUT_DIR, "merged_sar_hti.csv"), index=False)


# ************************************************************
# 6. H1 - CONTEMPORANEOUS CORRELATION (RAW COUNTS)
# ************************************************************
print("\n" + "*" * 60)
print("H1 - SAR COUNT vs. NEW SEX TRAFFICKING CASES (CONTEMPORANEOUS)")
print("*" * 60)

corr_data = merged[["sar_count", "new_sex_trafficking_cases"]].dropna()
pearson_r,  pearson_p  = stats.pearsonr( corr_data["sar_count"], corr_data["new_sex_trafficking_cases"])
spearman_r, spearman_p = stats.spearmanr(corr_data["sar_count"], corr_data["new_sex_trafficking_cases"])

print(f"\nState-year level  (n = {len(corr_data)})")
print(f"  Pearson  r = {pearson_r:.4f},  p = {pearson_p:.4f}")
print(f"  Spearman r = {spearman_r:.4f},  p = {spearman_p:.4f}")


# ************************************************************
# 7. H2 - T+1 LAG ANALYSIS
# ************************************************************
print("\n" + "*" * 60)
print("H2 - SAR(T) vs. NEW CASES(T+1)")
print("*" * 60)

sar_lagged = sar.copy()
sar_lagged["year"] = sar_lagged["year"] + 1

lag_df = sar_lagged.merge(
    hti[["state", "year", "new_sex_trafficking_cases"]],
    on=["state", "year"],
    how="inner"
).dropna()

if len(lag_df) > 2:
    lag_r,  lag_p  = stats.pearsonr( lag_df["sar_count"], lag_df["new_sex_trafficking_cases"])
    lag_sr, lag_sp = stats.spearmanr(lag_df["sar_count"], lag_df["new_sex_trafficking_cases"])
    print(f"\nSAR(T) vs. New Cases(T+1)  (n = {len(lag_df)})")
    print(f"  Pearson  r = {lag_r:.4f},  p = {lag_p:.4f}")
    print(f"  Spearman r = {lag_sr:.4f},  p = {lag_sp:.4f}")
    lag_df.to_csv(os.path.join(OUTPUT_DIR, "lag_analysis_t1.csv"), index=False)
else:
    print("  Not enough overlapping state-years for lag analysis.")


# ************************************************************
# 8. LOAD CENSUS POPULATION DATA
# ************************************************************
print("\nLoading Census population data...")
pop_raw = pd.read_csv(CENSUS_POP_PATH)
pop_long = pop_raw.melt(
    id_vars="State",
    value_vars=["Pop2020", "Pop2021", "Pop2022", "Pop2023"],
    var_name="year_str",
    value_name="population"
)
pop_long["year"] = pop_long["year_str"].str.replace("Pop", "").astype(int)
pop_long = pop_long.rename(columns={"State": "state"})[["state", "year", "population"]]

# Use 2023 population as proxy for 2024 since Census estimates lag by a year
pop_2024 = pop_long[pop_long["year"] == 2023].copy()
pop_2024["year"] = 2024
pop_long = pd.concat([pop_long, pop_2024], ignore_index=True)
print(f"  Population data loaded: {len(pop_long)} state-year rows")


# ************************************************************
# 9. H3 - POPULATION-NORMALIZED ANALYSIS (SAR vs HTI)
# ************************************************************
print("\n" + "*" * 60)
print("H3 - PER-CAPITA SAR RATE vs. PER-CAPITA PROSECUTION RATE")
print("*" * 60)

merged_h3 = merged.merge(pop_long, on=["state", "year"], how="inner")
merged_h3["sar_rate_per100k"]  = merged_h3["sar_count"]                / merged_h3["population"] * 100_000
merged_h3["case_rate_per100k"] = merged_h3["new_sex_trafficking_cases"] / merged_h3["population"] * 100_000
h3_data = merged_h3[["sar_rate_per100k", "case_rate_per100k"]].dropna()

h3_pearson_r,  h3_pearson_p  = stats.pearsonr( h3_data["sar_rate_per100k"], h3_data["case_rate_per100k"])
h3_spearman_r, h3_spearman_p = stats.spearmanr(h3_data["sar_rate_per100k"], h3_data["case_rate_per100k"])

print(f"\nPer-capita SAR rate vs. per-capita prosecution rate  (n = {len(h3_data)})")
print(f"  Pearson  r = {h3_pearson_r:.4f},  p = {h3_pearson_p:.4f}")
print(f"  Spearman r = {h3_spearman_r:.4f},  p = {h3_spearman_p:.4f}")
print(f"  Result: H30 {'REJECTED' if h3_pearson_p < 0.05 else 'RETAINED'}")

dropped_h3 = set(merged["state"].unique()) - set(merged_h3["state"].unique())
if dropped_h3:
    print(f"  States dropped (no Census match): {sorted(dropped_h3)}")

merged_h3.to_csv(os.path.join(OUTPUT_DIR, "h3_normalized_data.csv"), index=False)


# ************************************************************
# 10. LOAD FBI ARREST DATA
# ************************************************************
print("\nLoading FBI arrest data...")
fbi_raw = pd.read_csv(FBI_CSV_PATH)
fbi = (
    fbi_raw[
        (fbi_raw["OFFENSE_SUBCAT_NAME"] == "Commercial Sex Acts") &
        (fbi_raw["STATE_NAME"] != "Federal")
    ]
    .groupby(["STATE_NAME", "DATA_YEAR"])["ACTUAL_COUNT"]
    .sum()
    .reset_index()
    .rename(columns={"STATE_NAME": "state", "DATA_YEAR": "year", "ACTUAL_COUNT": "fbi_csa_count"})
)
print(f"  FBI state-year observations: {len(fbi)}")
print(f"  FBI years: {sorted(fbi['year'].unique())}")


# ************************************************************
# 11. H4 - SAR RATE vs. FBI ARREST RATE (PER CAPITA)
# ************************************************************
print("\n" + "*" * 60)
print("H4 - PER-CAPITA SAR RATE vs. PER-CAPITA FBI ARREST RATE")
print("*" * 60)

merged_h4 = merged.merge(fbi, on=["state", "year"], how="inner")
merged_h4 = merged_h4.merge(pop_long, on=["state", "year"], how="inner")
merged_h4["sar_rate"] = merged_h4["sar_count"]     / merged_h4["population"] * 100_000
merged_h4["fbi_rate"] = merged_h4["fbi_csa_count"] / merged_h4["population"] * 100_000
h4_data = merged_h4[["sar_rate", "fbi_rate"]].dropna()

h4_pearson_r,  h4_pearson_p  = stats.pearsonr( h4_data["sar_rate"], h4_data["fbi_rate"])
h4_spearman_r, h4_spearman_p = stats.spearmanr(h4_data["sar_rate"], h4_data["fbi_rate"])

print(f"\nPer-capita SAR rate vs. per-capita FBI arrest rate  (n = {len(h4_data)})")
print(f"  Pearson  r = {h4_pearson_r:.4f},  p = {h4_pearson_p:.4f}")
print(f"  Spearman r = {h4_spearman_r:.4f},  p = {h4_spearman_p:.4f}")
print(f"  Result: H40 {'REJECTED' if h4_pearson_p < 0.05 else 'RETAINED'}")
merged_h4.to_csv(os.path.join(OUTPUT_DIR, "h4_sar_fbi_data.csv"), index=False)


# ************************************************************
# 12. REGIONAL MAPPING
# ************************************************************
REGION_MAP = {
    'Connecticut':'Northeast','Maine':'Northeast','Massachusetts':'Northeast',
    'New Hampshire':'Northeast','Rhode Island':'Northeast','Vermont':'Northeast',
    'New Jersey':'Northeast','New York':'Northeast','Pennsylvania':'Northeast',
    'Illinois':'Midwest','Indiana':'Midwest','Michigan':'Midwest',
    'Ohio':'Midwest','Wisconsin':'Midwest','Iowa':'Midwest',
    'Kansas':'Midwest','Minnesota':'Midwest','Missouri':'Midwest',
    'Nebraska':'Midwest','North Dakota':'Midwest','South Dakota':'Midwest',
    'Delaware':'South','Florida':'South','Georgia':'South',
    'Maryland':'South','North Carolina':'South','South Carolina':'South',
    'Virginia':'South','District of Columbia':'South','West Virginia':'South',
    'Alabama':'South','Kentucky':'South','Mississippi':'South',
    'Tennessee':'South','Arkansas':'South','Louisiana':'South',
    'Oklahoma':'South','Texas':'South',
    'Arizona':'West','Colorado':'West','Idaho':'West','Montana':'West',
    'Nevada':'West','New Mexico':'West','Utah':'West','Wyoming':'West',
    'Alaska':'West','California':'West','Hawaii':'West',
    'Oregon':'West','Washington':'West',
}

fbi_is = (
    fbi_raw[
        (fbi_raw["OFFENSE_SUBCAT_NAME"] == "Involuntary Servitude") &
        (fbi_raw["STATE_NAME"] != "Federal")
    ]
    .groupby(["STATE_NAME", "DATA_YEAR"])["ACTUAL_COUNT"]
    .sum()
    .reset_index()
    .rename(columns={"STATE_NAME": "state", "DATA_YEAR": "year", "ACTUAL_COUNT": "fbi_is_count"})
)

full = merged_h4.copy()
full = full.merge(fbi_is, on=["state", "year"], how="left")
full["fbi_is_count"] = full["fbi_is_count"].fillna(0)
full["fbi_is_rate"]  = full["fbi_is_count"] / full["population"] * 100_000
full["hti_rate"]     = full["new_sex_trafficking_cases"] / full["population"] * 100_000
full["region"]       = full["state"].map(REGION_MAP).fillna("Territory")


# ************************************************************
# 13. MULTIVARIATE OLS REGRESSION
# ************************************************************
print("\n" + "*" * 60)
print("MULTIVARIATE OLS REGRESSION")
print("Dependent variable: HTI prosecution rate per 100k")
print("*" * 60)

reg_data = full[["sar_rate", "fbi_rate", "population", "hti_rate"]].dropna()
reg_data = reg_data[reg_data["population"] > 0]

X = reg_data[["sar_rate", "fbi_rate", "population"]].values
y = reg_data["hti_rate"].values
X_c = np.column_stack([np.ones(len(X)), X])
beta = np.linalg.lstsq(X_c, y, rcond=None)[0]
y_pred = X_c @ beta
residuals = y - y_pred
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot

n_reg, k_reg = X_c.shape
mse = ss_res / (n_reg - k_reg)
var_beta = mse * np.linalg.inv(X_c.T @ X_c)
se_beta = np.sqrt(np.diag(var_beta))
t_stats = beta / se_beta
p_values = [2 * (1 - stats.t.cdf(abs(t), df=n_reg - k_reg)) for t in t_stats]

labels = ["Intercept", "SAR rate per 100k", "FBI arrest rate per 100k", "Population"]
print(f"\n  {'Variable':<30} {'beta':>8} {'SE':>8} {'t':>8} {'p':>8} {'Sig':>4}")
print("  " + "-" * 68)
for lbl, b, se, t, p in zip(labels, beta, se_beta, t_stats, p_values):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {lbl:<30} {b:>8.4f} {se:>8.4f} {t:>8.3f} {p:>8.4f} {sig:>4}")
print(f"\n  R-squared = {r_squared:.4f}  |  n = {n_reg}")

reg_df = pd.DataFrame({
    "variable": labels, "beta": beta, "se": se_beta,
    "t": t_stats, "p": p_values
})
reg_df.to_csv(os.path.join(OUTPUT_DIR, "regression_results.csv"), index=False)


# ************************************************************
# 14. REGIONAL ANALYSIS
# ************************************************************
print("\n" + "*" * 60)
print("REGIONAL ANALYSIS")
print("*" * 60)

core_regions = ["Northeast", "Midwest", "South", "West"]
reg_summary = (
    full[full["region"].isin(core_regions)]
    .groupby("region")
    .agg(
        n=("state", "count"),
        avg_sar_rate=("sar_rate", "mean"),
        avg_fbi_csa_rate=("fbi_rate", "mean"),
        avg_fbi_is_rate=("fbi_is_rate", "mean"),
        avg_hti_rate=("hti_rate", "mean"),
    )
    .round(4)
)
print("\nMean per-capita rates by region (2020-2023):")
print(reg_summary.to_string())

print("\nCorrelations by region - SAR rate vs outcomes:")
print(f"\n  {'Region':<12} {'n':>4}  {'vs FBI arrests':>16}  {'vs HTI prosecution':>20}")
print("  " + "-" * 58)
regional_corr = []
for region in core_regions:
    rdf = full[full["region"] == region][["sar_rate", "fbi_rate", "hti_rate"]].dropna()
    if len(rdf) > 5:
        pr_fbi, pp_fbi = stats.pearsonr(rdf["sar_rate"], rdf["fbi_rate"])
        pr_hti, pp_hti = stats.pearsonr(rdf["sar_rate"], rdf["hti_rate"])
        sig_fbi = "*" if pp_fbi < 0.05 else " "
        sig_hti = "*" if pp_hti < 0.05 else " "
        print(f"  {region:<12} {len(rdf):>4}  r={pr_fbi:>6.4f} p={pp_fbi:.4f}{sig_fbi}  r={pr_hti:>6.4f} p={pp_hti:.4f}{sig_hti}")
        regional_corr.append({
            "region": region, "n": len(rdf),
            "pearson_r_fbi": pr_fbi, "p_fbi": pp_fbi,
            "pearson_r_hti": pr_hti, "p_hti": pp_hti
        })

pd.DataFrame(regional_corr).to_csv(os.path.join(OUTPUT_DIR, "regional_correlations.csv"), index=False)
reg_summary.to_csv(os.path.join(OUTPUT_DIR, "regional_summary.csv"))

print("\nInvoluntary Servitude - SAR rate vs FBI IS arrest rate:")
is_data = full[full["fbi_is_rate"] > 0][["sar_rate", "fbi_is_rate"]].dropna()
pr_is, pp_is = stats.pearsonr(is_data["sar_rate"], is_data["fbi_is_rate"])
sr_is, sp_is = stats.spearmanr(is_data["sar_rate"], is_data["fbi_is_rate"])
print(f"  n = {len(is_data)}")
print(f"  Pearson  r = {pr_is:.4f},  p = {pp_is:.4f}")
print(f"  Spearman r = {sr_is:.4f},  p = {sp_is:.4f}")


# ************************************************************
# 15. EXTENDED LAG ANALYSIS - SAR(T) vs FBI ARRESTS(T+N)
# ************************************************************
print("\n" + "*" * 60)
print("EXTENDED LAG ANALYSIS - SAR(T) vs. FBI ARRESTS(T+N)")
print("*" * 60)

sar24_raw = pd.read_csv(SAR_2024_PATH, dtype=str)
sar24_raw.columns = [c.strip() for c in sar24_raw.columns]
sar24_raw["Count"] = (
    sar24_raw["Count"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
sar24_raw["Count"] = pd.to_numeric(sar24_raw["Count"], errors="coerce")
sar24_raw["Year Month"] = sar24_raw["Year Month"].str.strip()

sar24 = sar24_raw[
    (sar24_raw["Instrument"]   == "[Total]") &
    (sar24_raw["Regulator"]    == "[Total]") &
    (sar24_raw["Relationship"] == "[Total]") &
    (sar24_raw["Product"]      == "[Total]") &
    (~sar24_raw["State"].isin(["[Total]", "Unknown"])) &
    (sar24_raw["Year Month"]   == "2024")
][["Year Month", "State", "Count"]].copy()
sar24 = sar24.rename(columns={"Year Month": "year", "State": "state", "Count": "sar_count"})
sar24["year"] = 2024

sar_full = pd.concat([sar[["state", "year", "sar_count"]], sar24], ignore_index=True)
print(f"\nFull SAR dataset (2020-2024): {len(sar_full)} state-year observations")
print("\nNational totals by year:")
for yr, tot in sar_full.groupby("year")["sar_count"].sum().items():
    print(f"  {yr}: {int(tot):,}")

print(f"\n{'Lag':<8} {'n':>5} {'Pearson r':>10} {'p':>10} {'Spearman r':>12} {'p':>10} {'Sig?':>6}")
print("-" * 60)

lag_results = {}
for lag in [0, 1, 2, 3, 4]:
    sar_lagged = sar_full.copy()
    sar_lagged["year"] = sar_lagged["year"] + lag
    tmp = sar_lagged.merge(fbi, on=["state", "year"], how="inner")
    tmp = tmp.merge(pop_long, on=["state", "year"], how="inner")
    tmp["sar_rate"] = tmp["sar_count"]     / tmp["population"] * 100_000
    tmp["fbi_rate"] = tmp["fbi_csa_count"] / tmp["population"] * 100_000
    tmp = tmp.dropna(subset=["sar_rate", "fbi_rate"])
    if len(tmp) > 2:
        pr, pp = stats.pearsonr( tmp["sar_rate"], tmp["fbi_rate"])
        sr, sp = stats.spearmanr(tmp["sar_rate"], tmp["fbi_rate"])
        sig = "YES" if pp < 0.05 else "no"
        lag_results[lag] = {"n": len(tmp), "pearson_r": pr, "pearson_p": pp,
                            "spearman_r": sr, "spearman_p": sp}
        print(f"T+{lag:<6} {len(tmp):>5} {pr:>10.4f} {pp:>10.4f} {sr:>12.4f} {sp:>10.4f} {sig:>6}")

lag_df_results = pd.DataFrame(lag_results).T
lag_df_results.index.name = "lag"
lag_df_results.to_csv(os.path.join(OUTPUT_DIR, "lag_analysis_fbi_extended.csv"))


# ************************************************************
# 16. FIGURES
# ************************************************************
print("\nGenerating figures...")

# Figure 1 - National SAR trend (2020-2024)
fig, ax = plt.subplots(figsize=(8, 4))
ann = sar_full.groupby("year")["sar_count"].sum().reset_index()
ax.bar(ann["year"].astype(str), ann["sar_count"], color="#2c7bb6", edgecolor="white")
ax.set_title("National Human Trafficking SAR Filings by Year (2020-2024)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Total SAR Filings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, val in zip(ax.patches, ann["sar_count"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
            f"{int(val):,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_sar_trend.png"), dpi=150)
plt.close()

# Figure 2 - HTI prosecution trend
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hti_natl["year"].astype(str), hti_natl["new_sex_trafficking_cases"],
        marker="o", color="#d7191c", linewidth=2, label="Sex Trafficking")
ax.plot(hti_natl["year"].astype(str), hti_natl["new_forced_labor_cases"],
        marker="s", color="#fdae61", linewidth=2, linestyle="--", label="Forced Labor")
ax.set_title("New Federal Human Trafficking Cases Filed by Year",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("New Cases Filed")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_hti_trend.png"), dpi=150)
plt.close()

# Figure 3 - Top 10 states by cumulative SAR count
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(top10["State"][::-1], top10["Total SARs"][::-1], color="#1a9641", edgecolor="white")
ax.set_title("Top 10 States: Cumulative HT SAR Filings 2020-2023",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Total SAR Filings")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_top10_states.png"), dpi=150)
plt.close()

# Figure 4 - H1 scatter (raw counts)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(corr_data["sar_count"], corr_data["new_sex_trafficking_cases"],
           alpha=0.4, color="#7b2d8b", edgecolors="none", s=30)
m, b = np.polyfit(corr_data["sar_count"], corr_data["new_sex_trafficking_cases"], 1)
x_line = np.linspace(corr_data["sar_count"].min(), corr_data["sar_count"].max(), 200)
ax.plot(x_line, m * x_line + b, color="#d7191c", linewidth=1.5, linestyle="--",
        label=f"OLS fit  (r = {pearson_r:.3f}, p < 0.001)")
ax.set_title("H1: SAR Filings vs. New Sex Trafficking Cases\n(State-Year Level, Raw Counts)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("SAR Count")
ax.set_ylabel("New Sex Trafficking Cases")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_scatter_h1_raw.png"), dpi=150)
plt.close()

# Figure 5 - H3 per-capita scatter
h3_plot = h3_data.dropna()
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(h3_plot["sar_rate_per100k"], h3_plot["case_rate_per100k"],
           alpha=0.4, color="#2c7bb6", edgecolors="none", s=30)
m3, b3 = np.polyfit(h3_plot["sar_rate_per100k"], h3_plot["case_rate_per100k"], 1)
x3 = np.linspace(h3_plot["sar_rate_per100k"].min(), h3_plot["sar_rate_per100k"].max(), 200)
ax.plot(x3, m3 * x3 + b3, color="#d7191c", linewidth=1.5, linestyle="--",
        label=f"OLS fit  (r = {h3_pearson_r:.3f}, p = {h3_pearson_p:.3f})")
ax.set_title("H3: Per-Capita SAR Rate vs. Per-Capita Prosecution Rate\n(State-Year Level)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("SAR Filings per 100,000 Residents")
ax.set_ylabel("New Sex Trafficking Cases per 100,000 Residents")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_scatter_h3_normalized.png"), dpi=150)
plt.close()

# Figure 6 - H4 SAR rate vs FBI arrest rate
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(h4_data["sar_rate"], h4_data["fbi_rate"],
           alpha=0.4, color="#1a9641", edgecolors="none", s=30)
m4, b4 = np.polyfit(h4_data["sar_rate"], h4_data["fbi_rate"], 1)
x4 = np.linspace(h4_data["sar_rate"].min(), h4_data["sar_rate"].max(), 200)
ax.plot(x4, m4 * x4 + b4, color="#d7191c", linewidth=1.5, linestyle="--",
        label=f"OLS fit  (r = {h4_pearson_r:.3f}, p < 0.001)")
ax.set_title("H4: Per-Capita SAR Rate vs. Per-Capita FBI Arrest Rate\n(State-Year Level)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("SAR Filings per 100,000 Residents")
ax.set_ylabel("FBI Human Trafficking Arrests per 100,000 Residents")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_scatter_h4_fbi.png"), dpi=150)
plt.close()

# Figure 7 - Extended lag trend
fig, ax = plt.subplots(figsize=(7, 4))
lags = list(lag_results.keys())
rs   = [lag_results[l]["pearson_r"] for l in lags]
ax.plot([f"T+{l}" for l in lags], rs, marker="o", color="#2c7bb6", linewidth=2)
ax.set_title("SAR-to-FBI Arrest Correlation by Lag Year\n(Pearson r, per-capita rates)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Lag")
ax.set_ylabel("Pearson r")
ax.set_ylim(0, 0.7)
ax.axhline(0.3, color="gray", linestyle="--", linewidth=0.8, label="r = 0.30 reference")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_lag_trend.png"), dpi=150)
plt.close()

# Figure 8 - Regional mean SAR rate vs FBI arrest rate (grouped bar)
regions_plot = reg_summary.loc[core_regions].reset_index()
x = np.arange(len(regions_plot))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, regions_plot["avg_sar_rate"],          width, label="SAR rate per 100k",    color="#2c7bb6")
ax.bar(x + width/2, regions_plot["avg_fbi_csa_rate"] * 10, width, label="FBI arrest rate x 10", color="#1a9641")
ax.set_xticks(x)
ax.set_xticklabels(regions_plot["region"])
ax.set_title("Regional Mean SAR Rate vs. FBI Arrest Rate per 100k\n(FBI rate scaled x10 for visibility)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Rate per 100,000 Residents")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig8_regional_rates.png"), dpi=150)
plt.close()

# Figure 9 - Regional SAR-to-FBI correlation bar chart
reg_corr_df = pd.DataFrame(regional_corr)
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#2c7bb6" if p < 0.05 else "#aaaaaa" for p in reg_corr_df["p_fbi"]]
ax.bar(reg_corr_df["region"], reg_corr_df["pearson_r_fbi"], color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("SAR Rate vs. FBI Arrest Rate Correlation by Region\n(blue = p < 0.05)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Pearson r")
ax.set_ylim(-0.2, 0.7)
for i, (r, p) in enumerate(zip(reg_corr_df["pearson_r_fbi"], reg_corr_df["p_fbi"])):
    ax.text(i, r + 0.02, f"r={r:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig9_regional_correlations.png"), dpi=150)
plt.close()

# Figure 10 - H2 scatter: SAR(T) vs prosecution(T+1)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(lag_df["sar_count"], lag_df["new_sex_trafficking_cases"],
           alpha=0.4, color="#7b2d8b", edgecolors="none", s=30)
m2, b2 = np.polyfit(lag_df["sar_count"], lag_df["new_sex_trafficking_cases"], 1)
x2 = np.linspace(lag_df["sar_count"].min(), lag_df["sar_count"].max(), 200)
ax.plot(x2, m2 * x2 + b2, color="#d7191c", linewidth=1.5, linestyle="--",
        label=f"OLS fit  (r = {lag_r:.3f}, p < 0.001)")
ax.set_title("H2: SAR Filings (Year T) vs. New Sex Trafficking Cases (Year T+1)\n(State-Year Level, Raw Counts)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("SAR Count (Year T)")
ax.set_ylabel("New Sex Trafficking Cases (Year T+1)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig10_scatter_h2_lag.png"), dpi=150)
plt.close()

# Figure 11 - Nevada sensitivity: Western correlation with vs. without Nevada
west_data = full[full["region"] == "West"][["state", "sar_rate", "fbi_rate"]].dropna()
west_no_nv = west_data[west_data["state"] != "Nevada"]
r_west_all, _ = stats.pearsonr(west_data["sar_rate"], west_data["fbi_rate"])
r_west_no_nv, p_west_no_nv = stats.pearsonr(west_no_nv["sar_rate"], west_no_nv["fbi_rate"])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, data, title, r_val, p_val in zip(
    axes,
    [west_data, west_no_nv],
    ["West Region (All States)", "West Region (Nevada Excluded)"],
    [r_west_all, r_west_no_nv],
    [None, p_west_no_nv],
):
    nv = data[data["state"] == "Nevada"] if "Nevada" in data["state"].values else None
    non_nv = data[data["state"] != "Nevada"]
    ax.scatter(non_nv["sar_rate"], non_nv["fbi_rate"],
               alpha=0.5, color="#2c7bb6", edgecolors="none", s=40, label="Other states")
    if nv is not None and len(nv) > 0:
        ax.scatter(nv["sar_rate"], nv["fbi_rate"],
                   alpha=0.9, color="#d7191c", edgecolors="none", s=60, label="Nevada")
        ax.legend(fontsize=9)
    m_w, b_w = np.polyfit(data["sar_rate"], data["fbi_rate"], 1)
    x_w = np.linspace(data["sar_rate"].min(), data["sar_rate"].max(), 200)
    p_label = "< 0.001" if p_val is None else f"= {p_val:.3f}"
    ax.plot(x_w, m_w * x_w + b_w, color="#d7191c", linewidth=1.5, linestyle="--",
            label=f"r = {r_val:.3f}, p {p_label}")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("SAR Rate per 100,000")
    ax.set_ylabel("FBI Arrest Rate per 100,000")
    ax.legend(fontsize=9)
plt.suptitle("Nevada Sensitivity: Western Regional SAR-to-Arrest Correlation",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig11_nevada_sensitivity.png"), dpi=150, bbox_inches="tight")
plt.close()

# Figure 12 - Involuntary servitude scatter
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(is_data["sar_rate"], is_data["fbi_is_rate"],
           alpha=0.4, color="#fdae61", edgecolors="none", s=30)
m_is, b_is = np.polyfit(is_data["sar_rate"], is_data["fbi_is_rate"], 1)
x_is = np.linspace(is_data["sar_rate"].min(), is_data["sar_rate"].max(), 200)
ax.plot(x_is, m_is * x_is + b_is, color="#d7191c", linewidth=1.5, linestyle="--",
        label=f"OLS fit  (r = {pr_is:.3f}, p = {pp_is:.3f})")
ax.set_title("Per-Capita SAR Rate vs. Per-Capita FBI Involuntary Servitude Arrest Rate\n(State-Year Level)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("SAR Filings per 100,000 Residents")
ax.set_ylabel("FBI Involuntary Servitude Arrests per 100,000 Residents")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig12_scatter_involuntary_servitude.png"), dpi=150)
plt.close()


print("\nDone. All outputs saved to:", OUTPUT_DIR)
print("\nFiles written:")
output_files = [
    "sar_annual_summary.csv", "sar_top10_states.csv",
    "hti_national_totals.csv", "hti_state_year.csv",
    "merged_sar_hti.csv", "lag_analysis_t1.csv",
    "h3_normalized_data.csv", "h4_sar_fbi_data.csv",
    "lag_analysis_fbi_extended.csv",
    "regression_results.csv", "regional_correlations.csv", "regional_summary.csv",
    "fig1_sar_trend.png", "fig2_hti_trend.png",
    "fig3_top10_states.png", "fig4_scatter_h1_raw.png",
    "fig5_scatter_h3_normalized.png", "fig6_scatter_h4_fbi.png",
    "fig7_lag_trend.png", "fig8_regional_rates.png",
    "fig9_regional_correlations.png",
    "fig10_scatter_h2_lag.png", "fig11_nevada_sensitivity.png",
    "fig12_scatter_involuntary_servitude.png",
]
for f in output_files:
    print(f"  {f}")
