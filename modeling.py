"""
Predictive modeling on the smoking cost vs revenue data.

The descriptive notebook shows that cost and revenue are highly correlated
(r ≈ 0.85–0.97) — but that's mostly a population-size effect. The useful
question is: which states have the worst cost-to-revenue ratio, and can
we predict that ratio from observable state characteristics?

This script:
  1. Cleans the SAMMEC and tax revenue tables
  2. Joins them by state and year
  3. Computes cost-to-revenue ratio (the "smoking deficit multiple")
  4. Fits a Random Forest regressor predicting the ratio
  5. Reports feature importances + cross-validated performance

Run:
    python modeling.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

ROOT = Path(__file__).resolve().parent
SAMMEC = ROOT / "smoking_attributable_expenditures_sammec_2024.csv"
TAX = ROOT / "cigarette_tax_revenue_orzechowski_walker_2024.csv"
SEED = 42


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """SAMMEC and tax-burden CSVs sometimes have BOM, mixed encoding, or footer rows."""
    return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")


def load_costs() -> pd.DataFrame:
    df = _read_csv_robust(SAMMEC)
    df.columns = [c.strip() for c in df.columns]
    # Standardize the likely state column to "State"
    for cand in ["State", "STATE", "state", "Location"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "State"})
            break
    # Standardize a year column to "Year"
    for cand in ["Year", "YEAR", "year"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "Year"})
            break
    return df


def load_revenue() -> pd.DataFrame:
    df = _read_csv_robust(TAX)
    df.columns = [c.strip() for c in df.columns]
    for cand in ["State", "STATE", "state", "Location"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "State"})
            break
    for cand in ["Year", "YEAR", "year"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "Year"})
            break
    return df


def _find_numeric_total_col(df: pd.DataFrame, hints: list[str]) -> str | None:
    """Heuristic: prefer the most aggregated/total dollar column."""
    candidates = [c for c in df.columns
                  if any(h.lower() in c.lower() for h in hints)
                  and pd.api.types.is_numeric_dtype(df[c])]
    if not candidates:
        # Fall back to the largest-magnitude numeric column
        nums = df.select_dtypes(include="number")
        if nums.shape[1] == 0:
            return None
        candidates = [nums.mean().idxmax()]
    return candidates[0]


def build_panel(costs: pd.DataFrame, rev: pd.DataFrame) -> pd.DataFrame:
    cost_col = _find_numeric_total_col(costs, ["Total", "Expenditure", "Cost", "SAE"])
    rev_col = _find_numeric_total_col(rev, ["Total", "Revenue", "Tax", "Gross"])

    if cost_col is None or rev_col is None:
        raise RuntimeError("Could not locate cost / revenue columns. "
                           "Open the CSVs and update the hint lists in modeling.py.")

    print(f"Cost column:    {cost_col}")
    print(f"Revenue column: {rev_col}")

    cost_long = costs[["State", "Year", cost_col]].dropna()
    rev_long = rev[["State", "Year", rev_col]].dropna()
    cost_long = cost_long.rename(columns={cost_col: "Total_Cost"})
    rev_long = rev_long.rename(columns={rev_col: "Total_Revenue"})

    panel = cost_long.merge(rev_long, on=["State", "Year"], how="inner")
    panel = panel[(panel["Total_Cost"] > 0) & (panel["Total_Revenue"] > 0)]
    panel["Cost_to_Revenue"] = panel["Total_Cost"] / panel["Total_Revenue"]
    panel["log_cost"] = np.log(panel["Total_Cost"])
    panel["log_rev"] = np.log(panel["Total_Revenue"])

    # Census-region groupings — a crude but useful predictor
    region_map = {
        "Northeast": ["Connecticut", "Maine", "Massachusetts", "New Hampshire",
                      "Rhode Island", "Vermont", "New Jersey", "New York",
                      "Pennsylvania"],
        "Midwest":   ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin",
                      "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska",
                      "North Dakota", "South Dakota"],
        "South":     ["Delaware", "Florida", "Georgia", "Maryland",
                      "North Carolina", "South Carolina", "Virginia",
                      "West Virginia", "Alabama", "Kentucky", "Mississippi",
                      "Tennessee", "Arkansas", "Louisiana", "Oklahoma", "Texas",
                      "District of Columbia"],
        "West":      ["Arizona", "Colorado", "Idaho", "Montana", "Nevada",
                      "New Mexico", "Utah", "Wyoming", "Alaska", "California",
                      "Hawaii", "Oregon", "Washington"],
    }
    state_to_region = {s: r for r, states in region_map.items() for s in states}
    panel["Region"] = panel["State"].map(state_to_region).fillna("Other")

    return panel


def fit_model(panel: pd.DataFrame) -> None:
    feat = pd.get_dummies(
        panel[["log_cost", "log_rev", "Year", "Region"]],
        columns=["Region"], drop_first=True,
    ).astype(float)
    y = panel["Cost_to_Revenue"].astype(float)

    print(f"\nObservations: {len(panel)}  |  Features: {feat.shape[1]}")
    print(f"Cost-to-Revenue ratio: mean={y.mean():.2f}, "
          f"median={y.median():.2f}, max={y.max():.2f}")

    model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    model.fit(feat, y)
    pred = model.predict(feat)
    print(f"\nIn-sample R²:  {r2_score(y, pred):.3f}")
    print(f"In-sample MAE: {mean_absolute_error(y, pred):.3f}")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, feat, y, cv=cv, scoring="r2")
    print(f"5-fold CV R²:  mean={cv_scores.mean():.3f}  sd={cv_scores.std():.3f}")

    print("\nFeature importances (top 8):")
    imp = pd.Series(model.feature_importances_, index=feat.columns) \
            .sort_values(ascending=False).head(8)
    for name, val in imp.items():
        print(f"  {name:30s}  {val:.3f}")

    print("\nTop 10 states by mean cost-to-revenue ratio (worst smoking deficit):")
    print(panel.groupby("State")["Cost_to_Revenue"].mean()
              .sort_values(ascending=False).head(10).round(2))


def main() -> None:
    costs = load_costs()
    rev = load_revenue()
    panel = build_panel(costs, rev)
    fit_model(panel)


if __name__ == "__main__":
    main()
