# U.S. Smoking Cost & Revenue Analysis

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-choropleth-3F4F75?logo=plotly&logoColor=white) ![License](https://img.shields.io/github/license/PatelSahil-HDS/Smoking-Cost-Revenue-Analysis)

State-level analysis of the gap between smoking-attributable healthcare costs and cigarette tax revenue across U.S. states (2005–2009). Built in a single Jupyter notebook with pandas, matplotlib, and plotly choropleth maps.

## What's in here

- `smoking_cost_revenue_analysis.ipynb` — the full analysis: cleaning, aggregation, correlation, and choropleth maps
- `smoking_attributable_expenditures_sammec_2024.csv` — CDC SAMMEC state-level smoking-attributable expenditures
- `cigarette_tax_revenue_orzechowski_walker_2024.csv` — Orzechowski & Walker cigarette tax revenue by state
- `US.json` — U.S. states GeoJSON for the choropleth maps

## Findings

- Smoking-attributable healthcare costs rose steadily from 2005 to 2009, with hospital expenses as the largest single component.
- Cigarette tax revenue rose over the same period but stayed far below total costs — every state ran a "smoking deficit."
- State-level correlation between cost and revenue sits at roughly **r = 0.85–0.97**, which is mostly a population-size effect: bigger states spend more *and* collect more.
- The cost-to-revenue ratio is what actually varies — and it's the more useful policy metric.

## Run it locally

```bash
git clone https://github.com/PatelSahil-HDS/Smoking-Cost-Revenue-Analysis.git
cd Smoking-Cost-Revenue-Analysis

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

jupyter notebook smoking_cost_revenue_analysis.ipynb
```

## Data sources

- **CDC** — [Smoking-Attributable Mortality, Morbidity, and Economic Costs (SAMMEC)](https://www.cdc.gov/tobacco/php/sammec/index.html)
- **Orzechowski & Walker** — *The Tax Burden on Tobacco*, gross cigarette tax revenue per state
- **U.S. Census** — state boundary GeoJSON

## Limitations

- The SAMMEC data covers 2005–2009 only — not a current view
- Revenue data is gross cigarette tax, doesn't include other tobacco products or federal excise
- Correlations don't control for state population or smoking prevalence
- Choropleth maps are descriptive, not predictive

## License

MIT — see [LICENSE](LICENSE).
