# GDP Distribution: Inclusive Growth Simulator

A simulation model and interactive dashboard for sharing India's projected 6.5 to 7.3% GDP growth evenly across sectors, demographic groups, states and income deciles, with the policy cost of doing so.

**Live dashboard:** https://rka0285058-afk.github.io/gdp-distribution-/ (needs GitHub Pages enabled for the `main` branch, root folder)

## Abstract

India's growth is projected at 6.5 to 7.3% a year, but trend growth does not spread evenly. Sectors differ widely in output per worker, and agriculture still employs over 40% of the workforce. This study builds a simulation model that asks which sector growth mix and labour reallocation would share the gains most evenly. It covers 9 sectors, 11 demographic groups, 14 states and regions, and 10 income deciles, from FY2027 to FY2047.

Each year an optimiser chooses sector growth rates and employment shifts. They must hit the aggregate growth target exactly, minimise differences in earnings growth across all groups, and stay close to trend growth. A cost layer adds the extra capital needed (through capital-output ratios), worker reallocation costs, and the cost of an equalising transfer scheme as an alternative.

On rounded illustrative inputs at 6.9% growth, the economy reaches about 4.1 times its FY2026 size under both trend and inclusive paths. The inclusive path raises the lowest decile's earnings multiple from 3.02 to about 3.20 and cuts the spread of earnings growth across groups from about 2% to under 0.5%. It does not narrow level inequality. The decile Gini stays near 0.57, and the richest-to-poorest state ratio, excluding Delhi, stays near 5. Restructuring costs about 2.5% of GDP a year as a gross upper bound, against about 0.25% for transfers that equalise growth rates.

The findings suggest that evenness of growth is cheap to approximate, but narrowing level gaps needs deliberately faster growth in lagging deciles and states.

## Important caveat

All inputs (sector shares, group employment mixes, state income indices, capital-output ratios, reallocation cost) are rounded placeholders at the scale of MoSPI, PLFS and state data. Read the patterns, not the decimals, until official series are loaded into the data block at the top of `model/india_inclusive_growth_model_v2.py`.

Known limits of the current model:
- Equal growth multiples preserve existing level gaps, so Gini and the state income ratio do not fall. Closing level gaps needs an objective that favours faster growth in lagging deciles and states.
- Investment cost is gross: it counts extra capital where sectors are pushed above trend and does not net out capital freed where others slow.

## What is in the repo

| Path | What it is |
|---|---|
| `index.html` | The dashboard (single page, Chart.js from a CDN) |
| `data.json` | Precomputed scenario grid: 9 growth targets x 5 inclusion pushes, year by year |
| `model/india_inclusive_growth_model.py` | v1: sectors and 11 demographic groups |
| `model/india_inclusive_growth_model_v2.py` | v2: adds states, deciles and the policy-cost layer |
| `model/gen_data.py` | Regenerates `data.json` from the v2 engine |
| `results/` | Scenario summaries, state and decile tables, cost frontier, charts |

## Run it

```bash
pip install numpy scipy matplotlib
python model/india_inclusive_growth_model_v2.py   # scenarios, tables and chart (writes to the current folder)
python model/gen_data.py                          # rebuild data.json for the dashboard
python -m http.server 8000                        # then open http://localhost:8000
```

The dashboard fetches `data.json`, so open it through a web server rather than as a local file.

## Dashboard

Controls: growth target (6.5 to 7.3%), inclusion push (Off to Max), horizon (FY2027 to FY2047, with playback), comparison lens (deciles, states, groups), and live cost assumptions (capital intensity, cost per worker moved). Trend growth is always the reference. Scenarios are precomputed on a grid; cost assumptions recalculate in the browser.
