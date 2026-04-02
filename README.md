# Human Capitalists in The Intangible Era

**Author:** Linus Lindquist
**Institution:** University of Copenhagen, Department of Economics
**Supervisor:** Søren Hove Ravn

---

## Abstract

This thesis studies the welfare incidence of capital income taxation in a two-sector general equilibrium model calibrated to Denmark, drawing closely on Gomez & Gouin-Bonenfant (2025). The central argument is that the secular rise of intangible investment has tightened cross-sectoral labor reallocation frictions, compressing the economy's effective capital supply elasticity and thereby altering who gains and who loses from a capital tax cut. Empirical evidence from Danish input-output and national accounts data documents rising intangible investment shares and diverging sectoral labor shares; calibrated simulations show that investment workers capture the largest welfare gains (~59%) from a permanent 10-percentage-point cut in the corporate income tax rate.

---

## Repository Structure

```
BA/
├── py_files/                   # Core Python library
│   ├── setup.py                # Shared imports, plot styles, notebook initialisation
│   ├── capinc_single.py        # CapIncModel_single — two-sector GE model (calibration,
│   │                           #   steady-state, dynamic simulation, welfare decomposition)
│   ├── elas.py                 # Closed-form demand/supply and wage-rent tax elasticities
│   ├── shocks.py               # Shock path constructors (permanent/temporary tax cuts)
│   ├── build_output_single.py  # Welfare decomposition; maps simulations to plot horizons
│   ├── IRF.py                  # IRF plotting (τ, q, pI, K, sK, sL, wages, welfare)
│   ├── sweep.py                # φ and εˢ parameter sweeps; elasticity grids; sweep figures
│   ├── QP_diagram.py           # Capital market equilibrium diagram for alternative φ values
│   ├── direct_NX.py            # Direct final-demand IO analysis from NAIO1F (DST API);
│   │                           #   per-year cache management (no Leontief inverse)
│   ├── LS_aggregator.py        # Sectoral labor shares: fetch NABP36, weight by IO shares,
│   │                           #   aggregate to C/I sectors; timeseries cache wrapper
│   ├── wage_employment.py      # Sectoral wage ratio wI/wC and employment ratio LI/LC
│   │                           #   using continuous IO weights; data from NABP36 & NABB36
│   └── var_groups.py           # Variable mappings: model ↔ IO notation; investment
│                               #   classification (structures, equipment, IP, org. capital)
│
├── 1_qp_mechanism.ipynb        # Capital market equilibrium diagram; Uzawa mechanism
├── 2_labour_shares.ipynb       # Sectoral labor share timeseries (1966–2024) with IO weights
├── 2_gross_investment.ipynb    # Intangible vs. tangible investment composition over time
├── 2_gini_v_itan.ipynb         # Gini coefficient vs. intangible investment share
├── 2_phi_argument.ipynb        # Empirical case for declining φ: wI/wC and LI/LC ratios
├── 3_IRF.ipynb                 # Impulse response functions for the baseline calibration
├── 3_phi_sweep.ipynb           # Sensitivity analysis: welfare and elasticities across φ
│
├── 0_intermediate/             # Cached intermediate outputs
│   └── direct_NX_cache/        # Per-year IO results (pickle) and timeseries (parquet)
├── 0_output/                   # Generated figures (PNG)
├── 0_raw_data/                 # Raw DST data files
└── 0_litterature/              # Literature (primary and secondary)
```

---

## Replication

### Dependencies

```
numpy
scipy
pandas
matplotlib
dstapi          # Statistics Denmark API wrapper
```

Install with:
```bash
pip install numpy scipy pandas matplotlib dstapi
```

### Running the notebooks

All notebooks begin with:
```python
from py_files.setup import *
setup_notebook()
```

This imports the full library, sets plot styles, and enables autoreload.

**Recommended execution order:**

| Step | Notebook | Notes |
|------|----------|-------|
| 1 | `2_labour_shares.ipynb` | Fetches DST data; cached after first run |
| 2 | `2_gross_investment.ipynb` | Investment composition from NAIO1F |
| 3 | `2_phi_argument.ipynb` | Wage/employment evidence for declining φ |
| 4 | `2_gini_v_itan.ipynb` | Distributional correlates |
| 5 | `1_qp_mechanism.ipynb` | Model mechanism diagram |
| 6 | `3_IRF.ipynb` | Baseline IRF simulation |
| 7 | `3_phi_sweep.ipynb` | Sensitivity sweep |

### Caching

IO computations (NAIO1F) are slow on first run. Subsequent runs load from `0_intermediate/direct_NX_cache/`. Use the `load_or_compute_*` wrappers:

```python
# Labour share timeseries — loads parquet on repeat runs
df_ts = sls.load_or_compute_ls_timeseries(range(1966, 2025), kappa=0.6)

# Wage/employment timeseries
df_we = we.load_or_compute_we_timeseries(range(1975, 2025), kappa=0.6)
```

Force a full refresh with `force=True`.

---

## Data Sources

All data are retrieved via the [Statistics Denmark API](https://www.dst.dk/en/Statistik/brug-statistikken/muligheder-i-statistikbanken/api) using the `dstapi` wrapper. Industry classification: **36a2** (Statistics Denmark).

| Table | Content | Used in |
|-------|---------|---------|
| NAIO1F | 69-industry symmetric input-output tables | Investment shares, IO weights |
| NABP36 | GVA and compensation of employees by industry | Labor shares |
| NABB36 | Hours worked and number of employees by industry | Wage and employment ratios |
| RAS310 | Employment by education and industry | Human capital composition |
| FORSK01 | Business R&D expenditure | Intangible capital proxy |
| INDKP201 | Household income by source | Distributional analysis |

---

## Model

The quantitative model (`CapIncModel_single`) is a small open economy with:

- **Two sectors:** consumption (C) and investment (I), with Cobb-Douglas production and potentially different factor intensities (αK, αL) and (βK, βL)
- **Capital accumulation** with convex adjustment costs; Tobin's q defined as q/p_I
- **Capital income tax** τ on corporate returns
- **Sectoral labor supply elasticity** φ — governs cross-sectoral reallocation frictions (not a Frisch elasticity)
- **Fixed world interest rate** (small open economy)
- **Baseline calibration:** φ = 0.75

The thesis proves that the Chamley-Judd zero-capital-tax result requires α = β; whenever factor intensities differ across sectors, the tax has first-order welfare effects even in the long run.

### Methodological note

GG-B construct sectoral labor shares by consolidating industry shares through the full Leontief inverse, distributing intermediate-input labor proportionally. This paper instead applies IO weights directly to observed industry-level labor shares from NABP36. The deviation is data-driven: Statistics Denmark's industry classification (36a2) does not provide the sub-industry granularity needed for full Leontief consolidation without material measurement error.

---

## Reference

Gomez, M. & Gouin-Bonenfant, E. (2025). *Human Capitalists.* Working paper.

---

## Citation

```
Lindquist, L. (2026). Human Capitalists in The Intangible Era.
Bachelor's thesis, University of Copenhagen.
Supervisor: Søren Hove Ravn.
```