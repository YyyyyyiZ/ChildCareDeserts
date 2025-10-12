# Child Care Deserts — Project I (Course: IEOR E4004)

This repository addresses **child-care “deserts”** in New York State (NYS). You will build optimization models to **eliminate deserts** by combining (i) **capacity expansions** at existing facilities and (ii) **new facility siting & sizing**, under policy and spatial constraints.

---

## Repository Structure

```
.
├─ ChildCareDeserts_Data/          # Place all input CSVs here
│  ├─ child_care_regulated.csv
│  ├─ population.csv
│  ├─ employment_rate.csv
│  ├─ avg_individual_income.csv
│  └─ potential_locations.csv
│
├─ task2.py                        # Gurobi-only solver for Task 2
└─ Task2_Output/                   # Auto-generated after running a solver
   ├─ new_builds.csv
   ├─ expansions.csv
   └─ zip_summary.csv
```

---

## Data Description (inputs)

Put the following CSVs into `ChildCareDeserts_Data/`:

* **child_care_regulated.csv** — Existing regulated facilities (lat/lon, capacities by age, total capacity, status).
* **population.csv** — ZIP-level population in age bins (e.g., `-5`, `5-9`, `10-14`).
* **employment_rate.csv** — ZIP-level employment rate (0–1).
* **avg_individual_income.csv** — ZIP-level average individual income (USD).
* **potential_locations.csv** — Candidate sites for new facilities (lat/lon) with ZIP.

> **Column names**: The solvers include a flexible alias map, so mild header differences are OK (e.g., `zipcode` vs `zip`).

---

## Task 1 — *[Placeholder]*

* **Goal**: *TODO (external teammate)*
* **Assumptions / Constraints**: *TODO*
* **Solver / Outputs**: *TODO*

---


## Task 2 — Realistic Capacity & Location (Detailed)

### Objective

**Minimize total cost** while eliminating deserts under **realistic** constraints:

* **Expansion cap** per existing facility: at most **+20%** of current total capacity.
* **Piecewise expansion costs** (convex, by tiers of current capacity):

  * 0–10%, 10–15%, 15–20% (increasing per-seat cost).
* **Minimum spacing**: any two facilities (existing or new) within the **same ZIP** must be **≥ 0.06 miles** apart.

We also enforce:

* **Desert elimination** per ZIP:

  * If ZIP is **high demand** (employment rate ≥ 0.60 **or** avg income ≤ $60k): total seats ≥ **½** of 0–12 population.
  * Otherwise: total seats ≥ **⅓** of 0–12 population.
* **0–5 rule** per ZIP: 0–5 seats ≥ **⅔** of 0–5 population.
* **New facility sizes** (choose exactly one size per selected candidate location):

  * Small **S** (100 seats; up to 50 for 0–5) — build cost $65k
  * Medium **M** (200; up to 100 for 0–5) — $95k
  * Large **L** (400; up to 200 for 0–5) — $115k
* **0–5 equipment cost:** $100 per newly created 0–5 seat (both expansions and new builds).

> **Age groups in model:** 0–5 (“Early”), 5–12 (“School”).
> **Population bin mapping (approx.):**
> 0–5 ≈ `-5` + 20% of `5-9`; 5–12 ≈ `5-9` + 60% of `10-14`; 0–12 = 0–5 + 5–12.

### Running the Solver (Gurobi)

**Recommended** (no PuLP installation needed):

```bash
pip install gurobipy pandas
python solve_task2_gurobi.py
```

Outputs will appear in `Task2_Output/`:

* `new_builds.csv` — chosen candidate sites & sizes, age split, build & equipment costs
* `expansions.csv` — expansions by tier (0–10%, 10–15%, 15–20%), age split, costs
* `zip_summary.csv` — per-ZIP population, thresholds, supplies, and feasibility flags

> If you prefer CBC/PuLP, use `solve_task2.py` (requires `pip install pulp pandas`).

### Model Sketch (What’s inside the scripts)

* **Decision variables**

  * New siting & sizing: binary ( y_{\ell s} \in {0,1} ) (at most one size per candidate location).
  * New build age split: ( u^{E}*{\ell s} ) (0–5 seats), ( u^{K}*{\ell s} ) (5–12 seats).
  * Existing facility expansions (three tiers): ( x^{(1)}_f, x^{(2)}_f, x^{(3)}_f \ge 0 ).
  * Expansion age split: ( r^{E}_f, r^{K}_f \ge 0 ) with ( r^{E}_f + r^{K}_f = \sum_k x^{(k)}_f ).

* **Objective (minimize total cost)**

  * Build cost (by size)
  * * Equipment cost for new 0–5 seats
  * * Tiered expansion costs (per-seat cost increases across tiers)

* **Key constraints**

  * **One size per location**; **age balance** in new builds; **0–5 cap** within a new facility.
  * **Spacing**: candidate–candidate mutual exclusion within 0.06 miles (same ZIP); candidate–existing “forbid” if too close.
  * **Expansion caps**: 0–10%, 10–15%, 15–20% of current total capacity.
  * **Desert elimination**: total (0–12) supply ≥ α × (0–12) population (α = ½ or ⅓ by ZIP).
  * **0–5 rule**: (0–5) supply ≥ ⅔ × (0–5) population.

> A full LaTeX formulation is provided in `ProjectI_Task2_Model_Report.tex` (with a **Notation longtable**).

### Result Interpretation (what to include in your report)

* **Per ZIP**: before/after supply vs thresholds; whether the desert & 0–5 rules are satisfied.
* **Chosen investments**: where to build (and size), how expansions are allocated across tiers and age groups.
* **Cost breakdown**: build vs equipment vs expansions.
* **Sensitivity**: spacing radius (e.g., 0.05–0.08 miles), high-demand classification, size costs.

---

## How to Reproduce (Quick Start)

1. Ensure the five CSVs are in `ChildCareDeserts_Data/`.
2. Run Gurobi version:

   ```bash
   python task2.py
   ```
3. Inspect `Task2_Output/` CSVs and incorporate into the report.

---

## Validation Checklist

* [ ] Each ZIP satisfies **both** rules (desert elimination & 0–5 ≥ ⅔ pop).
* [ ] No spacing violations (check mutual exclusion / forbids).
* [ ] Expansions per facility **≤ +20%** total, tier caps respected.
* [ ] New builds’ 0–5 seats **≤ size-specific max** (50/100/200).
* [ ] Costs correctly add up (build + equipment + tiered expansions).
* [ ] Edge cases handled (ZIPs with missing employment/income default to α = ⅓).

---

## Troubleshooting

* **Non-optimal / infeasible**: check raw demand (pop) vs existing supply; spacing may block feasible siting in some ZIPs. Try validating candidate locations or revisiting the 0.06-mile pairs.
* **Column name mismatches**: see alias map inside the scripts (`ALIASES`).
* **License / solver**: Gurobi requires a valid license; use the PuLP version if needed.

---

## Task 3 — *[Placeholder]*

* **Goal**: Under a $100M budget, **maximize** coverage index with **fairness** constraint (pairwise coverage disparity ≤ 0.1).
* **Decision variables**: *TODO (external teammate)*
* **Solver / Outputs**: *TODO*

---

