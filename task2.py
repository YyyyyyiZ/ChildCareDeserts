
import os
import math
import sys
import warnings
from typing import Dict, List, Tuple, Set
import pandas as pd

try:
    import pulp
except ImportError as e:
    raise SystemExit("This script requires PuLP. Install with `pip install pulp`.")


# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "ChildCareDeserts_Data")

# New facility size specs (can be tweaked if your PDF specifies different numbers)
SIZES = {
    "S": {"K": 100, "K_E": 50,  "build_cost": 65000},
    "M": {"K": 200, "K_E": 100, "build_cost": 95000},
    "L": {"K": 400, "K_E": 200, "build_cost": 115000},
}
EQUIP_COST_PER_0_5 = 100.0  # $100 per new 0-5 seat

# Spacing threshold (miles)
MIN_SPACING_MILES = 0.06

# Expansion tier percents of current total capacity (0-10%, 10-15%, 15-20%)
TIER_BOUNDS = (0.10, 0.05, 0.05)

# Column name aliases to make the script robust to slight header differences
ALIASES = {
    "zip": ["zip", "ZIP", "zipcode", "Zip", "postal", "postal_code", "Postal Code", "ZCTA5"],
    "lat": ["lat", "latitude", "Latitude", "LAT"],
    "lon": ["lon", "longitude", "Longitude", "LON", "lng"],
    "status": ["status", "Status"],
    "infant": ["infant_capacity", "Infant", "infant", "capacity_infant"],
    "toddler": ["toddler_capacity", "Toddler", "toddler", "capacity_toddler"],
    "preschool": ["preschool_capacity", "Preschool", "preschool", "capacity_preschool"],
    "school_age": ["school_age_capacity", "SchoolAge", "school_age", "capacity_school_age"],
    "total_capacity": ["total_capacity", "TotalCapacity", "total", "capacity_total", "Total Capacity"],
    "employment_rate": ["employment_rate", "EmploymentRate", "employment", "emp_rate"],
    "avg_income": ["avg_individual_income", "income", "avg_income", "average_income", "Average Individual Income"],
    # population age bins
    "under5": ["-5", "0-4", "under5", "Under 5", "0_to_4"],
    "5to9": ["5-9", "5_to_9", "5to9"],
    "10to14": ["10-14", "10_to_14", "10to14"]
}

LICENSE_STATUSES = {"License", "Registration", "LICENSE", "REGISTRATION", "Licensed", "Registered"}


# -----------------------------
# Utilities
# -----------------------------
def find_col(df: pd.DataFrame, keys: List[str], required: bool = True, default=None):
    for k in keys:
        if k in df.columns:
            return k
    if required:
        raise KeyError(f"Could not find any of the expected columns {keys} in {list(df.columns)[:10]} ...")
    return default


def coerce_zip(series: pd.Series) -> pd.Series:
    # Normalize zip to 5-character string where possible
    def norm(z):
        try:
            s = str(int(float(z)))
            return s.zfill(5)
        except Exception:
            s = str(z).strip()
            # if already 5+ chars, take first 5 (best-effort)
            if len(s) >= 5 and s[:5].isdigit():
                return s[:5]
            return s
    return series.map(norm)


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    km = R_km * c
    return km * 0.621371


def tag_high_demand(emp_rate: float, avg_income: float) -> bool:
    if pd.isna(emp_rate) or pd.isna(avg_income):
        return False
    return (emp_rate >= 0.60) or (avg_income <= 60000.0)


# -----------------------------
# Data loading
# -----------------------------
def load_population(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    zip_col = find_col(df, ALIASES["zip"])
    df["zip"] = coerce_zip(df[zip_col])
    u5_col = find_col(df, ALIASES["under5"])
    a5_9_col = find_col(df, ALIASES["5to9"])
    a10_14_col = find_col(df, ALIASES["10to14"])
    # Clean numeric
    for c in [u5_col, a5_9_col, a10_14_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["pop_0_5"] = df[u5_col] + 0.2 * df[a5_9_col]        # include the 5-year-olds
    df["pop_5_12"] = df[a5_9_col] + 0.6 * df[a10_14_col]   # include ages 10-12
    df["pop_0_12"] = df["pop_0_5"] + df["pop_5_12"]
    return df[["zip", "pop_0_5", "pop_5_12", "pop_0_12"]]


def load_emp_income(emp_path: str, inc_path: str) -> pd.DataFrame:
    emp = pd.read_csv(emp_path)
    inc = pd.read_csv(inc_path)
    emp_zip = find_col(emp, ALIASES["zip"])
    emp_rate_col = find_col(emp, ALIASES["employment_rate"])
    inc_zip = find_col(inc, ALIASES["zip"])
    inc_col = find_col(inc, ALIASES["avg_income"])
    emp = emp.rename(columns={emp_zip: "zip", emp_rate_col: "employment_rate"})
    inc = inc.rename(columns={inc_zip: "zip", inc_col: "avg_income"})
    emp["zip"] = coerce_zip(emp["zip"])
    inc["zip"] = coerce_zip(inc["zip"])
    df = emp.merge(inc, on="zip", how="outer")
    df["high_demand"] = df.apply(lambda r: tag_high_demand(r.get("employment_rate"), r.get("avg_income")), axis=1)
    df["alpha"] = df["high_demand"].map(lambda b: 0.5 if b else 1.0/3.0)
    return df[["zip", "employment_rate", "avg_income", "high_demand", "alpha"]]


def load_facilities(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    zip_col = find_col(df, ALIASES["zip"])
    lat_col = find_col(df, ALIASES["lat"])
    lon_col = find_col(df, ALIASES["lon"])
    status_col = find_col(df, ALIASES["status"], required=False)
    infant_col = find_col(df, ALIASES["infant"])
    toddler_col = find_col(df, ALIASES["toddler"])
    preschool_col = find_col(df, ALIASES["preschool"])
    school_age_col = find_col(df, ALIASES["school_age"])
    total_col = find_col(df, ALIASES["total_capacity"])

    # Filter by status if available
    if status_col and status_col in df.columns:
        df = df[df[status_col].astype(str).isin(LICENSE_STATUSES)].copy()

    # Clean
    for c in [infant_col, toddler_col, preschool_col, school_age_col, total_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df = df.rename(columns={
        zip_col: "zip", lat_col: "lat", lon_col: "lon",
        infant_col: "cap_infant", toddler_col: "cap_toddler",
        preschool_col: "cap_preschool", school_age_col: "cap_school",
        total_col: "cap_total"
    })
    df["zip"] = coerce_zip(df["zip"])
    df["cap_0_5"] = df["cap_infant"] + df["cap_toddler"] + df["cap_preschool"]
    return df[["zip","lat","lon","cap_0_5","cap_school","cap_total"]].reset_index(drop=True).rename(columns={"index":"fid"})


def load_candidates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    zip_col = find_col(df, ALIASES["zip"])
    lat_col = find_col(df, ALIASES["lat"])
    lon_col = find_col(df, ALIASES["lon"])
    df = df.rename(columns={zip_col:"zip", lat_col:"lat", lon_col:"lon"})
    df["zip"] = coerce_zip(df["zip"])
    df = df.reset_index(drop=True)
    df["lid"] = df.index.astype(int)
    return df[["lid","zip","lat","lon"]]


# -----------------------------
# Conflict set construction
# -----------------------------
def build_conflicts(cands: pd.DataFrame, facs: pd.DataFrame) -> Tuple[Set[Tuple[int,int]], Set[Tuple[int,int]]]:
    """
    Returns:
      P: set of unordered candidate-candidate pairs {lid1,lid2} (as tuple sorted) that conflict (< MIN_SPACING) within same ZIP
      Q: set of (lid, fid) candidate-existing conflicts (< MIN_SPACING) within same ZIP
    """
    P = set()
    Q = set()
    # group by zip to limit comparisons
    cand_groups = {z: g[["lid","lat","lon"]].to_numpy() for z,g in cands.groupby("zip")}
    fac_groups = {z: g[["lat","lon"]].to_numpy() for z,g in facs.groupby("zip")}

    for z, arr in cand_groups.items():
        # candidate-candidate
        n = arr.shape[0]
        for i in range(n):
            lid_i, lat_i, lon_i = int(arr[i,0]), float(arr[i,1]), float(arr[i,2])
            # compare only j>i
            for j in range(i+1, n):
                lid_j, lat_j, lon_j = int(arr[j,0]), float(arr[j,1]), float(arr[j,2])
                d = haversine_miles(lat_i, lon_i, lat_j, lon_j)
                if d < MIN_SPACING_MILES:
                    P.add((min(lid_i,lid_j), max(lid_i,lid_j)))

        # candidate-existing
        fac_arr = fac_groups.get(z, None)
        if fac_arr is not None:
            for i in range(arr.shape[0]):
                lid_i, lat_i, lon_i = int(arr[i,0]), float(arr[i,1]), float(arr[i,2])
                for (flat, flon) in fac_arr:
                    d = haversine_miles(lat_i, lon_i, float(flat), float(flon))
                    if d < MIN_SPACING_MILES:
                        Q.add((lid_i, -1))  # we only need to forbid the candidate; fid not required by the MILP here

    return P, Q


# -----------------------------
# MILP build & solve (PuLP)
# -----------------------------
def build_and_solve(pop: pd.DataFrame, empinc: pd.DataFrame, facs: pd.DataFrame, cands: pd.DataFrame):
    # Merge ZIP-level frames
    zdf = pop.merge(empinc[["zip","alpha"]], on="zip", how="left")
    zdf["alpha"] = zdf["alpha"].fillna(1.0/3.0)
    Z = zdf["zip"].tolist()

    # Index maps
    F = list(range(len(facs)))   # existing facility indices
    LIDS = cands["lid"].tolist() # candidate ids
    S = list(SIZES.keys())

    # Split facilities by zip
    facs = facs.copy()
    facs["fid"] = facs.index
    # Precompute per-facility per-seat expansion costs (tiers)
    gammas = {}
    for f, row in facs.iterrows():
        Ntot = float(row["cap_total"]) if row["cap_total"] > 0 else 1.0
        g1 = (20000.0 + 200.0 * Ntot) / Ntot
        g2 = (20000.0 + 400.0 * Ntot) / Ntot
        g3 = (20000.0 + 1000.0 * Ntot) / Ntot
        gammas[f] = (g1, g2, g3)

    # Build conflicts
    P_pairs, Q_forbid = build_conflicts(cands, facs)

    # Grouping helpers
    Fz = {z: facs.loc[facs["zip"]==z, "fid"].tolist() for z in Z}
    Lz = {z: cands.loc[cands["zip"]==z, "lid"].tolist() for z in Z}

    # ----------------- Model -----------------
    m = pulp.LpProblem("Task2_RealisticCapacityLocation", pulp.LpMinimize)

    # Variables
    y = pulp.LpVariable.dicts("y", ((l,s) for l in LIDS for s in S), lowBound=0, upBound=1, cat="Binary")
    uE = pulp.LpVariable.dicts("uE", ((l,s) for l in LIDS for s in S), lowBound=0, cat="Continuous")
    uK = pulp.LpVariable.dicts("uK", ((l,s) for l in LIDS for s in S), lowBound=0, cat="Continuous")

    x1 = pulp.LpVariable.dicts("x1", F, lowBound=0, cat="Continuous")
    x2 = pulp.LpVariable.dicts("x2", F, lowBound=0, cat="Continuous")
    x3 = pulp.LpVariable.dicts("x3", F, lowBound=0, cat="Continuous")
    rE = pulp.LpVariable.dicts("rE", F, lowBound=0, cat="Continuous")
    rK = pulp.LpVariable.dicts("rK", F, lowBound=0, cat="Continuous")

    # Objective
    build_cost = pulp.lpSum(SIZES[s]["build_cost"] * y[(l,s)] for l in LIDS for s in S)
    equip_cost = pulp.lpSum(EQUIP_COST_PER_0_5 * (uE[(l,s)]) for l in LIDS for s in S) + \
                 pulp.lpSum(EQUIP_COST_PER_0_5 * rE[f] for f in F)

    exp_cost = pulp.lpSum(gammas[f][0]*x1[f] + gammas[f][1]*x2[f] + gammas[f][2]*x3[f] for f in F)

    m += build_cost + equip_cost + exp_cost, "TotalCost"

    # Constraints

    # (C1) one size per location
    for l in LIDS:
        m += pulp.lpSum(y[(l,s)] for s in S) <= 1, f"C1_onesize_{l}"

    # (C2) new build age-split conservation
    for l in LIDS:
        for s in S:
            K = SIZES[s]["K"]
            m += uE[(l,s)] + uK[(l,s)] == K * y[(l,s)], f"C2_balance_{l}_{s}"

    # (C3) 0-5 cap within new facility
    for l in LIDS:
        for s in S:
            KE = SIZES[s]["K_E"]
            m += uE[(l,s)] <= KE * y[(l,s)], f"C3_capE_{l}_{s}"

    # (C4) candidate-candidate spacing conflicts (same ZIP)
    for (l1, l2) in P_pairs:
        m += pulp.lpSum(y[(l1,s)] for s in S) + pulp.lpSum(y[(l2,s)] for s in S) <= 1, f"C4_spacing_{l1}_{l2}"

    # (C5) candidate-existing spacing conflicts (forbid)
    for (l, _fid) in Q_forbid:
        m += pulp.lpSum(y[(l,s)] for s in S) == 0, f"C5_forbid_{l}"

    # (C6) expansion tier caps
    for f, row in facs.iterrows():
        Ntot = float(row["cap_total"])
        m += x1[f] <= TIER_BOUNDS[0] * Ntot, f"C6_t1cap_{f}"
        m += x2[f] <= TIER_BOUNDS[1] * Ntot, f"C6_t2cap_{f}"
        m += x3[f] <= TIER_BOUNDS[2] * Ntot, f"C6_t3cap_{f}"

    # (C7) expansion age split
    for f in F:
        m += rE[f] + rK[f] == x1[f] + x2[f] + x3[f], f"C7_split_{f}"

    # (C8) desert elimination per ZIP
    for _, zr in zdf.iterrows():
        z = zr["zip"]
        alpha = float(zr["alpha"])
        pop012 = float(zr["pop_0_12"])
        lhs_existing = pulp.lpSum((facs.loc[f,"cap_total"] + x1[f] + x2[f] + x3[f]) for f in Fz.get(z, []))
        lhs_new = pulp.lpSum((uE[(l,s)] + uK[(l,s)]) for l in Lz.get(z, []) for s in S)
        m += lhs_existing + lhs_new >= alpha * pop012, f"C8_desert_{z}"

    # (C9) 0-5 coverage per ZIP
    for _, zr in zdf.iterrows():
        z = zr["zip"]
        pop05 = float(zr["pop_0_5"])
        lhs_existing = pulp.lpSum((facs.loc[f,"cap_0_5"] + rE[f]) for f in Fz.get(z, []))
        lhs_new = pulp.lpSum(uE[(l,s)] for l in Lz.get(z, []) for s in S)
        m += lhs_existing + lhs_new >= (2.0/3.0) * pop05, f"C9_0_5_{z}"

    # Solve
    solver = pulp.GUROBI_CMD(msg=True)     res = m.solve(solver)

    print("Status:", pulp.LpStatus[m.status])
    print("Objective (Total Cost):", pulp.value(m.objective))

    # ----------------- Extract solution -----------------
    # New builds
    picks = []
    for l in LIDS:
        for s in S:
            if pulp.value(y[(l,s)]) > 0.5:
                picks.append({
                    "lid": l,
                    "zip": str(cands.loc[cands["lid"]==l, "zip"].iloc[0]),
                    "size": s,
                    "uE": pulp.value(uE[(l,s)]),
                    "uK": pulp.value(uK[(l,s)]),
                    "build_cost": SIZES[s]["build_cost"],
                    "equip_cost_0_5": EQUIP_COST_PER_0_5 * pulp.value(uE[(l,s)])
                })
    builds_df = pd.DataFrame(picks)

    # Expansions
    exps = []
    for f in F:
        valx1, valx2, valx3 = pulp.value(x1[f]), pulp.value(x2[f]), pulp.value(x3[f])
        if (valx1 or valx2 or valx3):
            g1, g2, g3 = gammas[f]
            exps.append({
                "fid": f,
                "zip": str(facs.loc[f, "zip"]),
                "x1_0_10": valx1,
                "x2_10_15": valx2,
                "x3_15_20": valx3,
                "rE": pulp.value(rE[f]),
                "rK": pulp.value(rK[f]),
                "tier_cost": g1*valx1 + g2*valx2 + g3*valx3,
                "equip_cost_0_5": EQUIP_COST_PER_0_5 * pulp.value(rE[f])
            })
    exps_df = pd.DataFrame(exps)

    # ZIP summaries
    rows = []
    for _, zr in zdf.iterrows():
        z = zr["zip"]
        pop05 = float(zr["pop_0_5"])
        pop012 = float(zr["pop_0_12"])
        alpha = float(zr["alpha"])

        supply012 = sum(facs.loc[f,"cap_total"] + pulp.value(x1[f] + x2[f] + x3[f]) for f in Fz.get(z, []))
        supply012 += sum(pulp.value(uE[(l,s)] + uK[(l,s)]) for l in Lz.get(z, []) for s in S)

        supply05 = sum(facs.loc[f,"cap_0_5"] + pulp.value(rE[f]) for f in Fz.get(z, []))
        supply05 += sum(pulp.value(uE[(l,s)]) for l in Lz.get(z, []) for s in S)

        rows.append({
            "zip": z,
            "alpha": alpha,
            "pop_0_5": pop05,
            "pop_0_12": pop012,
            "supply_0_5": supply05,
            "supply_0_12": supply012,
            "desert_req": alpha * pop012,
            "zero_to_five_req": (2.0/3.0) * pop05,
            "desert_met": supply012 >= alpha * pop012 - 1e-6,
            "0_5_met": supply05 >= (2.0/3.0) * pop05 - 1e-6
        })
    zip_df = pd.DataFrame(rows)

    # Write outputs
    out_dir = os.path.join(os.path.dirname(__file__), "Task2_Output")
    os.makedirs(out_dir, exist_ok=True)
    builds_path = os.path.join(out_dir, "new_builds.csv")
    exps_path = os.path.join(out_dir, "expansions.csv")
    zip_path = os.path.join(out_dir, "zip_summary.csv")

    builds_df.to_csv(builds_path, index=False)
    exps_df.to_csv(exps_path, index=False)
    zip_df.to_csv(zip_path, index=False)

    print(f"Saved new builds to: {builds_path}")
    print(f"Saved expansions to: {exps_path}")
    print(f"Saved ZIP summary to: {zip_path}")

    return {
        "status": pulp.LpStatus[m.status],
        "objective": float(pulp.value(m.objective)) if m.status == 1 else None,
        "new_builds_csv": builds_path,
        "expansions_csv": exps_path,
        "zip_summary_csv": zip_path,
    }


def main():
    # Load data
    pop = load_population(os.path.join(DATA_DIR, "population.csv"))
    empinc = load_emp_income(os.path.join(DATA_DIR, "employment_rate.csv"),
                             os.path.join(DATA_DIR, "avg_individual_income.csv"))
    facs = load_facilities(os.path.join(DATA_DIR, "child_care_regulated.csv"))
    cands = load_candidates(os.path.join(DATA_DIR, "potential_locations.csv"))

    # Solve
    result = build_and_solve(pop, empinc, facs, cands)
    if result["status"] != "Optimal":
        warnings.warn(f"Solve status: {result['status']}")

if __name__ == "__main__":
    main()
