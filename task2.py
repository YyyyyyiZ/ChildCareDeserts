
import os
import math
import warnings
from typing import List, Tuple, Set
import pandas as pd
from gurobipy import Model, GRB, quicksum

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "ChildCareDeserts_Data")

# New facility size specs
SIZES = {
    "S": {"K": 100, "K_E": 50,  "build_cost": 65000},
    "M": {"K": 200, "K_E": 100, "build_cost": 95000},
    "L": {"K": 400, "K_E": 200, "build_cost": 115000},
}
EQUIP_COST_PER_0_5 = 100.0  # $ per new 0-5 seat
MIN_SPACING_MILES = 0.06
TIER_BOUNDS = (0.10, 0.05, 0.05)

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
    "under5": ["-5", "0-4", "under5", "Under 5", "0_to_4"],
    "5to9": ["5-9", "5_to_9", "5to9"],
    "10to14": ["10-14", "10_to_14", "10to14"]
}
LICENSE_STATUSES = {"License", "Registration", "LICENSE", "REGISTRATION", "Licensed", "Registered"}

# -----------------------------
# Utils
# -----------------------------
def find_col(df: pd.DataFrame, keys: List[str], required: bool = True, default=None):
    for k in keys:
        if k in df.columns:
            return k
    if required:
        raise KeyError(f"Could not find any of the expected columns {keys} in {list(df.columns)[:10]} ...")
    return default

def coerce_zip(series: pd.Series) -> pd.Series:
    def norm(z):
        try:
            s = str(int(float(z)))
            return s.zfill(5)
        except Exception:
            s = str(z).strip()
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
    return (R_km * c) * 0.621371

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
    for c in [u5_col, a5_9_col, a10_14_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["pop_0_5"] = df[u5_col] + 0.2 * df[a5_9_col]
    df["pop_5_12"] = df[a5_9_col] + 0.6 * df[a10_14_col]
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
    if status_col and status_col in df.columns:
        df = df[df[status_col].astype(str).isin(LICENSE_STATUSES)].copy()
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
    df = df.reset_index(drop=True)
    df["fid"] = df.index.astype(int)
    return df[["fid","zip","lat","lon","cap_0_5","cap_school","cap_total"]]

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
# Conflicts
# -----------------------------
def build_conflicts(cands: pd.DataFrame, facs: pd.DataFrame) -> Tuple[Set[Tuple[int,int]], Set[int]]:
    P = set()     # candidate-candidate pairs (l1,l2) with l1<l2
    forbid = set()  # candidate ids forbidden due to proximity to existing
    cand_groups = {z: g[["lid","lat","lon"]].to_numpy() for z,g in cands.groupby("zip")}
    fac_groups = {z: g[["lat","lon"]].to_numpy() for z,g in facs.groupby("zip")}
    for z, carr in cand_groups.items():
        n = carr.shape[0]
        for i in range(n):
            l1, lat1, lon1 = int(carr[i,0]), float(carr[i,1]), float(carr[i,2])
            for j in range(i+1, n):
                l2, lat2, lon2 = int(carr[j,0]), float(carr[j,1]), float(carr[j,2])
                d = haversine_miles(lat1, lon1, lat2, lon2)
                if d < MIN_SPACING_MILES:
                    P.add((min(l1,l2), max(l1,l2)))
        farr = fac_groups.get(z, None)
        if farr is not None:
            for i in range(n):
                l1, lat1, lon1 = int(carr[i,0]), float(carr[i,1]), float(carr[i,2])
                for (flat, flon) in farr:
                    d = haversine_miles(lat1, lon1, float(flat), float(flon))
                    if d < MIN_SPACING_MILES:
                        forbid.add(l1)
                        break
    return P, forbid

# -----------------------------
# MILP with Gurobi
# -----------------------------
def solve_task2():
    # Load data
    pop = load_population(os.path.join(DATA_DIR, "population.csv"))
    empinc = load_emp_income(os.path.join(DATA_DIR, "employment_rate.csv"),
                             os.path.join(DATA_DIR, "avg_individual_income.csv"))
    facs = load_facilities(os.path.join(DATA_DIR, "child_care_regulated.csv"))
    cands = load_candidates(os.path.join(DATA_DIR, "potential_locations.csv"))

    zdf = pop.merge(empinc[["zip","alpha"]], on="zip", how="left")
    zdf["alpha"] = zdf["alpha"].fillna(1.0/3.0)
    Z = zdf["zip"].tolist()

    F = facs["fid"].tolist()
    LIDS = cands["lid"].tolist()
    S = list(SIZES.keys())

    # Precompute gammas per facility
    gammas = {}
    for f, row in facs.set_index("fid").iterrows():
        Ntot = float(row["cap_total"]) if row["cap_total"] > 0 else 1.0
        gammas[f] = (
            (20000.0 + 200.0 * Ntot) / Ntot,
            (20000.0 + 400.0 * Ntot) / Ntot,
            (20000.0 + 1000.0 * Ntot) / Ntot,
        )

    # Conflicts
    P_pairs, forbid = build_conflicts(cands, facs)

    Fz = {z: facs.loc[facs["zip"]==z, "fid"].tolist() for z in Z}
    Lz = {z: cands.loc[cands["zip"]==z, "lid"].tolist() for z in Z}

    # Model
    m = Model("Task2_Gurobi")
    m.Params.OutputFlag = 1

    # Vars
    y = {(l,s): m.addVar(vtype=GRB.BINARY, name=f"y[{l},{s}]") for l in LIDS for s in S}
    uE = {(l,s): m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"uE[{l},{s}]") for l in LIDS for s in S}
    uK = {(l,s): m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"uK[{l},{s}]") for l in LIDS for s in S}

    x1 = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x1[{f}]") for f in F}
    x2 = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x2[{f}]") for f in F}
    x3 = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x3[{f}]") for f in F}
    rE = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"rE[{f}]") for f in F}
    rK = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"rK[{f}]") for f in F}

    # Objective
    build_cost = quicksum(SIZES[s]["build_cost"] * y[(l,s)] for l in LIDS for s in S)
    equip_cost = quicksum(EQUIP_COST_PER_0_5 * uE[(l,s)] for l in LIDS for s in S) + \
                 quicksum(EQUIP_COST_PER_0_5 * rE[f] for f in F)
    exp_cost = quicksum(gammas[f][0]*x1[f] + gammas[f][1]*x2[f] + gammas[f][2]*x3[f] for f in F)
    m.setObjective(build_cost + equip_cost + exp_cost, GRB.MINIMIZE)

    # Constraints
    # (C1) one size per location
    for l in LIDS:
        m.addConstr(quicksum(y[(l,s)] for s in S) <= 1, name=f"C1_onesize[{l}]")

    # (C2) age split in new builds
    for l in LIDS:
        for s in S:
            K = SIZES[s]["K"]
            m.addConstr(uE[(l,s)] + uK[(l,s)] == K * y[(l,s)], name=f"C2_balance[{l},{s}]")

    # (C3) 0-5 cap in new builds
    for l in LIDS:
        for s in S:
            KE = SIZES[s]["K_E"]
            m.addConstr(uE[(l,s)] <= KE * y[(l,s)], name=f"C3_capE[{l},{s}]")

    # (C4) candidate-candidate spacing
    for (l1,l2) in P_pairs:
        m.addConstr(quicksum(y[(l1,s)] for s in S) + quicksum(y[(l2,s)] for s in S) <= 1, name=f"C4_spacing[{l1},{l2}]")

    # (C5) forbid candidates too close to existing
    for l in forbid:
        m.addConstr(quicksum(y[(l,s)] for s in S) == 0, name=f"C5_forbid[{l}]")

    # (C6) expansion caps
    for f, row in facs.set_index("fid").iterrows():
        Ntot = float(row["cap_total"])
        m.addConstr(x1[f] <= TIER_BOUNDS[0] * Ntot, name=f"C6_t1cap[{f}]")
        m.addConstr(x2[f] <= TIER_BOUNDS[1] * Ntot, name=f"C6_t2cap[{f}]")
        m.addConstr(x3[f] <= TIER_BOUNDS[2] * Ntot, name=f"C6_t3cap[{f}]")

    # (C7) expansion split
    for f in F:
        m.addConstr(rE[f] + rK[f] == x1[f] + x2[f] + x3[f], name=f"C7_split[{f}]")

    # (C8) desert elimination per ZIP
    for _, zr in zdf.iterrows():
        z = zr["zip"]
        alpha = float(zr["alpha"])
        pop012 = float(zr["pop_0_12"])
        lhs_existing = quicksum((facs.set_index('fid').loc[f,"cap_total"] + x1[f] + x2[f] + x3[f]) for f in F if facs.loc[f,"zip"]==z)
        lhs_new = quicksum(uE[(l,s)] + uK[(l,s)] for l in Lz.get(z, []) for s in S)
        m.addConstr(lhs_existing + lhs_new >= alpha * pop012, name=f"C8_desert[{z}]")

    # (C9) 0-5 coverage per ZIP
    for _, zr in zdf.iterrows():
        z = zr["zip"]
        pop05 = float(zr["pop_0_5"])
        lhs_existing = quicksum((facs.set_index('fid').loc[f,"cap_0_5"] + rE[f]) for f in F if facs.loc[f,"zip"]==z)
        lhs_new = quicksum(uE[(l,s)] for l in Lz.get(z, []) for s in S)
        m.addConstr(lhs_existing + lhs_new >= (2.0/3.0) * pop05, name=f"C9_0_5[{z}]")

    m.optimize()

    status = m.Status
    if status != GRB.OPTIMAL:
        warnings.warn(f"Solve status: {status}")

    # Extract solution
    # New builds
    picks = []
    for l in LIDS:
        for s in S:
            if y[(l,s)].X > 0.5:
                row = {
                    "lid": l,
                    "zip": str(cands.loc[cands["lid"]==l, "zip"].iloc[0]),
                    "size": s,
                    "uE": uE[(l,s)].X,
                    "uK": uK[(l,s)].X,
                    "build_cost": SIZES[s]["build_cost"],
                    "equip_cost_0_5": EQUIP_COST_PER_0_5 * uE[(l,s)].X
                }
                picks.append(row)
    builds_df = pd.DataFrame(picks)

    # Expansions
    exps = []
    for f in F:
        valx1, valx2, valx3 = x1[f].X, x2[f].X, x3[f].X
        if (valx1 + valx2 + valx3) > 1e-6:
            g1, g2, g3 = gammas[f]
            exps.append({
                "fid": f,
                "zip": str(facs.loc[f, "zip"]),
                "x1_0_10": valx1,
                "x2_10_15": valx2,
                "x3_15_20": valx3,
                "rE": rE[f].X,
                "rK": rK[f].X,
                "tier_cost": g1*valx1 + g2*valx2 + g3*valx3,
                "equip_cost_0_5": EQUIP_COST_PER_0_5 * rE[f].X
            })
    exps_df = pd.DataFrame(exps)

    # ZIP summaries
    rows = []
    facs_idx = facs.set_index("fid")
    for _, zr in zdf.iterrows():
        z = zr["zip"]
        pop05 = float(zr["pop_0_5"])
        pop012 = float(zr["pop_0_12"])
        alpha = float(zr["alpha"])
        supply012 = 0.0
        for f in F:
            if facs.loc[f,"zip"] == z:
                supply012 += facs_idx.loc[f,"cap_total"] + x1[f].X + x2[f].X + x3[f].X
        for l in Lz.get(z, []):
            for s in S:
                supply012 += uE[(l,s)].X + uK[(l,s)].X
        supply05 = 0.0
        for f in F:
            if facs.loc[f,"zip"] == z:
                supply05 += facs_idx.loc[f,"cap_0_5"] + rE[f].X
        for l in Lz.get(z, []):
            for s in S:
                supply05 += uE[(l,s)].X
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
            "zero_to_five_met": supply05 >= (2.0/3.0) * pop05 - 1e-6
        })
    zip_df = pd.DataFrame(rows)

    # Outputs
    out_dir = os.path.join(os.path.dirname(__file__), "Task2_Output")
    os.makedirs(out_dir, exist_ok=True)
    builds_path = os.path.join(out_dir, "new_builds.csv")
    exps_path = os.path.join(out_dir, "expansions.csv")
    zip_path = os.path.join(out_dir, "zip_summary.csv")

    builds_df.to_csv(builds_path, index=False)
    exps_df.to_csv(exps_path, index=False)
    zip_df.to_csv(zip_path, index=False)

    print("Objective (Total Cost):", m.ObjVal if status == GRB.OPTIMAL else None)
    print(f"Saved new builds to: {builds_path}")
    print(f"Saved expansions to: {exps_path}")
    print(f"Saved ZIP summary to: {zip_path}")

if __name__ == "__main__":
    solve_task2()
