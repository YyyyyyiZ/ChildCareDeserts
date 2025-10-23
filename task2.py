# -*- coding: utf-8 -*-
# Task 2 (Realistic Capacity & Location) – Gurobi solver with robust infeasibility diagnostics (no IIS)
# Usage:
#   pip install gurobipy pandas
#   python task2.py
#
# Expects sibling folder "ChildCareDeserts_Data" with:
#   child_care_regulated.csv, population.csv, employment_rate.csv, avg_individual_income.csv, potential_locations.csv
#
# Outputs:
#   Task2_Output/
#     new_builds.csv, expansions.csv, zip_summary.csv                  (当主模型可行/最优)
#     zip_upper_bound_check.csv                                        (乐观上界可行性体检)
#     diagnosis_*.csv (diagnosis_zip_slacks / spacing_slacks / forbid_slacks / expansion_slacks)  (当主模型不可行)

import os
import math
import warnings
from typing import List, Tuple, Set, Dict
import pandas as pd
from gurobipy import Model, GRB, quicksum

# ---------------- Config ----------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "ChildCareDeserts_Data")

SIZES = {
    "S": {"K": 100, "K_E": 50,  "build_cost": 65000},
    "M": {"K": 200, "K_E": 100, "build_cost": 95000},
    "L": {"K": 400, "K_E": 200, "build_cost": 115000},
}
EQUIP_COST_PER_0_5 = 100.0
MIN_SPACING_MILES = 0.06
TIER_BOUNDS = (0.10, 0.05, 0.05)  # 0–10%, 10–15%, 15–20%

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

# ---------------- Utils ----------------
def find_col(df: pd.DataFrame, keys: List[str], required: bool = True, default=None):
    for k in keys:
        if k in df.columns:
            return k
    if required:
        raise KeyError(f"Could not find columns {keys} in {list(df.columns)[:12]} ...")
    return default

def coerce_zip(series: pd.Series) -> pd.Series:
    def norm(z):
        try:
            s = str(int(float(z))); return s.zfill(5)
        except Exception:
            s = str(z).strip()
            if len(s) >= 5 and s[:5].isdigit(): return s[:5]
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
    if pd.isna(emp_rate) or pd.isna(avg_income): return False
    return (emp_rate >= 0.60) or (avg_income <= 60000.0)

# ---------------- Loaders ----------------
def load_population(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    zip_col = find_col(df, ALIASES["zip"])
    df["zip"] = coerce_zip(df[zip_col])
    u5_col = find_col(df, ALIASES["under5"])
    a5_9_col = find_col(df, ALIASES["5to9"])
    a10_14_col = find_col(df, ALIASES["10to14"])
    for c in [u5_col, a5_9_col, a10_14_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # 0–5 / 5–12 / 0–12 近似映射（如课程给了更精确映射，可直接替换系数）
    df["pop_0_5"] = df[u5_col] + 0.2 * df[a5_9_col]      # + five-year-olds
    df["pop_5_12"] = df[a5_9_col] + 0.6 * df[a10_14_col] # + ages 10–12
    df["pop_0_12"] = df["pop_0_5"] + df["pop_5_12"]
    return df[["zip", "pop_0_5", "pop_5_12", "pop_0_12"]]

def load_emp_income(emp_path: str, inc_path: str) -> pd.DataFrame:
    emp = pd.read_csv(emp_path); inc = pd.read_csv(inc_path)
    emp_zip = find_col(emp, ALIASES["zip"]); emp_rate_col = find_col(emp, ALIASES["employment_rate"])
    inc_zip = find_col(inc, ALIASES["zip"]); inc_col = find_col(inc, ALIASES["avg_income"])
    emp = emp.rename(columns={emp_zip: "zip", emp_rate_col: "employment_rate"})
    inc = inc.rename(columns={inc_zip: "zip", inc_col: "avg_income"})
    emp["zip"] = coerce_zip(emp["zip"]); inc["zip"] = coerce_zip(inc["zip"])
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

# ---------------- Conflicts ----------------
def build_conflicts(cands: pd.DataFrame, facs: pd.DataFrame) -> Tuple[Set[Tuple[int,int]], Set[int]]:
    P: Set[Tuple[int,int]] = set()
    forbid: Set[int] = set()
    cand_groups = {z: g[["lid","lat","lon"]].to_numpy() for z,g in cands.groupby("zip")}
    fac_groups  = {z: g[["lat","lon"]].to_numpy() for z,g in facs.groupby("zip")}
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

# ---------------- Upper-bound quick check ----------------
def write_upper_bound_check(zdf: pd.DataFrame, facs: pd.DataFrame, cands: pd.DataFrame, Fz: Dict[str, list], Lz: Dict[str, list], out_dir: str):
    rows = []
    for _, zr in zdf.iterrows():
        z = zr["zip"]; pop05 = float(zr["pop_0_5"]); pop012 = float(zr["pop_0_12"]); alpha = float(zr["alpha"])
        fids = Fz.get(z, [])
        existing_tot = sum(float(facs.loc[fid, "cap_total"]) for fid in fids)
        existing_05  = sum(float(facs.loc[fid, "cap_0_5"]) for fid in fids)
        max_expand   = 0.20 * existing_tot
        count_cands  = len(Lz.get(z, []))
        max_new_tot  = 400.0 * count_cands
        max_new_05   = 200.0 * count_cands
        UB_012 = existing_tot + max_expand + max_new_tot
        UB_05  = existing_05  + max_new_05
        rows.append({
            "zip": z,
            "UB_supply_0_12_ignore_spacing": UB_012,
            "need_0_12": alpha * pop012,
            "UB_supply_0_5_ignore_spacing": UB_05,
            "need_0_5": (2.0/3.0) * pop05,
            "hopeless_0_12": UB_012 + 1e-6 < alpha * pop012,
            "hopeless_0_5": UB_05  + 1e-6 < (2.0/3.0) * pop05
        })
    ub_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    ub_df.to_csv(os.path.join(out_dir, "zip_upper_bound_check.csv"), index=False)
    print("[Diagnostic] Wrote upper-bound check: Task2_Output/zip_upper_bound_check.csv")

# ---------------- Main solve ----------------
def solve_task2():
    pop = load_population(os.path.join(DATA_DIR, "population.csv"))
    empinc = load_emp_income(os.path.join(DATA_DIR, "employment_rate.csv"),
                             os.path.join(DATA_DIR, "avg_individual_income.csv"))
    facs = load_facilities(os.path.join(DATA_DIR, "child_care_regulated.csv"))
    cands = load_candidates(os.path.join(DATA_DIR, "potential_locations.csv"))

    zdf = pop.merge(empinc[["zip","alpha"]], on="zip", how="left")
    zdf["alpha"] = zdf["alpha"].fillna(1.0/3.0)
    Z = zdf["zip"].tolist()

    # Index sets
    F = facs["fid"].tolist()
    LIDS = cands["lid"].tolist()
    S = list(SIZES.keys())

    # Group by ZIP
    Fz = {z: facs.loc[facs["zip"]==z, "fid"].tolist() for z in Z}
    Lz = {z: cands.loc[cands["zip"]==z, "lid"].tolist() for z in Z}

    # Per-facility tier costs
    facs_idx = facs.set_index("fid")
    gammas: Dict[int, Tuple[float,float,float]] = {}
    for f in F:
        Ntot = float(facs_idx.loc[f, "cap_total"]);  Ntot = max(Ntot, 1.0)
        gammas[f] = ((20000.0 + 200.0 * Ntot) / Ntot,
                     (20000.0 + 400.0 * Ntot) / Ntot,
                     (20000.0 + 1000.0 * Ntot) / Ntot)

    # Spacing conflicts
    P_pairs, forbid = build_conflicts(cands, facs)

    out_dir = os.path.join(os.path.dirname(__file__), "Task2_Output")
    os.makedirs(out_dir, exist_ok=True)
    write_upper_bound_check(zdf, facs, cands, Fz, Lz, out_dir)

    # ----- Primary model -----
    m = Model("Task2_Gurobi"); m.Params.OutputFlag = 1

    y  = {(l,s): m.addVar(vtype=GRB.BINARY, name=f"y[{l},{s}]") for l in LIDS for s in S}
    uE = {(l,s): m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"uE[{l},{s}]") for l in LIDS for s in S}
    uK = {(l,s): m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"uK[{l},{s}]") for l in LIDS for s in S}
    x1 = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x1[{f}]") for f in F}
    x2 = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x2[{f}]") for f in F}
    x3 = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x3[{f}]") for f in F}
    rE = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"rE[{f}]") for f in F}
    rK = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"rK[{f}]") for f in F}

    build_cost = quicksum(SIZES[s]["build_cost"] * y[(l,s)] for l in LIDS for s in S)
    equip_cost = quicksum(EQUIP_COST_PER_0_5 * uE[(l,s)] for l in LIDS for s in S) + quicksum(EQUIP_COST_PER_0_5 * rE[f] for f in F)
    exp_cost = quicksum(gammas[f][0]*x1[f] + gammas[f][1]*x2[f] + gammas[f][2]*x3[f] for f in F)
    m.setObjective(build_cost + equip_cost + exp_cost, GRB.MINIMIZE)

    for l in LIDS:
        m.addConstr(quicksum(y[(l,s)] for s in S) <= 1, name=f"C1_onesize[{l}]")
    for l in LIDS:
        for s in S:
            K, KE = SIZES[s]["K"], SIZES[s]["K_E"]
            m.addConstr(uE[(l,s)] + uK[(l,s)] == K * y[(l,s)], name=f"C2_balance[{l},{s}]")
            m.addConstr(uE[(l,s)] <= KE * y[(l,s)], name=f"C3_capE[{l},{s}]")
    for (l1, l2) in P_pairs:
        m.addConstr(quicksum(y[(l1,s)] for s in S) + quicksum(y[(l2,s)] for s in S) <= 1, name=f"C4_spacing[{l1},{l2}]")
    for l in forbid:
        m.addConstr(quicksum(y[(l,s)] for s in S) == 0, name=f"C5_forbid[{l}]")
    for f in F:
        Ntot = float(facs_idx.loc[f, "cap_total"])
        m.addConstr(x1[f] <= TIER_BOUNDS[0] * Ntot, name=f"C6_t1cap[{f}]")
        m.addConstr(x2[f] <= TIER_BOUNDS[1] * Ntot, name=f"C6_t2cap[{f}]")
        m.addConstr(x3[f] <= TIER_BOUNDS[2] * Ntot, name=f"C6_t3cap[{f}]")
    for f in F:
        m.addConstr(rE[f] + rK[f] == x1[f] + x2[f] + x3[f], name=f"C7_split[{f}]")
    for _, zr in zdf.iterrows():
        z = zr["zip"]; alpha = float(zr["alpha"]); pop012 = float(zr["pop_0_12"])
        lhs_existing = quicksum((facs_idx.loc[f,"cap_total"] + x1[f] + x2[f] + x3[f]) for f in Fz.get(z, []))
        lhs_new = quicksum(uE[(l,s)] + uK[(l,s)] for l in Lz.get(z, []) for s in S)
        m.addConstr(lhs_existing + lhs_new >= alpha * pop012, name=f"C8_desert[{z}]")
    for _, zr in zdf.iterrows():
        z = zr["zip"]; pop05 = float(zr["pop_0_5"])
        lhs_existing = quicksum((facs_idx.loc[f,"cap_0_5"] + rE[f]) for f in Fz.get(z, []))
        lhs_new = quicksum(uE[(l,s)] for l in Lz.get(z, []) for s in S)
        m.addConstr(lhs_existing + lhs_new >= (2.0/3.0) * pop05, name=f"C9_0_5[{z}]")

    m.optimize()
    status = m.Status
    print("Primary model status =", status)

    def val(v):
        try: return float(v.X)
        except Exception: return 0.0

    has_solution = False
    if status == GRB.OPTIMAL:
        has_solution = True
    elif status in (GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        try:
            if m.SolCount and m.SolCount > 0: has_solution = True
        except Exception: has_solution = False

    os.makedirs(out_dir, exist_ok=True)

    if has_solution:
        print("Objective (Total Cost):", getattr(m, "ObjVal", None))
        # write outputs
        picks, exps, rows = [], [], []
        for l in LIDS:
            for s in S:
                if val(y[(l,s)]) > 0.5:
                    picks.append({
                        "lid": l, "zip": str(cands.loc[cands["lid"]==l, "zip"].iloc[0]),
                        "size": s, "uE": val(uE[(l,s)]), "uK": val(uK[(l,s)]),
                        "build_cost": SIZES[s]["build_cost"], "equip_cost_0_5": EQUIP_COST_PER_0_5 * val(uE[(l,s)])
                    })
        builds_df = pd.DataFrame(picks)

        for f in F:
            vx1, vx2, vx3 = val(x1[f]), val(x2[f]), val(x3[f])
            if (vx1 + vx2 + vx3) > 1e-6:
                g1, g2, g3 = gammas[f]
                exps.append({
                    "fid": f, "zip": str(facs_idx.loc[f, "zip"]),
                    "x1_0_10": vx1, "x2_10_15": vx2, "x3_15_20": vx3,
                    "rE": val(rE[f]), "rK": val(rK[f]),
                    "tier_cost": g1*vx1 + g2*vx2 + g3*vx3,
                    "equip_cost_0_5": EQUIP_COST_PER_0_5 * val(rE[f])
                })
        exps_df = pd.DataFrame(exps)

        for _, zr in zdf.iterrows():
            z = zr["zip"]; pop05 = float(zr["pop_0_5"]); pop012 = float(zr["pop_0_12"]); alpha = float(zr["alpha"])
            supply012 = sum(float(facs_idx.loc[f, "cap_total"]) + val(x1[f]) + val(x2[f]) + val(x3[f]) for f in Fz.get(z, []))
            supply012 += sum(val(uE[(l,s)]) + val(uK[(l,s)]) for l in Lz.get(z, []) for s in S)
            supply05 = sum(float(facs_idx.loc[f, "cap_0_5"]) + val(rE[f]) for f in Fz.get(z, []))
            supply05 += sum(val(uE[(l,s)]) for l in Lz.get(z, []) for s in S)
            rows.append({
                "zip": z, "alpha": alpha, "pop_0_5": pop05, "pop_0_12": pop012,
                "supply_0_5": supply05, "supply_0_12": supply012,
                "desert_req": alpha * pop012, "zero_to_five_req": (2.0/3.0) * pop05,
                "desert_met": supply012 >= alpha * pop012 - 1e-6,
                "zero_to_five_met": supply05 >= (2.0/3.0) * pop05 - 1e-6
            })
        zip_df = pd.DataFrame(rows)

        builds_df.to_csv(os.path.join(out_dir, "new_builds.csv"), index=False)
        exps_df.to_csv(os.path.join(out_dir, "expansions.csv"), index=False)
        zip_df.to_csv(os.path.join(out_dir, "zip_summary.csv"), index=False)
        print("Saved new_builds.csv, expansions.csv, zip_summary.csv")
        return

    # ----- Diagnostic soft-constraint model (no IIS; finds which constraints need violation) -----
    print("[Diagnostic] Primary infeasible. Solving penalized slack model...")

    md = Model("Task2_Diagnostic"); md.Params.OutputFlag = 1

    y_d  = {(l,s): md.addVar(vtype=GRB.BINARY, name=f"y[{l},{s}]") for l in LIDS for s in S}
    uE_d = {(l,s): md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"uE[{l},{s}]") for l in LIDS for s in S}
    uK_d = {(l,s): md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"uK[{l},{s}]") for l in LIDS for s in S}
    x1_d = {f: md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x1[{f}]") for f in F}
    x2_d = {f: md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x2[{f}]") for f in F}
    x3_d = {f: md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x3[{f}]") for f in F}
    rE_d = {f: md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"rE[{f}]") for f in F}
    rK_d = {f: md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"rK[{f}]") for f in F}

    # slacks
    s_desert = {z: md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s_desert[{z}]") for z in Z}
    s_05     = {z: md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s_05[{z}]") for z in Z}
    s_t1     = {f: md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s_t1[{f}]") for f in F}
    s_t2     = {f: md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s_t2[{f}]") for f in F}
    s_t3     = {f: md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s_t3[{f}]") for f in F}

    # reuse conflicts for diag
    P_pairs, forbid = build_conflicts(cands, facs)
    s_pair   = {(l1,l2): md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s_pair[{l1},{l2}]") for (l1,l2) in P_pairs}
    s_forbid = {l: md.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s_forbid[{l}]") for l in forbid}

    # penalties (order encodes priority)
    PEN_05, PEN_012, PEN_SP, PEN_FB, PEN_TIER = 1e6, 1e5, 5e4, 5e4, 1e3
    md.setObjective(
        quicksum(PEN_05  * s_05[z]     for z in Z) +
        quicksum(PEN_012 * s_desert[z] for z in Z) +
        quicksum(PEN_SP  * s_pair[p]   for p in P_pairs) +
        quicksum(PEN_FB  * s_forbid[l] for l in forbid) +
        quicksum(PEN_TIER* (s_t1[f] + s_t2[f] + s_t3[f]) for f in F),
        GRB.MINIMIZE
    )

    # relaxed constraints
    for l in LIDS:
        md.addConstr(quicksum(y_d[(l,s)] for s in S) <= 1, name=f"D1_onesize[{l}]")
        for s in S:
            K, KE = SIZES[s]["K"], SIZES[s]["K_E"]
            md.addConstr(uE_d[(l,s)] + uK_d[(l,s)] == K * y_d[(l,s)], name=f"D2_balance[{l},{s}]")
            md.addConstr(uE_d[(l,s)] <= KE * y_d[(l,s)], name=f"D3_capE[{l},{s}]")
    for (l1, l2) in P_pairs:
        md.addConstr(quicksum(y_d[(l1,s)] for s in S) + quicksum(y_d[(l2,s)] for s in S) <= 1 + s_pair[(l1,l2)], name=f"D4_spacing[{l1},{l2}]")
    for l in forbid:
        md.addConstr(quicksum(y_d[(l,s)] for s in S) <= s_forbid[l], name=f"D5_forbid[{l}]")
    for f in F:
        Ntot = float(facs_idx.loc[f, "cap_total"])
        md.addConstr(x1_d[f] <= TIER_BOUNDS[0] * Ntot + s_t1[f], name=f"D6_t1cap[{f}]")
        md.addConstr(x2_d[f] <= TIER_BOUNDS[1] * Ntot + s_t2[f], name=f"D6_t2cap[{f}]")
        md.addConstr(x3_d[f] <= TIER_BOUNDS[2] * Ntot + s_t3[f], name=f"D6_t3cap[{f}]")
        md.addConstr(rE_d[f] + rK_d[f] == x1_d[f] + x2_d[f] + x3_d[f], name=f"D7_split[{f}]")
    for _, zr in zdf.iterrows():
        z = zr["zip"]; alpha = float(zr["alpha"]); pop012 = float(zr["pop_0_12"])
        lhs_existing = quicksum((facs_idx.loc[f,"cap_total"] + x1_d[f] + x2_d[f] + x3_d[f]) for f in Fz.get(z, []))
        lhs_new = quicksum(uE_d[(l,s)] + uK_d[(l,s)] for l in Lz.get(z, []) for s in S)
        md.addConstr(lhs_existing + lhs_new + s_desert[z] >= alpha * pop012, name=f"D8_desert[{z}]")
    for _, zr in zdf.iterrows():
        z = zr["zip"]; pop05 = float(zr["pop_0_5"])
        lhs_existing = quicksum((facs_idx.loc[f,"cap_0_5"] + rE_d[f]) for f in Fz.get(z, []))
        lhs_new = quicksum(uE_d[(l,s)] for l in Lz.get(z, []) for s in S)
        md.addConstr(lhs_existing + lhs_new + s_05[z] >= (2.0/3.0) * pop05, name=f"D9_0_5[{z}]")

    md.optimize()

    # write diagnostic CSVs
    def vval(v):
        try: return float(v.X)
        except Exception: return 0.0

    pd.DataFrame(
        [{"zip": zr["zip"], "s_0_5": vval(s_05[zr["zip"]]), "s_desert": vval(s_desert[zr["zip"]])} for _, zr in zdf.iterrows()]
    ).to_csv(os.path.join(out_dir, "diagnosis_zip_slacks.csv"), index=False)

    pd.DataFrame(
        [{"l1": l1, "l2": l2, "s_pair": vval(s_pair[(l1,l2)])} for (l1,l2) in P_pairs if vval(s_pair[(l1,l2)]) > 1e-6]
    ).to_csv(os.path.join(out_dir, "diagnosis_spacing_slacks.csv"), index=False)

    pd.DataFrame(
        [{"lid": l, "s_forbid": vval(s_forbid[l])} for l in forbid if vval(s_forbid[l]) > 1e-6]
    ).to_csv(os.path.join(out_dir, "diagnosis_forbid_slacks.csv"), index=False)

    tier_rows = []
    for f in F:
        vt1, vt2, vt3 = vval(s_t1[f]), vval(s_t2[f]), vval(s_t3[f])
        if (vt1+vt2+vt3) > 1e-9:
            tier_rows.append({"fid": f, "zip": str(facs_idx.loc[f, "zip"]), "s_t1": vt1, "s_t2": vt2, "s_t3": vt3})
    pd.DataFrame(tier_rows).to_csv(os.path.join(out_dir, "diagnosis_expansion_slacks.csv"), index=False)

    print("Wrote diagnostic CSVs in Task2_Output/: diagnosis_zip_slacks.csv, diagnosis_spacing_slacks.csv, diagnosis_forbid_slacks.csv, diagnosis_expansion_slacks.csv")

if __name__ == "__main__":
    solve_task2()
