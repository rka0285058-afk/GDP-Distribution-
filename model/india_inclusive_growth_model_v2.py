"""
India inclusive growth distribution model - v2
----------------------------------------------
Extends v1 with:
  * 14 state/region lenses (sector mix + relative income level)
  * 10 income deciles (sector mix gradient + within-sector earnings spread) -> Gini
  * a policy-cost layer: extra investment (ICOR x extra sector growth),
    labour reallocation cost (skilling/relocation per mover) and the
    transfer cost of simply equalising decile growth on trend growth
  * a cost-vs-inclusion frontier (sweep of the inclusion weight)

Each year an optimiser chooses sector growth g_s and employment-share shifts d_s to
  - hit the aggregate GDP growth target exactly,
  - minimise dispersion of earnings growth within each lens
    (demographics, states, deciles),
  - stay near the trend growth vector (feasibility penalty).

IMPORTANT: every number in the DATA block is a rounded, illustrative placeholder
at the magnitude of MoSPI / PLFS / NSDP / HCES figures. Replace with official
series before drawing conclusions. Costs are in real terms (Rs, FY27 prices).
"""
import csv
import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

# ------------------------------ DATA (placeholders) --------------------------
SECTORS = ["Agriculture", "Mining", "Manufacturing", "Utilities", "Construction",
           "Trade & hotels", "Transport & comm", "Finance/RE/Prof", "Public & other svc"]
GVA = np.array([15.5, 1.8, 17.5, 2.4, 9.0, 11.5, 6.0, 21.5, 14.8]) / 100
EMP = np.array([46.1, 0.4, 11.4, 0.6, 12.0, 12.6, 5.0, 3.5, 8.4]) / 100
TREND_G = np.array([3.6, 3.0, 8.0, 6.5, 8.5, 7.0, 7.5, 8.5, 7.0])
TREND_SHIFT = np.array([-0.005, 0, 0.001, 0, 0.0015, 0.001, 0.0005, 0.0005, 0.0005])
LABOUR_GROWTH = 0.012

ICOR = np.array([6.0, 5.0, 4.5, 7.0, 3.5, 3.0, 3.5, 3.0, 3.0])  # capital per unit output
MOVER_COST = 1.5e5      # Rs per worker moved (skilling, relocation, job creation)
LF0 = 600e6             # workforce, FY27
GDP0 = 360e12           # Rs, real FY27 base
LABOUR_SHARE = 0.55     # share of GDP paid as labour earnings (transfer base)

DEMOG = {
    "Rural male":   [50, 1, 10, .5, 17, 9, 4, 1, 7.5],
    "Rural female": [72, .1, 12, .2, 4, 5, .5, 1, 5.2],
    "Urban male":   [5, .5, 20, 1.2, 12, 22, 11, 10, 18.3],
    "Urban female": [9, .2, 25, .5, 3, 13, 2, 15, 32.3],
    "Youth 15-29":  [38, .4, 13, .5, 15, 14, 6, 4, 9.1],
    "Adult 30-59":  [45, .4, 11, .6, 12, 13, 5, 3.5, 9.5],
    "Older 60+":    [70, .1, 6, .1, 4, 10, 2, 1, 6.8],
    "SC":           [40, .6, 12, .5, 17, 11, 5, 2, 11.9],
    "ST":           [65, 1, 6, .4, 12, 5, 2, 1, 7.6],
    "OBC":          [47, .4, 11, .6, 12, 12, 5, 3, 9],
    "General":      [30, .4, 13, .8, 10, 16, 6, 8, 15.8],
}

# state: (population share %, income index India=100, agri emp share %, manuf tilt, services tilt)
STATES = {
    "Uttar Pradesh": (16.5, 55, 48, .8, .9), "Bihar": (9.0, 33, 53, .4, .9),
    "Maharashtra": (9.0, 135, 30, 1.4, 1.1), "Tamil Nadu": (5.5, 160, 28, 1.8, 1.1),
    "Gujarat": (5.0, 145, 38, 1.9, .8), "Karnataka": (4.7, 165, 40, 1.2, 1.2),
    "West Bengal": (6.4, 85, 38, .9, 1.0), "Rajasthan": (5.8, 80, 52, .8, .8),
    "Madhya Pradesh": (6.0, 70, 55, .6, .7), "Kerala": (2.4, 160, 22, .8, 1.5),
    "Odisha": (3.2, 80, 52, .8, .8), "Punjab & Haryana": (4.3, 150, 36, 1.2, 1.0),
    "Delhi": (1.4, 330, 3, .8, 2.2), "Rest of India": (20.8, 90, 45, 1.0, 1.0),
}

DEC_LOW = np.array([62, .3, 6, .1, 15, 7, 3, .5, 6.1])
DEC_HIGH = np.array([8, .5, 17, 1.2, 6, 14, 7, 20, 26.3])
DEC_PSI = np.array([.20, .33, .43, .54, .66, .80, 1.0, 1.3, 1.85, 3.9])  # within-sector earnings spread
DEC_PSI = DEC_PSI / DEC_PSI.mean()

# ------------------------------- build lenses -------------------------------
def state_mix(a, tm, ts):
    mix = EMP.copy()
    mix[2] *= tm
    mix[[5, 6, 7, 8]] *= ts
    non = mix[1:] / mix[1:].sum() * (1 - a / 100)
    return np.concatenate([[a / 100], non])

names, rows, LENS = [], [], {}
def add(lens, nm, mix):
    LENS.setdefault(lens, []).append(len(names))
    names.append(nm)
    rows.append(np.array(mix, float) / np.sum(mix))

for k, v in DEMOG.items():
    add("demographic", k, v)
for k, (pop, inc, a, tm, ts) in STATES.items():
    add("state", k, state_mix(a, tm, ts))
for d in range(10):
    t = d / 9
    add("decile", f"D{d + 1}", (1 - t) * DEC_LOW / 100 + t * DEC_HIGH / 100)
M0 = np.array(rows)
LENS = {k: np.array(v) for k, v in LENS.items()}
ST, DEC = LENS["state"], LENS["decile"]
ST_POP = np.array([v[0] for v in STATES.values()]) / 100
ST_RHO = np.array([v[1] for v in STATES.values()], float)
ST_RHO = ST_RHO / (ST_POP @ ST_RHO) * 100


def gini(x):
    x = np.asarray(x, float)
    return np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) ** 2 * x.mean())

# --------------------------------- engine -----------------------------------
def step(w, e, m, g, d):
    g = g / 100
    e2 = e + d
    l = e2 * (1 + LABOUR_GROWTH) / e - 1
    y_g = (1 + g) / (1 + l)
    y = w / e
    I0 = m @ y
    m2 = m * (e2 / e)
    m2 /= m2.sum(1, keepdims=True)
    I1 = m2 @ (y * y_g)
    w2 = w * (1 + g)
    w2 /= w2.sum()
    return w2, e2, m2, (I1 / I0 - 1) * 100, (1 + g) @ w - 1


def lens_var(grp):
    return sum(np.var(grp[ix]) for ix in LENS.values())


def solve_year(w, e, m, G, lam):
    base = TREND_G * G / (w @ TREND_G)
    if minimize is None or lam == 0:
        return base, TREND_SHIFT
    n = len(w)

    def obj(x):
        g, d = x[:n], x[n:]
        grp = step(w, e, m, g, d)[3]
        return lam * lens_var(grp) + 0.02 * np.sum(((g - base) / base) ** 2)

    cons = [{"type": "eq", "fun": lambda x: w @ x[:n] - G},
            {"type": "eq", "fun": lambda x: x[n:].sum()}]
    bnds = [(0.6 * b, 1.6 * b) for b in base] + [(max(-0.010, 0.002 - ei), 0.010) for ei in e]
    x0 = np.concatenate([base, TREND_SHIFT])
    r = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 300, "ftol": 1e-10})
    x = r.x if r.success else x0
    return x[:n], x[n:]


def run(G, lam=10.0, inclusive=True, years=21):
    w, e, m = GVA.copy(), EMP.copy(), M0.copy()
    I0 = m[DEC] @ (w / e)
    L = DEC_PSI * I0 / I0.mean()          # decile earnings levels
    L0, Ltar = L.copy(), L.copy()
    idx = np.ones(len(names))
    gdp, gdp_mult = GDP0, 1.0
    inv, rea, tra, cum_s, cum_t = [], [], [], 0.0, 0.0
    for t in range(years):
        base = TREND_G * G / (w @ TREND_G)
        g, d = solve_year(w, e, m, G, lam) if inclusive else (base, TREND_SHIFT)
        w, e, m, grp, agg = step(w, e, m, g, d)
        idx *= 1 + grp / 100
        gbar = (L @ grp[DEC]) / L.sum()
        L = L * (1 + grp[DEC] / 100)
        Ltar = Ltar * (1 + gbar / 100)
        c_inv = float(np.sum(ICOR * np.maximum(0, g - base) / 100 * w))
        c_rea = float(np.maximum(0, d).sum() * LF0 * (1 + LABOUR_GROWTH) ** t * MOVER_COST / gdp)
        c_tra = float(np.maximum(0, Ltar - L).sum() / L.sum() * LABOUR_SHARE)
        inv.append(c_inv); rea.append(c_rea); tra.append(c_tra)
        cum_s += (c_inv + c_rea) * gdp
        cum_t += c_tra * gdp
        gdp *= 1 + agg
        gdp_mult *= 1 + agg
        if t == 0:
            g_first = g.copy()
    st_inc = ST_RHO * idx[ST]
    ex = [i for i, k in enumerate(STATES) if k != "Delhi"]
    cvs = {k: float(np.std(idx[ix]) / np.mean(idx[ix])) for k, ix in LENS.items()}
    return dict(G=G, idx=idx, gdp_mult=gdp_mult, g_first=g_first, g_last=g, e_end=e, w_end=w,
                gini0=gini(L0), gini_end=gini(L), gini_equalised=gini(Ltar),
                st_inc=st_inc, st_ratio=st_inc.max() / st_inc.min(),
                st_ratio_ex_delhi=st_inc[ex].max() / st_inc[ex].min(),
                cv=cvs, cost_inv=np.mean(inv) * 100, cost_rea=np.mean(rea) * 100,
                cost_tra=np.mean(tra) * 100, cum_struct=cum_s / 1e12, cum_transfer=cum_t / 1e12,
                dec_end=L)


def main():
    out = []
    for G in (6.5, 6.9, 7.3):
        t, i = run(G, inclusive=False), run(G, inclusive=True)
        for label, r, gini_end, tra_cost, cum_tr in (
                ("Trend", t, t["gini_end"], 0.0, 0.0),
                ("Trend + transfers only", t, t["gini_equalised"], t["cost_tra"], t["cum_transfer"]),
                ("Inclusive restructuring", i, i["gini_end"], 0.0, 0.0)):
            out.append(dict(
                scenario=label, target_pct=G, gdp_multiple_2047=round(r["gdp_mult"], 2),
                gini_2026=round(r["gini0"], 3), gini_2047=round(gini_end, 3),
                state_income_ratio_2047=round(r["st_ratio"], 2),
                state_income_ratio_ex_delhi=round(r["st_ratio_ex_delhi"], 2),
                cv_demographic=round(r["cv"]["demographic"], 3), cv_state=round(r["cv"]["state"], 3),
                cv_decile=round(r["cv"]["decile"], 3),
                avg_structural_cost_pct_gdp=round(0 if label == "Trend + transfers only" else r["cost_inv"] + r["cost_rea"], 2),
                avg_transfer_cost_pct_gdp=round(tra_cost, 2),
                cumulative_cost_lakh_cr=round(cum_tr if label == "Trend + transfers only" else r["cum_struct"], 0)))
    with open("v2_scenario_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader(); wr.writerows(out)

    # detailed tables at 6.9%
    t, i = run(6.9, inclusive=False), run(6.9, inclusive=True)
    with open("v2_state_table_6.9.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["state", "pop_share_pct", "income_index_2026", "trend_multiple", "inclusive_multiple",
                     "trend_income_index_2047", "inclusive_income_index_2047"])
        for j, k in enumerate(STATES):
            wr.writerow([k, STATES[k][0], round(ST_RHO[j], 1), round(t["idx"][ST[j]], 2),
                         round(i["idx"][ST[j]], 2), round(t["st_inc"][j], 0), round(i["st_inc"][j], 0)])
    with open("v2_decile_table_6.9.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["decile", "trend_multiple", "inclusive_multiple", "trend_level_2047", "inclusive_level_2047"])
        for j, d in enumerate(DEC):
            wr.writerow([f"D{j + 1}", round(t["idx"][d], 2), round(i["idx"][d], 2),
                         round(t["dec_end"][j], 2), round(i["dec_end"][j], 2)])

    # cost vs inclusion frontier at 6.9%
    front = []
    for lam in (0, 2, 10, 50, 250):
        r = run(6.9, lam=lam, inclusive=True)
        front.append(dict(inclusion_weight=lam, avg_structural_cost_pct_gdp=round(r["cost_inv"] + r["cost_rea"], 3),
                          gini_2047=round(r["gini_end"], 3), state_income_ratio_ex_delhi=round(r["st_ratio_ex_delhi"], 2),
                          cv_state=round(r["cv"]["state"], 3), cv_decile=round(r["cv"]["decile"], 3),
                          gdp_multiple_2047=round(r["gdp_mult"], 3)))
    with open("v2_cost_frontier_6.9.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(front[0]))
        wr.writeheader(); wr.writerows(front)

    print("SCENARIOS")
    for r in out: print(r)
    print("\nFRONTIER")
    for r in front: print(r)
    print("\nSTATES 6.9% (trend x, inclusive x, income idx 2047 trend/incl)")
    for j, k in enumerate(STATES):
        print(f"{k:18s} {t['idx'][ST[j]]:.2f} {i['idx'][ST[j]]:.2f} {t['st_inc'][j]:6.0f} {i['st_inc'][j]:6.0f}")
    print("\nDECILES 6.9% (trend x, inclusive x)")
    for j, d in enumerate(DEC):
        print(f"D{j + 1:<2d} {t['idx'][d]:.2f} {i['idx'][d]:.2f}")
    print("\nSECTOR g% inclusive FY27 / FY47 vs trend FY27")
    for s, a, b, c in zip(SECTORS, t["g_first"], i["g_first"], i["g_last"]):
        print(f"{s:20s} {a:5.1f} {b:5.1f} {c:5.1f}")
    print("agri emp share 2047 trend/incl:", round(t["e_end"][0] * 100, 1), round(i["e_end"][0] * 100, 1))
    return t, i, front


def chart(t, i, front):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#1f2328", "#5b6470", "#e3e6ea"
    C_TREND, C_INCL = "#2a6fdb", "#d9822b"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    a = ax[0]
    xs = [f["avg_structural_cost_pct_gdp"] for f in front]
    ys = [f["gini_2047"] for f in front]
    a.plot(xs, ys, color=C_INCL, lw=2, marker="o", ms=6, mec="white", mew=1.5)
    for f, x, y in list(zip(front, xs, ys))[:3]:
        a.annotate(f"weight {f['inclusion_weight']}", (x, y), textcoords="offset points", xytext=(4, -14), fontsize=8, color=MUTED)
    a.annotate("weight 50-250", (xs[-1], ys[-1]), textcoords="offset points", xytext=(-4, -14), ha="right", fontsize=8, color=MUTED)
    a.axhline(t["gini0"], color=MUTED, lw=1, ls="--")
    a.text(0.02, t["gini0"] + 0.0015, "Gini FY2026", transform=a.get_yaxis_transform(), fontsize=8, color=MUTED)
    a.set_ylim(0.54, 0.59)
    a.set_xlabel("Average structural cost (% of GDP per year)", color=INK)
    a.set_ylabel("Decile earnings Gini in FY2047", color=INK)
    a.set_title("Cost vs inclusion at 6.9% growth", color=INK, fontsize=11, loc="left")
    b = ax[1]
    ex = [j for j, k in enumerate(STATES) if k != "Delhi"]
    order = sorted(ex, key=lambda j: t["st_inc"][j])
    yy = np.arange(len(order))
    b.barh(yy - 0.19, [t["st_inc"][j] for j in order], 0.36, color=C_TREND, label="Trend")
    b.barh(yy + 0.19, [i["st_inc"][j] for j in order], 0.36, color=C_INCL, label="Inclusive")
    b.set_yticks(yy)
    b.set_yticklabels([list(STATES)[j] for j in order], fontsize=8, color=INK)
    b.set_xlabel("Earnings index FY2047 (India avg FY2026 = 100)", color=INK)
    b.set_title("State earnings, FY2047 (excl. Delhi)", color=INK, fontsize=11, loc="left")
    b.legend(frameon=False, fontsize=9, loc="lower right")
    for x in ax:
        x.grid(color=GRID, lw=0.6)
        x.set_axisbelow(True)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            x.spines[sp].set_color(GRID)
        x.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig("v2_cost_and_states_6.9.png", dpi=140)


if __name__ == "__main__":
    print("scipy optimizer:", minimize is not None)
    t, i, front = main()
    try:
        chart(t, i, front)
    except Exception as ex:
        print("chart skipped:", ex)
