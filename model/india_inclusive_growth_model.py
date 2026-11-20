"""
India inclusive growth distribution model
-----------------------------------------
Question: if India grows 6.5-7.3% a year, which sector growth mix and
labour reallocation spreads the gains most evenly across sectors and
demographic groups?

Method (annual, FY2027-FY2047):
  1. Trend scenario: sector growth follows historical-style differentials
     (scaled to hit the aggregate target), labour leaves agriculture slowly.
  2. Inclusive scenario: each year an optimizer picks sector growth rates g_s
     and employment-share shifts d_s that
        - hit the aggregate GDP growth target exactly,
        - minimise the dispersion of earnings growth across demographic groups,
        - stay near the trend growth vector (feasibility penalty),
        - respect bounds on sector growth and annual labour movement.
  3. Group earnings = sum over sectors of (group's employment mix x sector
     value added per worker).

IMPORTANT: sector shares and group employment mixes below are rounded,
illustrative placeholders based on public MoSPI (National Accounts, PLFS)
magnitudes. Replace the DATA block with official series before using
results for decisions.
"""
import numpy as np
import csv

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

# ----------------------------- DATA (placeholders) --------------------------
SECTORS = ["Agriculture", "Mining", "Manufacturing", "Utilities", "Construction",
           "Trade & hotels", "Transport & comm", "Finance/RE/Prof", "Public & other svc"]
GVA = np.array([15.5, 1.8, 17.5, 2.4, 9.0, 11.5, 6.0, 21.5, 14.8]) / 100
EMP = np.array([46.1, 0.4, 11.4, 0.6, 12.0, 12.6, 5.0, 3.5, 8.4]) / 100
TREND_G = np.array([3.6, 3.0, 8.0, 6.5, 8.5, 7.0, 7.5, 8.5, 7.0])  # % unscaled
TREND_SHIFT = np.array([-0.005, 0, 0.001, 0, 0.0015, 0.001, 0.0005, 0.0005, 0.0005])
LABOUR_GROWTH = 0.012

GROUPS = {  # employment mix across SECTORS (normalised in code)
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
GN = list(GROUPS)
M0 = np.array([GROUPS[k] for k in GN], float)
M0 /= M0.sum(1, keepdims=True)


def step(w, e, m, g, d):
    """One year. g in %, d = employment share shift. Returns new state + group growth."""
    g = g / 100
    e2 = e + d
    l = e2 * (1 + LABOUR_GROWTH) / e - 1            # sector employment growth
    y_g = (1 + g) / (1 + l)                         # per-worker VA growth factor
    y = w / e
    I0 = m @ y
    m2 = m * (e2 / e)
    m2 /= m2.sum(1, keepdims=True)
    I1 = m2 @ (y * y_g)
    grp = I1 / I0 - 1
    w2 = w * (1 + g)
    w2 /= w2.sum()
    return w2, e2, m2, grp * 100, (1 + g) @ w - 1


def solve_year(w, e, m, G):
    base = TREND_G * G / (w @ TREND_G)
    if minimize is None:
        return base, TREND_SHIFT
    n = len(w)

    def obj(x):
        g, d = x[:n], x[n:]
        _, _, _, grp, _ = step(w, e, m, g, d)
        return np.var(grp) * 10 + 0.02 * np.sum(((g - base) / base) ** 2)

    cons = [{"type": "eq", "fun": lambda x: w @ x[:n] - G},
            {"type": "eq", "fun": lambda x: x[n:].sum()}]
    bnds = [(0.6 * b, 1.6 * b) for b in base] + \
           [(max(-0.010, 0.002 - ei), 0.010) for ei in e]
    x0 = np.concatenate([base, TREND_SHIFT])
    r = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 300, "ftol": 1e-10})
    x = r.x if r.success else x0
    return x[:n], x[n:]


def run(G, years=21, inclusive=True):
    w, e, m = GVA.copy(), EMP.copy(), M0.copy()
    rows, gpath, epath = [], [], [e.copy()]
    gdp = 1.0
    grp_idx = np.ones(len(GN))
    for t in range(years):
        if inclusive:
            g, d = solve_year(w, e, m, G)
        else:
            g, d = TREND_G * G / (w @ TREND_G), TREND_SHIFT
        w, e, m, grp, agg = step(w, e, m, g, d)
        gdp *= 1 + agg
        grp_idx *= 1 + grp / 100
        rows.append(grp)
        gpath.append(g)
        epath.append(e.copy())
    y = w / e
    agri_gap = (y[1:] @ e[1:] / e[1:].sum()) / y[0]
    return dict(G=G, gdp_mult=gdp, grp_idx=grp_idx, grp_yearly=np.array(rows),
                g_first=gpath[0], g_last=gpath[-1], e_end=e, w_end=w,
                cv=np.std(grp_idx) / np.mean(grp_idx), spread=grp_idx.max() / grp_idx.min(),
                agri_gap=agri_gap, nonag_over_ag=agri_gap)


def main():
    out = []
    for G in (6.5, 6.9, 7.3):
        t, i = run(G, inclusive=False), run(G, inclusive=True)
        for label, r in (("Trend", t), ("Inclusive", i)):
            out.append(dict(scenario=label, target=G, gdp_multiple_2047=round(r["gdp_mult"], 2),
                            group_income_cv=round(r["cv"], 3),
                            best_over_worst_group=round(r["spread"], 2),
                            nonag_to_ag_productivity=round(r["agri_gap"], 2),
                            agri_emp_share_2047=round(r["e_end"][0] * 100, 1),
                            agri_gva_share_2047=round(r["w_end"][0] * 100, 1)))
        print(f"\n=== Target {G}% ===")
        print(f"{'Group':14s} {'Trend x':>8s} {'Incl x':>8s}")
        for k, a, b in zip(GN, t["grp_idx"], i["grp_idx"]):
            print(f"{k:14s} {a:8.2f} {b:8.2f}")
        print(f"{'Sector':20s} {'Trend g%':>9s} {'Incl g% FY27':>13s} {'Incl g% FY47':>13s}")
        for s, a, b, c in zip(SECTORS, t["g_first"], i["g_first"], i["g_last"]):
            print(f"{s:20s} {a:9.1f} {b:13.1f} {c:13.1f}")
        if G == 6.9:
            with open("group_income_multiples_6.9.csv", "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["group", "trend_multiple_2047", "inclusive_multiple_2047"])
                for k, a, b in zip(GN, t["grp_idx"], i["grp_idx"]):
                    wr.writerow([k, round(a, 3), round(b, 3)])
    with open("scenario_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print("\nSUMMARY")
    for r in out:
        print(r)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t, i = run(6.9, inclusive=False), run(6.9, inclusive=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(GN))
        ax.bar(x - 0.2, t["grp_idx"], 0.4, label="Trend")
        ax.bar(x + 0.2, i["grp_idx"], 0.4, label="Inclusive")
        ax.set_xticks(x)
        ax.set_xticklabels(GN, rotation=45, ha="right")
        ax.set_ylabel("Earnings per worker, FY2047 vs FY2026 (x)")
        ax.set_title("Group earnings multiple at 6.9% GDP growth")
        ax.legend()
        fig.tight_layout()
        fig.savefig("group_earnings_6.9.png", dpi=130)
    except Exception as ex:
        print("chart skipped:", ex)


if __name__ == "__main__":
    print("scipy optimizer:", minimize is not None)
    main()
