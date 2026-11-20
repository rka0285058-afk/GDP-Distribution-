"""Precompute the scenario grid for the dashboard using the v2 model engine.
Run from anywhere: python model/gen_data.py  (writes ../data.json, takes about 2 minutes)."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import india_inclusive_growth_model_v2 as m

def run_path(G, lam, years=21):
    w, e, mm = m.GVA.copy(), m.EMP.copy(), m.M0.copy()
    I0 = mm[m.DEC] @ (w / e)
    L = m.DEC_PSI * I0 / I0.mean()
    L0, Ltar = L.copy(), L.copy()
    gdp = m.GDP0
    P = dict(grp=[], g=[], e=[], w=[], exc=[], posd=[], tra=[], gini=[])
    for t in range(years):
        base = m.TREND_G * G / (w @ m.TREND_G)
        g, d = m.solve_year(w, e, mm, G, lam)
        exc = w * np.maximum(0, g - base) / 100
        w_pre = w.copy()
        w, e, mm, grp, agg = m.step(w, e, mm, g, d)
        gbar = (L @ grp[m.DEC]) / L.sum()
        L = L * (1 + grp[m.DEC] / 100)
        Ltar = Ltar * (1 + gbar / 100)
        tra = float(np.maximum(0, Ltar - L).sum() / L.sum() * m.LABOUR_SHARE)
        P["grp"].append(np.round(grp, 3).tolist())
        P["g"].append(np.round(g, 3).tolist())
        P["e"].append(np.round(e, 4).tolist())
        P["w"].append(np.round(w_pre, 4).tolist())
        P["exc"].append(np.round(exc, 6).tolist())
        P["posd"].append(round(float(np.maximum(0, d).sum()), 6))
        P["tra"].append(round(tra, 6))
        P["gini"].append(round(m.gini(L), 4))
    return P

targets = [6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3]
weights = [0, 1, 3, 10, 50]
runs = {}
t0 = time.time()
for G in targets:
    for lam in weights:
        runs[f"{G}|{lam}"] = run_path(G, lam)
    print(G, round(time.time() - t0), "s", flush=True)

I0 = m.M0[m.DEC] @ (m.GVA / m.EMP)
L0 = m.DEC_PSI * I0 / I0.mean()
meta = dict(
    sectors=m.SECTORS, names=m.names,
    lens={k: v.tolist() for k, v in m.LENS.items()},
    gva=m.GVA.tolist(), emp=m.EMP.tolist(), icor=m.ICOR.tolist(),
    mover_cost=m.MOVER_COST, lf0=m.LF0, gdp0=m.GDP0, labour_growth=m.LABOUR_GROWTH,
    labour_share=m.LABOUR_SHARE, st_rho=np.round(m.ST_RHO, 2).tolist(),
    st_pop=np.round(m.ST_POP, 4).tolist(), l0=np.round(L0, 5).tolist(),
    states=list(m.STATES), targets=targets, weights=weights, years=21,
)
json.dump(dict(meta=meta, runs=runs), open(HERE.parent / "data.json", "w"), separators=(",", ":"))
print("done", round(time.time() - t0), "s")
