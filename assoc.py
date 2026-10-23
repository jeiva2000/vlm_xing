"""Asociación por evento a partir de los scores del VLM, y métricas que importan online.

score_ij = (logit_yes - logit_no) del par (memoria i, candidato j)
s_ij     = score_ij + beta * [candidato j tiene el mismo ID que la memoria i]
           (beta = sesgo hacia la decisión del tracker: el VLM solo la cambia si está seguro)
Hungarian maximizando s; se descarta la asignación si s_ij < tau.

Métrica principal: por memoria, ¿coincide la decisión final con la verdad?
  arreglados = tracker mal  -> final bien
  rotos      = tracker bien -> final mal
  neto       = arreglados - rotos      (lo que se selecciona, no la accuracy por par)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def group_scores(rows: list[dict]) -> dict[int, dict[tuple[int, int], float]]:
    g = defaultdict(dict)
    for r in rows:
        g[r["event"]][(r["mem"], r["cand"])] = r["score"]
    return g


def decide(ev: dict, sc: dict[tuple[int, int], float], beta: float, tau: float) -> list:
    mems, cands = ev["memories"], ev["candidates"]
    S = np.full((len(mems), len(cands)), -1e6)
    for (i, j), v in sc.items():
        S[i, j] = v + beta * (cands[j]["tid"] == mems[i]["tid"])
    out = [None] * len(mems)
    r, c = linear_sum_assignment(-S)
    for i, j in zip(r, c):
        if S[i, j] >= tau:
            out[i] = cands[j]["tid"]
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def event_metrics(events: list[dict], rows: list[dict], beta: float, tau: float = 0.0,
                  min_purity: float = 0.0) -> dict:
    g = group_scores(rows)
    st = defaultdict(int)
    for ei, sc in g.items():
        ev = events[ei]
        final = decide(ev, sc, beta, tau)
        for m, fin in zip(ev["memories"], final):
            if m["purity"] < min_purity:
                continue
            truth, trk = m["truth_choice"], m["tracker_choice"]
            t_ok, f_ok = trk == truth, fin == truth
            key = "clean" if m["purity"] >= 0.999 else "contam"
            st["n"] += 1
            st[f"n_{key}"] += 1
            st["tracker_ok"] += t_ok
            st["final_ok"] += f_ok
            st["fixed"] += (not t_ok) and f_ok
            st["broken"] += t_ok and (not f_ok)
            st[f"fixed_{key}"] += (not t_ok) and f_ok
            st[f"broken_{key}"] += t_ok and (not f_ok)
    n = max(st["n"], 1)
    return {"beta": beta, "tau": tau, **st, "net": st["fixed"] - st["broken"],
            "acc_tracker": st["tracker_ok"] / n, "acc_final": st["final_ok"] / n}


def pair_metrics(rows: list[dict]) -> dict:
    s = np.array([r["score"] for r in rows])
    y = np.array([r["label"] for r in rows])
    out = {"pairs": len(rows), "pos_rate": float(y.mean()) if len(y) else 0.0,
           "auc": auc(s, y), "acc@0": float(((s > 0) == y).mean()) if len(y) else 0.0}
    # AUC solo en eventos donde el tracker falló (lo difícil)
    hard = np.array([r["tracker_error"] for r in rows], dtype=bool)
    out["auc_tracker_error_events"] = auc(s[hard], y[hard]) if hard.any() else float("nan")
    return out


def sweep(events, rows, betas=(0, 1, 2, 4, 8, 1e3), taus=(-2.0, 0.0, 2.0), min_purity=0.0):
    return [event_metrics(events, rows, b, t, min_purity) for b in betas for t in taus]
