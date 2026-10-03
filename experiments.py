"""Experiment drivers E1-E9.  Each writes one CSV to results/."""
from __future__ import annotations

import csv
import math
import os
import time
from multiprocessing import Pool

import numpy as np

from config import Config, SEED, SLICES
from sim import (Simulator, run_once, PrefixCache, Node, prefill_mean)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
SEEDS = [SEED + i for i in range(10)]
DEFENDED = ("SHARED", "NOISE", "ISOLATED", "SELECTIVE", "PROPOSED-FULL",
            "PROPOSED")


def _job(args):
    kw, policy = args
    _, h, a = run_once(Config(**kw), policy)
    h = {k: v for k, v in h.items() if k not in ("ttft", "per_slice")}
    return kw, policy, h, a


def _run(jobs):
    with Pool(2) as p:
        return p.map(_job, jobs, chunksize=1)


def _write(name, rows, header):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _agg(vals):
    v = np.array(vals, float)
    v = v[~np.isnan(v)]
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


# ---------------------------------------------------------------------------
def e0_main():
    """Table: all policies, 10 seeds, nominal load."""
    jobs = [(dict(seed=s), p) for p in ("NOCACHE",) + DEFENDED for s in SEEDS]
    res = _run(jobs)
    rows = []
    for p in ("NOCACHE",) + DEFENDED:
        R = [r for r in res if r[1] == p]
        row = [p]
        for key in ("ttft_mean", "ttft_p95", "ttft_p99", "hit_ratio", "util",
                    "shaping_mean"):
            row += list(_agg([r[2][key] for r in R]))
        for key in ("success", "auc"):
            row += list(_agg([r[3][key] for r in R]))
        rows.append(row)
    hdr = ["policy"]
    for k in ("ttft_mean", "ttft_p95", "ttft_p99", "hit", "util", "shape",
              "succ", "auc"):
        hdr += [k, k + "_se"]
    _write("e0_main.csv", rows, hdr)
    # e12: pooled per-probe confusion counts of the threshold detector
    crow = []
    for p in DEFENDED:
        R = [r for r in res if r[1] == p]
        tp, fp, fn, tn = (sum(r[3][k] for r in R) for k in ("tp", "fp", "fn", "tn"))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        crow.append([p, tp, fp, fn, tn, round(tpr, 4), round(fpr, 4),
                     round(prec, 4)])
    _write("e12_confusion.csv", crow,
           ["policy", "tp", "fp", "fn", "tn", "tpr", "fpr", "precision"])


def e1_rounds():
    """Attack success vs number of rounds aggregated on one secret."""
    ns = (1, 2, 4, 8, 16)
    jobs = []
    for n in ns:
        for s in SEEDS:
            kw = dict(seed=s, horizon_s=1200.0, n_rounds=96,
                      probes_per_secret=n)
            for p in DEFENDED:
                jobs.append((kw, p))
    res = _run(jobs)
    rows = []
    for n in ns:
        row = [n]
        for p in DEFENDED:
            R = [r for r in res if r[1] == p and
                 r[0]["probes_per_secret"] == n]
            row += list(_agg([r[3]["success"] for r in R]))
        R = [r for r in res if r[1] == "PROPOSED" and
             r[0]["probes_per_secret"] == n]
        row += [float(np.mean([r[3]["bound"] for r in R]))]
        rows.append(row)
    hdr = ["n"] + [x for p in DEFENDED for x in (p, p + "_se")] + ["bound"]
    _write("e1_rounds.csv", rows, hdr)


def e2_cdf():
    """Pooled honest TTFT samples (seed SEED) -> empirical CDF quantiles."""
    qs = np.linspace(0.0, 1.0, 101)
    cols = {}
    for p in DEFENDED:
        _, h, _ = run_once(Config(), p)
        cols[p] = np.quantile(h["ttft"], qs)
    rows = [[f"{q:.2f}"] + [f"{cols[p][i]:.2f}" for p in DEFENDED]
            for i, q in enumerate(qs)]
    _write("e2_cdf.csv", rows, ["q"] + list(DEFENDED))


def e3_load():
    loads = (10, 14, 18, 22, 26, 30, 34)
    pols = ("NOCACHE",) + DEFENDED
    jobs = [(dict(seed=s, agg_rate=float(L), n_rounds=20), p)
            for L in loads for p in pols for s in SEEDS[:5]]
    res = _run(jobs)
    rows = []
    for L in loads:
        row = [L]
        for p in pols:
            R = [r for r in res if r[1] == p and r[0]["agg_rate"] == L]
            row += [_agg([r[2]["ttft_mean"] for r in R])[0],
                    _agg([r[2]["ttft_p95"] for r in R])[0],
                    _agg([r[2]["util"] for r in R])[0]]
        rows.append(row)
    hdr = ["load"] + [x for p in pols for x in (p + "_mean", p + "_p95",
                                                p + "_util")]
    _write("e3_load.csv", rows, hdr)
    grow = []
    for L in loads:
        row = [L]
        for p in pols:
            R = [r for r in res if r[1] == p and r[0]["agg_rate"] == L]
            slo = _agg([r[2]["slo_frac"] for r in R])[0]
            row += [slo, L * slo]
        grow.append(row)
    _write("e11_goodput.csv", grow,
           ["load"] + [x for p in pols for x in (p + "_slo", p + "_goodput")])


def e4_capacity():
    caps = (50_000, 100_000, 200_000, 300_000, 450_000, 600_000)
    pols = ("SHARED", "ISOLATED", "SELECTIVE", "PROPOSED")
    jobs = [(dict(seed=s, cache_tokens=c, n_rounds=20), p)
            for c in caps for p in pols for s in SEEDS[:5]]
    res = _run(jobs)
    rows = []
    for c in caps:
        row = [c // 1000]
        for p in pols:
            R = [r for r in res if r[1] == p and r[0]["cache_tokens"] == c]
            row += [_agg([r[2]["hit_ratio"] for r in R])[0],
                    _agg([r[2]["ttft_mean"] for r in R])[0]]
        rows.append(row)
    hdr = ["cap_k"] + [x for p in pols for x in (p + "_hit", p + "_ttft")]
    _write("e4_capacity.csv", rows, hdr)


def e5_radio():
    """Naive random delay vs attacker slice (radio jitter)."""
    widths = (0, 20, 40, 60, 90, 120)
    atk = ("URLLC-Industrial", "eMBB-Consumer", "mmWave-XR")
    jobs = []
    for w in widths:
        for a in atk:
            for s in SEEDS:
                jobs.append((dict(seed=s, noise_ms=float(w), attacker=a), "NOISE"))
    for a in atk:
        for s in SEEDS:
            jobs.append((dict(seed=s, attacker=a), "PROPOSED-FULL"))
    res = _run(jobs)
    rows = []
    for w in widths:
        row = [w]
        for a in atk:
            R = [r for r in res if r[1] == "NOISE" and r[0]["attacker"] == a
                 and r[0]["noise_ms"] == w]
            row += list(_agg([r[3]["success"] for r in R]))
        R = [r for r in res if r[1] == "NOISE" and r[0]["noise_ms"] == w]
        row += [_agg([r[2]["ttft_mean"] for r in R])[0]]
        rows.append(row)
    prop = []
    for a in atk:
        R = [r for r in res if r[1] == "PROPOSED-FULL" and r[0]["attacker"] == a]
        prop += list(_agg([r[3]["success"] for r in R]))
    rows.append(["proposed"] + prop + [float("nan")])
    hdr = ["width"] + [x for a in atk for x in (a, a + "_se")] + ["ttft"]
    _write("e5_radio.csv", rows, hdr)


def e6_detector():
    fnrs = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
    jobs = [(dict(seed=s, det_fnr=f), p) for f in fnrs
            for p in ("SELECTIVE", "PROPOSED-FULL") for s in SEEDS]
    res = _run(jobs)
    rows = []
    for f in fnrs:
        row = [f]
        for p in ("SELECTIVE", "PROPOSED-FULL"):
            R = [r for r in res if r[1] == p and r[0]["det_fnr"] == f]
            row += list(_agg([r[3]["success"] for r in R]))
        rows.append(row)
    _write("e6_detector.csv", rows,
           ["fnr", "sel", "sel_se", "prop", "prop_se"])


def e7_budget():
    """Budget accrual sweep: leakage spent, attack success, latency saved."""
    eps = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)
    jobs = [(dict(seed=s, eps_rate=e, kappa_max=0.05), "PROPOSED")
            for e in eps for s in SEEDS]
    jobs += [(dict(seed=s), "PROPOSED-FULL") for s in SEEDS]
    res = _run(jobs)
    rows = []
    full = [r for r in res if r[1] == "PROPOSED-FULL"]
    for e in eps:
        R = [r for r in res if r[1] == "PROPOSED" and r[0].get("eps_rate") == e]
        row = [e]
        row += list(_agg([r[3]["kl_total"] for r in R]))
        row += list(_agg([r[3]["success"] for r in R]))
        row += list(_agg([r[3]["auc"] for r in R]))
        row += [_agg([r[3]["bound"] for r in R])[0]]
        row += [_agg([r[2]["shaping_mean"] for r in R])[0]]
        rows.append(row)
    rows.append(["full"] + list(_agg([r[3]["kl_total"] for r in full]))
                + list(_agg([r[3]["success"] for r in full]))
                + list(_agg([r[3]["auc"] for r in full]))
                + [float("nan"), _agg([r[2]["shaping_mean"] for r in full])[0]])
    _write("e7_budget.csv", rows, ["eps", "kl", "kl_se", "succ", "succ_se",
                                   "auc", "auc_se", "bound", "shape"])


def e8_agent():
    """Agent attestation recall: efficiency moves, security does not."""
    recalls = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    jobs = [(dict(seed=s, agent_recall=r0, n_shared_unreg=24), p)
            for r0 in recalls for p in ("PROPOSED-FULL", "SELECTIVE")
            for s in SEEDS[:5]]
    res = _run(jobs)
    rows = []
    for r0 in recalls:
        row = [r0]
        for p in ("PROPOSED-FULL", "SELECTIVE"):
            R = [r for r in res if r[1] == p and r[0]["agent_recall"] == r0]
            row += [_agg([r[2]["foreign_frac"] for r in R])[0],
                    _agg([r[2]["shaping_mean"] for r in R])[0],
                    _agg([r[2]["ttft_mean"] for r in R])[0],
                    _agg([r[2]["ttft_p95"] for r in R])[0],
                    _agg([r[3]["success"] for r in R])[0]]
        rows.append(row)
    hdr = ["recall"] + [x for p in ("full", "sel") for x in
                        (p + "_foreign", p + "_shape", p + "_mean",
                         p + "_p95", p + "_succ")]
    _write("e8_agent.csv", rows, hdr)


def e9_scaling():
    """Micro-benchmarks: per-request lookup + provenance check vs cache size;
    closed-form alpha* vs a 1-D numerical search."""
    rng = np.random.default_rng(SEED)
    rows = []
    for n_nodes in (10**3, 10**4, 10**5, 10**6):
        c = PrefixCache(capacity=10**12)
        keys = [(0, (i, i + 1, i + 2)) for i in range(n_nodes)]
        for k in keys:
            c.insert(k, Node(100, {0}, 0, False))
        probe = [keys[j] for j in rng.integers(0, n_nodes, 20000)]
        t0 = time.perf_counter()
        for k in probe:
            nd = c.get(k)
            _ = (not nd.public) and (3 not in nd.owners)
        t_lookup = (time.perf_counter() - t0) / len(probe) * 1e6
        rows.append(["lookup", n_nodes, t_lookup])
    # alpha: closed form vs grid search over 1000 points
    V, Z, sig, g = 2e-4, 0.3, 2.5, 30.0
    t0 = time.perf_counter()
    for _ in range(20000):
        a = min(1.0, max(0.0, 1 - V * sig ** 2 / (Z * g)))
    t_cf = (time.perf_counter() - t0) / 20000 * 1e6
    grid = np.linspace(0, 1, 1000)
    t0 = time.perf_counter()
    for _ in range(2000):
        cost = V * grid * g + Z * ((1 - grid) * g) ** 2 / (2 * sig ** 2)
        a2 = grid[np.argmin(cost)]
    t_grid = (time.perf_counter() - t0) / 2000 * 1e6
    rows.append(["alpha_closed", 1, t_cf])
    rows.append(["alpha_grid", 1000, t_grid])
    rows.append(["alpha_gap", 0, abs(a - a2)])
    # partition search space: Bell numbers vs template set
    bell = [1]
    tri = [[1]]
    for n in range(1, 13):
        row = [tri[-1][-1]]
        for x in tri[-1]:
            row.append(row[-1] + x)
        tri.append(row)
        bell.append(row[0])
    for S in (4, 6, 8, 10, 12):
        rows.append(["bell", S, bell[S]])
    _write("e9_scaling.csv", rows, ["what", "n", "value"])


def e10_dynamic():
    """Load burst 18 -> 30 -> 18 req/s; windowed mean TTFT over time."""
    prof = ((0.0, 18.0), (200.0, 30.0), (400.0, 18.0))
    pols = ("SHARED", "NOISE", "ISOLATED", "SELECTIVE", "PROPOSED-FULL")
    W = 20.0
    edges = np.arange(0, 600 + W, W)
    jobs = [(dict(seed=s, rate_profile=prof, n_rounds=20), p)
            for p in pols for s in SEEDS[:5]]
    with Pool(2) as pool:
        res = pool.map(_job_series, jobs, chunksize=1)
    rows = []
    for i in range(len(edges) - 1):
        row = [edges[i] + W / 2]
        for p in pols:
            vals = [r[2][i] for r in res if r[1] == p and not math.isnan(r[2][i])]
            row.append(float(np.mean(vals)) if vals else float("nan"))
        rows.append(row)
    _write("e10_dynamic.csv", rows, ["t"] + list(pols))


def _job_series(args):
    kw, policy = args
    _, h, _ = run_once(Config(**kw), policy)
    W = 20.0
    edges = np.arange(0, 600 + W, W)
    idx = np.digitize(h["tsend"], edges) - 1
    out = []
    for i in range(len(edges) - 1):
        m = idx == i
        out.append(float(h["ttft"][m].mean()) if m.any() else float("nan"))
    return kw, policy, out


def e13_emulator():
    """Emulator fidelity: per-release KL between emulated and genuine miss
    prefill-time laws (Gaussian fit, conservative: excludes queue/radio noise),
    and measured attack success, as prefill variability nu grows."""
    nus = (0.02, 0.04, 0.08, 0.12, 0.20)
    rows = []
    for nu in nus:
        cfg = Config(pf_cv=nu, n_rounds=20)
        sim = Simulator(cfg, "SHARED").run()   # emulator fitted online
        rng = np.random.default_rng([SEED, 13, int(nu * 1000)])
        kls = []
        for _ in range(300):
            n_new = int(rng.integers(340, 660))      # secret + suffix tokens
            ctx = int(rng.integers(500, 1300))
            mu = prefill_mean(cfg, n_new, ctx)
            real = mu * rng.lognormal(-0.5 * nu ** 2, nu, 400)
            emu = np.array([sim.emu.sample(n_new, ctx) for _ in range(400)])
            me, se, mr, sr = emu.mean(), emu.std(), real.mean(), real.std()
            kls.append(math.log(sr / se) + (se ** 2 + (me - mr) ** 2)
                       / (2 * sr ** 2) - 0.5)
        k = float(np.mean(kls))
        succ = []
        for s in SEEDS:
            _, _, a = run_once(Config(seed=s, pf_cv=nu), "PROPOSED-FULL")
            succ.append(a["success"])
        ms, se_ = _agg(succ)
        rows.append([nu, k, float(np.percentile(kls, 95)),
                     1 / 8 + math.sqrt(k / 2), 1 / 8 + math.sqrt(16 * k / 2),
                     ms, se_])
    _write("e13_emulator.csv", rows, ["nu", "kappa_emu", "kappa_emu_p95",
                                      "bound_n1", "bound_n16", "succ",
                                      "succ_se"])


ALL = dict(e0=e0_main, e1=e1_rounds, e2=e2_cdf, e3=e3_load, e4=e4_capacity,
           e5=e5_radio, e6=e6_detector, e7=e7_budget, e8=e8_agent,
           e9=e9_scaling, e10=e10_dynamic, e13=e13_emulator)
