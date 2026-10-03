"""Property tests: each one checks a claim the paper makes about the model."""
import math
import subprocess
import sys

import numpy as np

from config import Config, SLICES
from sim import Simulator, run_once, prefill_mean, IDX

CFG = Config(horizon_s=240.0, n_rounds=40)


def test_determinism_across_processes():
    code = ("from config import Config;from sim import run_once;"
            "_,h,a=run_once(Config(horizon_s=120.0,n_rounds=10),'PROPOSED');"
            "print(round(h['ttft_mean'],9),a['success'])")
    outs = {subprocess.check_output([sys.executable, "-c", code]).strip()
            for _ in range(2)}
    assert len(outs) == 1, outs


def test_sample_path_identity():
    """Theorem 3: shaping never changes worker occupancy, so every request's
    wait and cache hits are identical to the undefended shared cache and the
    TTFT differs exactly by the shaping delay."""
    a = Simulator(CFG, "SHARED").run()
    b = Simulator(CFG, "PROPOSED-FULL").run()
    for x, y in zip(sorted(a.reqs, key=lambda r: r.rid),
                    sorted(b.reqs, key=lambda r: r.rid)):
        assert abs(x.wait - y.wait) < 1e-9
        assert x.hit_tokens == y.hit_tokens
        assert abs((y.ttft - x.ttft) - (y.release - y.prefill)) < 1e-9


def test_isolation_has_no_cross_tenant_hits():
    s = Simulator(CFG, "ISOLATED").run()
    assert all(r.foreign_tokens == 0 for r in s.reqs)


def test_nocache():
    _, h, _ = run_once(CFG, "NOCACHE")
    assert h["hit_ratio"] == 0.0


def test_kappa_cap():
    s = Simulator(CFG, "PROPOSED").run()
    assert max(r.kl for r in s.reqs) <= CFG.kappa_max + 1e-9


def test_budget_sample_path_bound():
    """Theorem 4: per ordered pair, sum(kl) <= eps*T + Z_N where
    Z_N^3 <= (c^(1/3) + kappa_max)^3 + 7 c N,  c = V^2 sigma_max^2 / 2."""
    s = Simulator(CFG, "PROPOSED").run()
    by = {}
    for r in s.reqs:
        if r.pair:
            n, sm, tot = by.get(r.pair, (0, 0.0, 0.0))
            by[r.pair] = (n + 1, max(sm, r.sig), tot + r.kl)
    assert by
    for pair, (n, sm, tot) in by.items():
        c = CFG.V ** 2 * sm ** 2 / 2
        zn = ((c ** (1 / 3) + CFG.kappa_max) ** 3 + 7 * c * n) ** (1 / 3)
        assert tot <= CFG.eps_rate * CFG.horizon_s + zn + 1e-12, (pair, tot)


def test_full_shaping_spends_less_than_budgeted():
    f = Simulator(CFG, "PROPOSED-FULL").run()
    b = Simulator(CFG, "PROPOSED").run()
    assert sum(f.kl_pair.values()) <= sum(b.kl_pair.values()) + 1e-12


def test_emulator_calibrated():
    s = Simulator(CFG, "SHARED").run()
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(400):
        n, c = int(rng.integers(100, 3000)), int(rng.integers(0, 2000))
        m = np.mean([s.emu.sample(n, c) for _ in range(50)])
        errs.append(abs(m - prefill_mean(CFG, n, c)) / prefill_mean(CFG, n, c))
    assert np.mean(errs) < 0.05, np.mean(errs)


def test_attack_works_without_defense():
    _, _, a = run_once(CFG, "SHARED")
    assert a["success"] > 0.5


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = 0
    for t in tests:
        try:
            t()
            ok += 1
            print("PASS", t.__name__)
        except AssertionError as e:
            print("FAIL", t.__name__, e)
    print(f"{ok}/{len(tests)} passed")
