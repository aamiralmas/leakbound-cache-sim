"""Central configuration for the leakage-bounded prefix-caching simulator.

Every tunable constant lives here.  Nothing in this file is a secret; the
seeds are public test vectors so that every figure in the paper can be
regenerated bit-for-bit.
"""
from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

SEED = 20260923


@dataclass(frozen=True)
class SliceProfile:
    """Radio-side profile of one network slice (one tenant)."""
    name: str
    kind: str                 # 'urllc' | 'embb' | 'mmwave' | 'mc'
    rate_share: float         # share of the aggregate request rate
    ul_base_ms: float         # one-way uplink base latency
    dl_base_ms: float         # one-way downlink base latency
    jitter_ms: float          # std. dev. of the Gaussian jitter component
    jitter_floor_ms: float    # configured jitter floor the controller trusts
    blockage_p: float = 0.0   # probability of a blockage event (mmWave)
    blockage_mean_ms: float = 0.0
    p_public_sys: float = 0.5
    p_public_doc: float = 0.4
    p_shared_unreg: float = 0.15  # prob. a request uses a cross-tenant but
                                  # unregistered segment (open-source prompt)


SLICES: Tuple[SliceProfile, ...] = (
    SliceProfile("URLLC-Industrial", "urllc", 0.15, 0.5, 0.5, 0.15, 0.10),
    SliceProfile("eMBB-Consumer",    "embb",  0.30, 6.0, 4.0, 2.5, 1.5),
    SliceProfile("mmWave-XR",        "mmwave", 0.15, 2.0, 1.5, 1.2, 0.8,
                 blockage_p=0.08, blockage_mean_ms=25.0),
    SliceProfile("Enterprise",       "embb",  0.20, 5.0, 3.5, 2.0, 1.2),
    SliceProfile("e-Health",         "embb",  0.10, 5.0, 3.5, 2.0, 1.2),
    SliceProfile("Public-Safety",    "mc",    0.10, 1.5, 1.0, 0.5, 0.3),
)


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    horizon_s: float = 600.0          # simulated seconds per run
    agg_rate: float = 22.0            # aggregate honest requests / s
    # optional piecewise-constant aggregate rate: ((t0, rate0), (t1, rate1), ...)
    # empty = constant agg_rate (default; keeps all baseline streams unchanged)
    rate_profile: Tuple[Tuple[float, float], ...] = ()
    slo_ms: float = 300.0             # TTFT service-level objective for goodput
    n_workers: int = 4                # prefill workers (disaggregated prefill)
    cache_tokens: int = 300_000       # prefix-cache capacity in tokens

    # --- prefill time model  t = a + b n + c n (ctx + n/2), ms -------------
    pf_a_ms: float = 8.0
    pf_b_ms: float = 0.085
    pf_c_ms: float = 1.6e-6
    pf_cv: float = 0.04               # multiplicative log-normal noise

    # --- segment library ----------------------------------------------------
    n_public_sys: int = 6
    n_public_doc: int = 40
    n_shared_unreg: int = 12          # cross-tenant, not in public registry
    n_private_sys: int = 3
    n_private_doc: int = 50
    n_templates: int = 25
    zipf_s: float = 1.05
    sys_len: Tuple[int, int] = (500, 1200)
    doc_len: Tuple[int, int] = (700, 1800)
    tpl_len: Tuple[int, int] = (150, 450)
    sfx_len: Tuple[int, int] = (40, 160)
    p_doc: float = 0.8

    # --- policy knobs -------------------------------------------------------
    noise_ms: float = 60.0            # SHARED+NOISE: U[0, noise_ms]
    det_fnr: float = 0.10             # SELECTIVE: sensitivity-detector
    det_fpr: float = 0.10             #   false-negative / false-positive
    agent_recall: float = 0.9         # fraction of shared-unregistered
                                      # segments the agent gets attested
    kappa_max: float = 0.02           # per-release KL cap (nats)
    eps_rate: float = 0.002           # per-pair KL budget accrual (nats/s)
    V: float = 2.0e-4                 # drift-plus-penalty trade-off
    alpha_fixed: float = 1.0          # used by the FULL variant

    # --- attacker -----------------------------------------------------------
    attacker: str = "URLLC-Industrial"
    victim: str = "Enterprise"
    n_candidates: int = 8
    secret_len: Tuple[int, int] = (300, 500)
    victim_sys_len: int = 900
    n_rounds: int = 120               # independent attack rounds per run
    probes_per_secret: int = 1        # rounds aggregated on the same secret
    probe_gap_s: float = 0.12

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)


POLICIES = ("NOCACHE", "SHARED", "NOISE", "ISOLATED", "SELECTIVE",
            "PROPOSED-FULL", "PROPOSED")

POLICY_LABEL: Dict[str, str] = {
    "NOCACHE": "No cache",
    "SHARED": "Shared (no defense)",
    "NOISE": "Shared + random delay",
    "ISOLATED": "Per-tenant isolation",
    "SELECTIVE": "Selective sharing",
    "PROPOSED-FULL": "Proposed ($\\alpha{=}1$)",
    "PROPOSED": "Proposed (budgeted)",
}
