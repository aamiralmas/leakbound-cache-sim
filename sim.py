"""Discrete-event simulator of a multi-tenant edge prefix cache.

One edge site serves language-model prefill for several network slices.
Requests are chains of segments (system prompt, document, template, suffix).
The prefix cache stores one KV node per chain prefix, keyed by
(salt, chain).  Policies differ only in (i) how keys are salted, (ii) which
hits are treated as *foreign*, and (iii) how the first-token release time is
shaped.  Worker occupancy is always the *actual* prefill time: shaping delays
the release of the first token, never the GPU.
"""
from __future__ import annotations

import heapq
import time
import math
import zlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Config, SLICES, SliceProfile

NAMES = [s.name for s in SLICES]
IDX = {n: i for i, n in enumerate(NAMES)}


def rng_for(cfg: Config, tag: str) -> np.random.Generator:
    """Deterministic, process-independent stream (no salted str hash)."""
    return np.random.default_rng([cfg.seed, zlib.crc32(tag.encode())])


# ---------------------------------------------------------------------------
# Segment library
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    sid: int
    length: int
    kind: str          # pub_sys pub_doc unreg priv_sys priv_doc tpl sfx vsys secret cand
    registry: bool     # attested public (in the public-prefix registry)
    tenant: int = -1   # owning tenant for private kinds
    flagged: bool = False  # SELECTIVE baseline: detector says 'sensitive'


class Library:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = rng_for(cfg, "library")
        self.segs: List[Segment] = []
        r = self.rng
        self.pub_sys = [self._new(r.integers(*cfg.sys_len), "pub_sys", True)
                        for _ in range(cfg.n_public_sys)]
        self.pub_doc = [self._new(r.integers(*cfg.doc_len), "pub_doc", True)
                        for _ in range(cfg.n_public_doc)]
        self.unreg = [self._new(r.integers(*cfg.sys_len), "unreg", False)
                      for _ in range(cfg.n_shared_unreg)]
        self.priv_sys, self.priv_doc, self.tpl = [], [], []
        for t in range(len(SLICES)):
            self.priv_sys.append([self._new(r.integers(*cfg.sys_len),
                                            "priv_sys", False, t)
                                  for _ in range(cfg.n_private_sys)])
            self.priv_doc.append([self._new(r.integers(*cfg.doc_len),
                                            "priv_doc", False, t)
                                  for _ in range(cfg.n_private_doc)])
            self.tpl.append([self._new(r.integers(*cfg.tpl_len), "tpl",
                                       False, t)
                             for _ in range(cfg.n_templates)])
        # agentic attestation of cross-tenant unregistered segments
        ar = rng_for(cfg, "agent")
        for s in self.unreg:
            if ar.random() < cfg.agent_recall:
                s.registry = True
        self._zipf_cache: Dict[int, np.ndarray] = {}

    PUBLIC_KINDS = ("pub_sys", "pub_doc", "unreg", "vsys")

    def _new(self, length, kind, registry, tenant=-1) -> Segment:
        s = Segment(len(self.segs), int(length), kind, registry, tenant)
        # content-deterministic sensitivity detector (SafeKV-style baseline)
        u = np.random.default_rng([self.cfg.seed, 7, s.sid]).random()
        if kind in self.PUBLIC_KINDS:
            s.flagged = u < self.cfg.det_fpr
        else:
            s.flagged = u >= self.cfg.det_fnr
        self.segs.append(s)
        return s

    def new_segment(self, length, kind, registry=False, tenant=-1):
        return self._new(length, kind, registry, tenant)

    def zipf_pick(self, rng, items):
        n = len(items)
        p = self._zipf_cache.get(n)
        if p is None:
            w = 1.0 / np.arange(1, n + 1) ** self.cfg.zipf_s
            p = w / w.sum()
            self._zipf_cache[n] = p
        return items[rng.choice(n, p=p)]


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
@dataclass
class Request:
    rid: int
    tenant: int
    t_send: float                   # s
    chain: Tuple[int, ...]          # segment ids
    role: str = "honest"            # honest | victim | probe | warm
    round_id: int = -1
    cand_idx: int = -1
    # filled by the simulator (all ms unless noted)
    ul: float = 0.0
    dl: float = 0.0
    t_edge: float = 0.0             # s
    wait: float = 0.0
    prefill: float = 0.0
    release: float = 0.0
    hit_tokens: int = 0
    foreign_tokens: int = 0
    total_tokens: int = 0
    alpha: float = 0.0
    kl: float = 0.0
    sig: float = 0.0
    pair: tuple = ()

    @property
    def ttft(self) -> float:
        return self.ul + self.wait + self.release + self.dl


def radio_latency(rng, prof: SliceProfile) -> Tuple[float, float]:
    def one(base):
        x = base + prof.jitter_ms * rng.standard_normal()
        if prof.blockage_p > 0 and rng.random() < prof.blockage_p:
            x += rng.exponential(prof.blockage_mean_ms)
        return max(0.2 * base, x)
    return one(prof.ul_base_ms), one(prof.dl_base_ms)


def rate_at(cfg: Config, t: float) -> float:
    r = cfg.agg_rate
    for t0, rt in cfg.rate_profile:
        if t >= t0:
            r = rt
    return r


def honest_workload(cfg: Config, lib: Library, rng) -> List[Request]:
    reqs: List[Request] = []
    rid = 0
    for t, prof in enumerate(SLICES):
        if not cfg.rate_profile:
            lam = cfg.agg_rate * prof.rate_share
            n = rng.poisson(lam * cfg.horizon_s)
            times = np.sort(rng.uniform(0, cfg.horizon_s, n))
        else:
            # time-varying Poisson process by thinning
            rmax = max(r for _, r in cfg.rate_profile)
            lam = rmax * prof.rate_share
            n = rng.poisson(lam * cfg.horizon_s)
            cand = np.sort(rng.uniform(0, cfg.horizon_s, n))
            keep = rng.random(n) < np.array(
                [rate_at(cfg, x) / rmax for x in cand])
            times = cand[keep]
        for ts in times:
            u = rng.random()
            if u < prof.p_shared_unreg:
                sys = lib.zipf_pick(rng, lib.unreg)
            elif u < prof.p_shared_unreg + prof.p_public_sys:
                sys = lib.zipf_pick(rng, lib.pub_sys)
            else:
                sys = lib.zipf_pick(rng, lib.priv_sys[t])
            chain = [sys.sid]
            if rng.random() < cfg.p_doc:
                doc = (lib.zipf_pick(rng, lib.pub_doc)
                       if rng.random() < prof.p_public_doc
                       else lib.zipf_pick(rng, lib.priv_doc[t]))
                chain.append(doc.sid)
            chain.append(lib.zipf_pick(rng, lib.tpl[t]).sid)
            sfx = lib.new_segment(rng.integers(*cfg.sfx_len), "sfx")
            chain.append(sfx.sid)
            reqs.append(Request(rid, t, float(ts), tuple(chain)))
            rid += 1
    return reqs


def attack_workload(cfg: Config, lib: Library, rng, rid0: int):
    """Prompt-leakage probing modeled on published KV-cache timing attacks: after the victim uses a secret template the
    attacker probes K equal-length candidates and ranks them by TTFT.
    Returns requests plus invalidation events (between repeated rounds on the
    same secret, standing in for a long gap that evicts the entries)."""
    if cfg.n_rounds <= 0:
        return [], [], []
    a, v = IDX[cfg.attacker], IDX[cfg.victim]
    reqs, invals, rounds = [], [], []
    rid = rid0
    vsys = lib.new_segment(cfg.victim_sys_len, "vsys", True, v)
    n_secrets = max(1, cfg.n_rounds // cfg.probes_per_secret)
    span = cfg.horizon_s * 0.9 / (n_secrets * cfg.probes_per_secret)
    k = 0
    for s in range(n_secrets):
        L = int(rng.integers(*cfg.secret_len))
        cands = [lib.new_segment(L, "cand", False, -1)
                 for _ in range(cfg.n_candidates)]
        true_idx = int(rng.integers(cfg.n_candidates))
        rounds.append(dict(secret=s, true_idx=true_idx,
                           cand_sids=[c.sid for c in cands]))
        for rep in range(cfg.probes_per_secret):
            t0 = 5.0 + k * span + rng.uniform(0, 0.2 * span)
            k += 1
            sfx = lib.new_segment(80, "sfx")
            reqs.append(Request(rid, v, t0,
                                (vsys.sid, cands[true_idx].sid, sfx.sid),
                                role="victim", round_id=s))
            rid += 1
            t = t0 + 1.5
            # warm-up probe acquires co-ownership of the known system prompt
            junk = lib.new_segment(L, "sfx")
            reqs.append(Request(rid, a, t, (vsys.sid, junk.sid),
                                role="warm", round_id=s))
            rid += 1
            order = rng.permutation(cfg.n_candidates)
            for j, ci in enumerate(order):
                sfx = lib.new_segment(40, "sfx")
                reqs.append(Request(rid, a, t + (j + 1) * cfg.probe_gap_s,
                                    (vsys.sid, cands[ci].sid, sfx.sid),
                                    role="probe", round_id=s,
                                    cand_idx=int(ci)))
                rid += 1
            t_end = t + (cfg.n_candidates + 3) * cfg.probe_gap_s + 2.0
            invals.append((t_end, set(c.sid for c in cands)))
    return reqs, invals, rounds


# ---------------------------------------------------------------------------
# Prefill model and miss-latency emulator
# ---------------------------------------------------------------------------
def prefill_mean(cfg: Config, n_new: int, ctx: int) -> float:
    return (cfg.pf_a_ms + cfg.pf_b_ms * n_new
            + cfg.pf_c_ms * n_new * (ctx + n_new / 2.0))


class MissEmulator:
    """Online least-squares model of prefill time, used to synthesise the
    release time a foreign hit would have had as a miss.  It is fitted only
    on observed prefills, never on the true generator parameters."""

    def __init__(self, cfg: Config, rng):
        self.cfg, self.rng = cfg, rng
        self.XtX = np.zeros((3, 3))
        self.Xty = np.zeros(3)
        self.n = 0
        self.beta = np.array([0.0, 0.1, 0.0])
        # offline profiling: 64 calibration prefills
        for _ in range(64):
            n_new = int(rng.integers(50, 3500))
            ctx = int(rng.integers(0, 2500))
            t = prefill_mean(cfg, n_new, ctx) * rng.lognormal(
                -0.5 * cfg.pf_cv ** 2, cfg.pf_cv)
            self.observe(n_new, ctx, t)
        self._refit()

    @staticmethod
    def _x(n_new, ctx):
        return np.array([1.0, n_new, n_new * (ctx + n_new / 2.0)])

    def observe(self, n_new, ctx, t):
        x = self._x(n_new, ctx)
        self.XtX += np.outer(x, x)
        self.Xty += x * t
        self.n += 1
        if self.n % 200 == 0:
            self._refit()

    def _refit(self):
        self.beta = np.linalg.solve(self.XtX + 1e-6 * np.eye(3), self.Xty)

    def sample(self, n_new, ctx) -> float:
        mu = float(self._x(n_new, ctx) @ self.beta)
        # heteroscedastic draw: the emulator assumes the same relative spread
        rel = self.cfg.pf_cv
        return max(0.0, mu * (1.0 + rel * self.rng.standard_normal()))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
@dataclass
class Node:
    tokens: int
    owners: set
    creator: int
    public: bool


class PrefixCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.used = 0
        self.d: "OrderedDict[tuple, Node]" = OrderedDict()

    def get(self, key):
        return self.d.get(key)

    def touch(self, keys):
        for k in reversed(keys):          # deepest first => ancestors newest
            if k in self.d:
                self.d.move_to_end(k)

    def insert(self, key, node: Node):
        if key in self.d or node.tokens > self.cap:
            return
        while self.used + node.tokens > self.cap and self.d:
            _, old = self.d.popitem(last=False)
            self.used -= old.tokens
        self.d[key] = node
        self.used += node.tokens

    def invalidate(self, sids: set):
        dead = [k for k in self.d if any(s in sids for s in k[1])]
        for k in dead:
            self.used -= self.d.pop(k).tokens


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class Simulator:
    def __init__(self, cfg: Config, policy: str):
        self.cfg, self.policy = cfg, policy
        self.lib = Library(cfg)
        self.rng_radio = rng_for(cfg, "radio")
        self.rng_pf = rng_for(cfg, "prefill")
        self.rng = rng_for(cfg, "policy:" + policy)
        wl_rng = rng_for(cfg, "workload")
        self.reqs = honest_workload(cfg, self.lib, wl_rng)
        atk, self.invals, self.rounds = attack_workload(
            cfg, self.lib, rng_for(cfg, "attack"), len(self.reqs))
        self.reqs += atk
        self.cache = PrefixCache(cfg.cache_tokens)
        self.emu = MissEmulator(cfg, rng_for(cfg, "emulator"))
        self.Z: Dict[Tuple[int, int], float] = defaultdict(float)
        self.Zt: Dict[Tuple[int, int], float] = {}
        self.kl_pair: Dict[Tuple[int, int], float] = defaultdict(float)
        self.busy_ms = 0.0
        self.ctrl_ns: List[float] = []

    # -- helpers ------------------------------------------------------------
    def _is_public_chain(self, chain_prefix) -> bool:
        return all(self.lib.segs[s].registry for s in chain_prefix)

    def _key(self, tenant, chain_prefix):
        p = self.policy
        if p == "ISOLATED":
            salt = tenant + 1
        elif p == "SELECTIVE":
            flagged = any(self.lib.segs[s].flagged for s in chain_prefix)
            salt = tenant + 1 if flagged else 0
        else:
            salt = 0
        return (salt, tuple(chain_prefix))

    def _sigma(self, tenant, t_ms):
        prof = SLICES[tenant]
        return math.sqrt(2 * prof.jitter_floor_ms ** 2
                         + (self.cfg.pf_cv * t_ms) ** 2)

    # -- main loop ------------------------------------------------------------
    def run(self) -> "Simulator":
        cfg, rng = self.cfg, self.rng
        for r in self.reqs:
            r.ul, r.dl = radio_latency(self.rng_radio, SLICES[r.tenant])
            r.t_edge = r.t_send + r.ul / 1000.0
            r.total_tokens = sum(self.lib.segs[s].length for s in r.chain)
        order = sorted(self.reqs, key=lambda q: (q.t_edge, q.rid))
        workers = [0.0] * cfg.n_workers
        heapq.heapify(workers)
        pending: List[Tuple[float, int, Request, list]] = []
        invals = sorted(self.invals, key=lambda e: e[0])
        ii = 0
        for r in order:
            free = heapq.heappop(workers)
            start = max(free, r.t_edge)
            while pending and pending[0][0] <= start:
                _, _, pr, ins = heapq.heappop(pending)
                self._commit(pr, ins)
            while ii < len(invals) and invals[ii][0] <= start:
                self.cache.invalidate(invals[ii][1])
                ii += 1
            r.wait = (start - r.t_edge) * 1000.0
            ins = self._serve(r, start)
            end = start + r.prefill / 1000.0
            heapq.heappush(workers, end)
            heapq.heappush(pending, (end, r.rid, r, ins))
            self.busy_ms += r.prefill
        return self

    def _serve(self, r: Request, start: float):
        cfg, lib = self.cfg, self.lib
        segs = [lib.segs[s] for s in r.chain]
        hit, foreign, first_owner = 0, 0, None
        keys, hit_keys = [], []
        if self.policy != "NOCACHE":
            keys = [self._key(r.tenant, r.chain[: d + 1])
                    for d in range(len(segs))]
            for k in keys:
                node = self.cache.get(k)
                if node is None:
                    break
                hit += node.tokens
                hit_keys.append(k)
                if not node.public and r.tenant not in node.owners:
                    foreign += node.tokens
                    if first_owner is None:
                        first_owner = node.creator
        n_new = r.total_tokens - hit
        actual = prefill_mean(cfg, n_new, hit) * self.rng_pf.lognormal(
            -0.5 * cfg.pf_cv ** 2, cfg.pf_cv)
        r.prefill, r.hit_tokens, r.foreign_tokens = actual, hit, foreign
        release = actual
        if self.policy == "NOISE":
            release = actual + self.rng.uniform(0, cfg.noise_ms)
        elif self.policy.startswith("PROPOSED") and foreign > 0:
            emul = self.emu.sample(n_new + foreign, hit - foreign)
            target = max(actual, emul)
            g = target - actual
            if self.policy == "PROPOSED-FULL":
                alpha = cfg.alpha_fixed
            else:
                t0 = time.perf_counter_ns()
                alpha = self._alpha(r.tenant, first_owner, g, target, start)
                self.ctrl_ns.append(time.perf_counter_ns() - t0)
            sig = self._sigma(r.tenant, target)
            kl = ((1 - alpha) * g) ** 2 / (2 * sig ** 2) if g > 0 else 0.0
            r.alpha, r.kl, r.sig = alpha, kl, sig
            r.pair = (r.tenant, first_owner)
            self.kl_pair[(r.tenant, first_owner)] += kl
            release = actual + alpha * g
        r.release = release
        self.cache.touch(hit_keys)
        if self.policy != "NOCACHE":
            self.emu.observe(n_new, hit, actual)
        return list(zip(keys, segs))

    def _alpha(self, req, owner, g, target, now) -> float:
        """Closed-form drift-plus-penalty release shaping (Eq. alpha-star)."""
        cfg = self.cfg
        if g <= 0:
            return 0.0
        pair = (req, owner)
        last = self.Zt.get(pair, now)
        Z = max(self.Z[pair] - cfg.eps_rate * (now - last), 0.0)
        sig = self._sigma(req, target)
        a_min = max(0.0, 1.0 - sig * math.sqrt(2 * cfg.kappa_max) / g)
        if Z <= 0:
            a = a_min
        else:
            a = 1.0 - cfg.V * sig ** 2 / (Z * g)
        a = min(1.0, max(a_min, a))
        kl = ((1 - a) * g) ** 2 / (2 * sig ** 2)
        self.Z[pair] = Z + kl
        self.Zt[pair] = now
        return a

    def _commit(self, r: Request, ins):
        """At prefill completion: insert new nodes, record co-ownership."""
        if self.policy == "NOCACHE":
            return
        for d, (k, seg) in enumerate(ins):
            node = self.cache.get(k)
            if node is None:
                pub = self._is_public_chain(r.chain[: d + 1])
                self.cache.insert(k, Node(seg.length, {r.tenant}, r.tenant,
                                          pub))
            else:
                node.owners.add(r.tenant)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def attack_metrics(sim: Simulator) -> Dict[str, float]:
    cfg = sim.cfg
    by_secret: Dict[int, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in sim.reqs:
        if r.role == "probe":
            by_secret[r.round_id][r.cand_idx].append(r.ttft)
    succ, aucs = [], []
    for info in sim.rounds:
        obs = by_secret.get(info["secret"])
        if not obs:
            continue
        means = {c: float(np.mean(v)) for c, v in obs.items()}
        guess = min(means, key=means.get)
        succ.append(1.0 if guess == info["true_idx"] else 0.0)
        tv = obs.get(info["true_idx"], [])
        fv = [x for c, v in obs.items() if c != info["true_idx"] for x in v]
        if tv and fv:
            # AUC of "smaller TTFT => cached"
            wins = sum((t < f) + 0.5 * (t == f) for t in tv for f in fv)
            aucs.append(wins / (len(tv) * len(fv)))
    # per-probe threshold detector: 'cached' if TTFT < round median - delta,
    # delta = half the prefill saving the attacker expects for the secret
    # length (public cost model), i.e. the midpoint between hit and miss
    tp = fp = fn = tn = 0
    by_rid: Dict[int, List[Request]] = defaultdict(list)
    for r in sim.reqs:
        if r.role == "probe":
            by_rid[r.round_id].append(r)
    rounds_probes: Dict[tuple, List[Request]] = {}
    for rid, P in by_rid.items():          # split repeated rounds by time gap
        P.sort(key=lambda q: q.t_send)
        k, last = 0, None
        for p in P:
            if last is not None and p.t_send - last > 1.0:
                k += 1
            rounds_probes.setdefault((rid, k), []).append(p)
            last = p.t_send
    truth = {info["secret"]: info["true_idx"] for info in sim.rounds}
    for key, P in rounds_probes.items():
        if len(P) < 3:
            continue
        med = float(np.median([p.ttft for p in P]))
        L = sim.lib.segs[P[0].chain[1]].length
        delta = 0.5 * cfg.pf_b_ms * L
        for p in P:
            pred = p.ttft < med - delta
            real = p.cand_idx == truth[p.round_id]
            tp += pred and real
            fp += pred and not real
            fn += (not pred) and real
            tn += (not pred) and not real
    a, v = IDX[cfg.attacker], IDX[cfg.victim]
    kl_av = sum(r.kl for r in sim.reqs
                if r.role == "probe" and r.foreign_tokens > 0)
    kl_max_secret = 0.0
    per = defaultdict(float)
    for r in sim.reqs:
        if r.role == "probe":
            per[r.round_id] += r.kl
    if per:
        kl_max_secret = max(per.values())
    K = cfg.n_candidates
    bound = min(1.0, 1.0 / K + float(np.mean(
        [math.sqrt(0.5 * x) for x in per.values()]))) if per else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, kl_total=kl_av, kl_secret_max=kl_max_secret, bound=bound,
                success=float(np.mean(succ)) if succ else float("nan"),
                auc=float(np.mean(aucs)) if aucs else float("nan"),
                n_secrets=len(succ))


def honest_metrics(sim: Simulator) -> Dict[str, float]:
    H = [r for r in sim.reqs if r.role == "honest"]
    tt = np.array([r.ttft for r in H])
    tok = sum(r.total_tokens for r in H)
    hit = sum(r.hit_tokens for r in H)
    fr = [r for r in H if r.foreign_tokens > 0]
    shaping = np.array([r.release - r.prefill for r in H])
    per_slice = {}
    for i, n in enumerate(NAMES):
        x = np.array([r.ttft for r in H if r.tenant == i])
        per_slice[n] = (float(x.mean()), float(np.percentile(x, 95)))
    util = sim.busy_ms / (sim.cfg.n_workers * sim.cfg.horizon_s * 1000.0)
    return dict(ttft_mean=float(tt.mean()), ttft_p50=float(np.median(tt)),
                ttft_p95=float(np.percentile(tt, 95)),
                ttft_p99=float(np.percentile(tt, 99)),
                hit_ratio=hit / tok, util=util,
                foreign_frac=len(fr) / len(H),
                shaping_mean=float(shaping.mean()),
                slo_frac=float(np.mean(tt <= sim.cfg.slo_ms)),
                n=len(H), per_slice=per_slice, ttft=tt,
                tsend=np.array([r.t_send for r in H]))


def run_once(cfg: Config, policy: str):
    sim = Simulator(cfg, policy).run()
    return sim, honest_metrics(sim), attack_metrics(sim)
