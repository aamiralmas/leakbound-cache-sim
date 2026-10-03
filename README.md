# Leakage-bounded prefix caching: simulator

Reproduces every figure and table in the evaluation section of
*"Leakage-Bounded Prefix Caching for Multi-Tenant Edge Language-Model
Inference in 6G Network Slices"*.

This is a discrete-event model of one edge site. Six network slices share a
language-model prefix (KV) cache, a prober tries to identify another tenant's
secret prompt template from its own time-to-first-token (TTFT), and seven
cache policies are compared. **No language model and no GPU are involved.**
The prefill cost coefficients are representative values, not measurements
(see the paper, Sec. VI-A and VII-B).

## Run

```
pip install -r requirements.txt      # numpy only
python test_sim.py                   # 9 property tests (about 1 min)
python run_all.py                    # all experiments (about 25 min, 2 cores)
python run_all.py e0 e5              # selected experiments
python make_figures.py               # CSV -> results/fig_*.tex, tab_main_rows.tex
```

## Files

| File | Contents |
|---|---|
| `config.py` | every constant: slices, workload, prefill model, policy knobs, attacker |
| `sim.py` | workload, prefix cache with provenance, miss-latency emulator, policies, metrics |
| `experiments.py` | drivers e0 to e9, one CSV each in `results/` |
| `make_figures.py` | turns CSVs into the pgfplots fragments used in the paper |
| `test_sim.py` | checks the claims the paper makes about the model |
| `run_all.py` | single entry point |

## Experiment to figure/table map

| Driver | Paper |
|---|---|
| e0 | Table (main results) |
| e1 | Fig. rounds (success vs. aggregated rounds) |
| e2 | Fig. TTFT CDF |
| e3 | Fig. load sweep |
| e4 | Fig. capacity sweep |
| e5 | Fig. random delay vs. attacker slice |
| e6 | Fig. detector false-negative rate |
| e7 | Fig. budgeted shaping |
| e8 | Fig. agent attestation recall |
| e9 | complexity table (micro-benchmarks) |
| e10 | Fig. load burst (time series, 18 -> 30 -> 18 req/s) |
| e3 (also writes e11) | Fig. goodput under a 300-ms TTFT objective |
| e0 (also writes e12) | Table: per-probe confusion counts of a threshold detector |
| e13 | Table: emulator fidelity (per-release KL) vs. prefill variability |

## Policies

`NOCACHE`, `SHARED` (undefended), `NOISE` (uniform random delay),
`ISOLATED` (per-tenant key salt), `SELECTIVE` (share only chains a
sensitivity detector does not flag), `PROPOSED-FULL` (provenance-shaped
release, alpha = 1), `PROPOSED` (budgeted drift-plus-penalty shaping).

## Reproducibility notes

* Seeds are `20260923 + i`, i = 0..9. Random streams are derived with
  `zlib.crc32(tag)`, never Python's salted `hash()`, so results are identical
  across processes (checked by `test_determinism_across_processes`).
* Arrivals, radio, prefill noise and policy randomness use separate streams,
  so all policies see the same sample path. This is what lets
  `test_sample_path_identity` check Theorem 2 request by request.
* The seeds are public test vectors, not secrets.

## License

MIT (see `LICENSE`).
