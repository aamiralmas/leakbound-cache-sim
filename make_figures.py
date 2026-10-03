"""CSV -> pgfplots figure environments (results/fig_*.tex) and table rows.

The manuscript inlines these fragments verbatim, so every plotted number is
traceable to a CSV produced by experiments.py.  Never hand-edit the output.
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

STYLE = {
    "NOCACHE": "black, densely dotted, mark=none, thick",
    "SHARED": "gray, dashed, mark=o, mark size=1.6pt, thick",
    "NOISE": "orange!90!black, mark=triangle, mark size=1.8pt, thick",
    "ISOLATED": "red!80!black, mark=square, mark size=1.5pt, thick",
    "SELECTIVE": "violet, mark=diamond, mark size=1.9pt, thick",
    "PROPOSED-FULL": "blue!80!black, mark=*, mark size=1.5pt, very thick",
    "PROPOSED": "cyan!70!black, dashed, mark=x, mark size=2pt, thick",
}
LABEL = {
    "NOCACHE": "No cache", "SHARED": "Shared (undefended)",
    "NOISE": "Shared + random delay", "ISOLATED": "Per-tenant isolation",
    "SELECTIVE": "Selective sharing", "PROPOSED-FULL": "Proposed ($\\alpha{=}1$)",
    "PROPOSED": "Proposed (budgeted)",
}


def rows(name):
    with open(os.path.join(RES, name)) as f:
        r = list(csv.reader(f))
    return r[0], r[1:]


def col(hdr, data, key, cast=float):
    i = hdr.index(key)
    return [cast(x[i]) for x in data]


def coords(xs, ys, es=None):
    if es is None:
        return " ".join(f"({x:g},{y:.4g})" for x, y in zip(xs, ys))
    return " ".join(f"({x:g},{y:.4g}) +- (0,{e:.3g})"
                    for x, y, e in zip(xs, ys, es))


def axis(opts, body):
    return ("\\begin{tikzpicture}\n\\begin{axis}[" + opts + "]\n" + body +
            "\\end{axis}\n\\end{tikzpicture}\n")


COMMON = ("width=0.96\\columnwidth, height=4.6cm, grid=major, "
          "grid style={gray!25}, tick label style={font=\\scriptsize}, "
          "label style={font=\\footnotesize}, "
          "legend style={font=\\scriptsize, fill=white, fill opacity=0.9, "
          "text opacity=1, draw=gray!50}, legend cell align=left")


def fig(label, caption, tikz):
    return ("\\begin{figure}[!t]\n\\centering\n" + tikz +
            "\\caption{" + caption + "}\n\\label{" + label + "}\n"
            "\\end{figure}\n")


def f_rounds():
    h, d = rows("e1_rounds.csv")
    n = col(h, d, "n")
    body = ""
    for p in ("SHARED", "NOISE", "SELECTIVE", "ISOLATED", "PROPOSED",
              "PROPOSED-FULL"):
        body += (f"\\addplot+[{STYLE[p]}, error bars/.cd, y dir=both, "
                 f"y explicit] coordinates {{{coords(n, col(h, d, p), col(h, d, p + '_se'))}}};\n"
                 f"\\addlegendentry{{{LABEL[p]}}}\n")
    body += (f"\\addplot[black, dashdotted, thick] coordinates "
             f"{{{coords(n, col(h, d, 'bound'))}}};\n"
             "\\addlegendentry{Thm.~\\ref{thm:ident} bound (budgeted)}\n")
    body += "\\addplot[black!50, thin, domain=1:16] {0.125};\n"
    opts = (COMMON + ", xmode=log, log basis x=2, xtick={1,2,4,8,16}, "
            "xticklabels={1,2,4,8,16}, ymin=0, ymax=1.05, "
            "xlabel={Rounds aggregated on the same secret $n$}, "
            "ylabel={Identification success}, legend pos=outer north east, "
            "legend columns=2, legend style={at={(0.5,1.03)}, anchor=south}")
    return fig("fig:rounds",
               "Attacker's secret-identification success versus the number "
               "of probing rounds aggregated on one secret ($K{=}8$ "
               "candidates, chance $=0.125$ shown as the thin grey line; "
               "10 seeds, $\\pm1$ s.e.). The dash-dotted curve is the "
               "Theorem~\\ref{thm:ident} bound evaluated on the leakage the "
               "budgeted controller actually spent.",
               axis(opts, body))


def f_cdf():
    h, d = rows("e2_cdf.csv")
    q = col(h, d, "q")
    body = ""
    for p in ("SHARED", "NOISE", "ISOLATED", "SELECTIVE", "PROPOSED-FULL"):
        xs = col(h, d, p)
        body += (f"\\addplot[{STYLE[p].replace('mark=', 'mark repeat=10, mark=')}] "
                 f"coordinates {{{coords(xs, q)}}};\n"
                 f"\\addlegendentry{{{LABEL[p]}}}\n")
    opts = (COMMON + ", xmin=0, xmax=520, ymin=0, ymax=1.0, "
            "xlabel={Honest time-to-first-token (ms)}, ylabel={CDF}, "
            "legend pos=south east")
    return fig("fig:cdf",
               "Empirical CDF of honest-request time-to-first-token at the "
               "nominal load of 22~req/s (seed 20260923). The proposed curve "
               "overlays the undefended shared cache.",
               axis(opts, body))


def f_load():
    h, d = rows("e3_load.csv")
    L = col(h, d, "load")
    body = ""
    for p in ("NOCACHE", "SHARED", "NOISE", "ISOLATED", "SELECTIVE",
              "PROPOSED-FULL"):
        body += (f"\\addplot[{STYLE[p]}] coordinates "
                 f"{{{coords(L, col(h, d, p + '_mean'))}}};\n"
                 f"\\addlegendentry{{{LABEL[p]}}}\n")
    opts = (COMMON + ", ymode=log, ymin=90, ymax=3e5, xmin=9, xmax=35, "
            "xlabel={Aggregate request rate $\\lambda$ (req/s)}, "
            "ylabel={Mean TTFT (ms)}, legend pos=north west")
    return fig("fig:load",
               "Mean honest TTFT versus aggregate load (5 seeds). Per-tenant "
               "isolation saturates near 30~req/s; the proposed scheme "
               "inherits the stability region of the shared cache.",
               axis(opts, body))


def f_capacity():
    h, d = rows("e4_capacity.csv")
    c = col(h, d, "cap_k")
    body = ""
    for p in ("SHARED", "ISOLATED", "SELECTIVE", "PROPOSED"):
        st = STYLE["PROPOSED-FULL"] if p == "PROPOSED" else STYLE[p]
        lab = "Proposed" if p == "PROPOSED" else LABEL[p]
        body += (f"\\addplot[{st}] coordinates "
                 f"{{{coords(c, col(h, d, p + '_hit'))}}};\n"
                 f"\\addlegendentry{{{lab}}}\n")
    opts = (COMMON + ", xlabel={Prefix-cache capacity (thousand tokens)}, "
            "ylabel={Token hit ratio}, ymin=0.1, ymax=0.6, "
            "legend pos=south east")
    return fig("fig:capacity",
               "Token-weighted prefix hit ratio versus cache capacity "
               "(5 seeds). Isolation pays compulsory misses on every "
               "cross-tenant public prefix.",
               axis(opts, body))


def f_radio():
    h, d = rows("e5_radio.csv")
    dd = [x for x in d if x[0] != "proposed"]
    prop = [x for x in d if x[0] == "proposed"][0]
    w = [float(x[0]) for x in dd]
    names = ("URLLC-Industrial", "eMBB-Consumer", "mmWave-XR")
    st = ("red!80!black, mark=square", "orange!90!black, mark=triangle",
          "teal, mark=o")
    lab = ("URLLC attacker", "eMBB attacker", "mmWave attacker")
    body = ""
    for nm, s, lb in zip(names, st, lab):
        body += (f"\\addplot+[{s}, thick, error bars/.cd, y dir=both, "
                 f"y explicit] coordinates {{{coords(w, col(h, dd, nm), col(h, dd, nm + '_se'))}}};\n"
                 f"\\addlegendentry{{{lb}}}\n")
    pv = float(prop[h.index("URLLC-Industrial")])
    body += (f"\\addplot[blue!80!black, very thick, domain=0:120] {{{pv:.4f}}};\n"
             "\\addlegendentry{Proposed, URLLC attacker}\n")
    opts = (COMMON + ", xlabel={Random delay width $w$ (ms), "
            "added to every response}, ylabel={Identification success}, "
            "ymin=0, ymax=0.9, legend style={at={(0.03,0.2)}, anchor=south west}")
    return fig("fig:radio",
               "Naive random-delay padding against attackers on three slice "
               "types (10 seeds). The attacker's air interface barely "
               "matters; the mean honest TTFT cost of padding is $w/2$.",
               axis(opts, body))


def f_detector():
    h, d = rows("e6_detector.csv")
    f = col(h, d, "fnr")
    body = (f"\\addplot+[{STYLE['SELECTIVE']}, error bars/.cd, y dir=both, "
            f"y explicit] coordinates {{{coords(f, col(h, d, 'sel'), col(h, d, 'sel_se'))}}};\n"
            "\\addlegendentry{Selective sharing (detector)}\n"
            f"\\addplot+[{STYLE['PROPOSED-FULL']}, error bars/.cd, y dir=both, "
            f"y explicit] coordinates {{{coords(f, col(h, d, 'prop'), col(h, d, 'prop_se'))}}};\n"
            "\\addlegendentry{Proposed (detector-free)}\n"
            "\\addplot[black!50, thin, domain=0:0.3] {0.125};\n")
    opts = (COMMON + ", xlabel={Sensitivity-detector false-negative rate}, "
            "ylabel={Identification success}, ymin=0, ymax=0.4, "
            "xtick={0,0.05,0.1,0.15,0.2,0.3}, "
            "xticklabels={0,0.05,0.10,0.15,0.20,0.30}, legend pos=north west")
    return fig("fig:detector",
               "Selective (detector-gated) sharing inherits its detector's "
               "false negatives; the proposed scheme does not use a "
               "sensitivity detector (10 seeds).",
               axis(opts, body))


def f_budget():
    h, d = rows("e7_budget.csv")
    dd = [x for x in d if x[0] != "full"]
    kl = col(h, dd, "kl")
    body = (f"\\addplot+[{STYLE['PROPOSED']}, error bars/.cd, y dir=both, "
            f"y explicit] coordinates {{{coords(kl, col(h, dd, 'succ'), col(h, dd, 'succ_se'))}}};\n"
            "\\addlegendentry{Measured success}\n"
            f"\\addplot[black, dashdotted, thick] coordinates "
            f"{{{coords(kl, col(h, dd, 'bound'))}}};\n"
            "\\addlegendentry{Thm.~\\ref{thm:ident} bound}\n"
            "\\addplot[black!50, thin, domain=0:6.2] {0.125};\n")
    opts = (COMMON + ", xlabel={Cumulative KL spent on the attacker pair "
            "(nats per run)}, ylabel={Identification success}, ymin=0.05, "
            "ymax=0.32, legend pos=north west")
    return fig("fig:budget",
               "Budgeted shaping: measured identification success and the "
               "Theorem~\\ref{thm:ident} bound versus the leakage the "
               "controller spent, swept through $\\bar\\epsilon\\in"
               "[0,0.02]$~nats/s (10 seeds, $\\kappa_{\\max}{=}0.05$).",
               axis(opts, body))


def f_agent():
    h, d = rows("e8_agent.csv")
    r = col(h, d, "recall")
    body = (f"\\addplot[{STYLE['PROPOSED-FULL']}] coordinates "
            f"{{{coords(r, col(h, d, 'full_shape'))}}};\n"
            "\\addlegendentry{Mean shaping delay (ms)}\n"
            f"\\addplot[{STYLE['PROPOSED']}] coordinates "
            f"{{{coords(r, [100 * x for x in col(h, d, 'full_foreign')])}}};\n"
            "\\addlegendentry{Foreign-hit requests (\\%)}\n")
    opts = (COMMON + ", xlabel={Agent attestation recall $\\rho$}, "
            "ylabel={Overhead}, ymin=0, ymax=3.6, legend pos=north east")
    return fig("fig:agent",
               "Effect of the agent's attestation recall on honest traffic "
               "(24 cross-tenant unregistered segments, 5 seeds). Recall "
               "moves efficiency only; identification success stays at "
               "chance (Table~\\ref{tab:main} and Sec.~\\ref{sec:res-agent}).",
               axis(opts, body))


def table_main():
    h, d = rows("e0_main.csv")
    out = []
    names = {"NOCACHE": "No cache", "SHARED": "Shared (undefended)",
             "NOISE": "Shared + random delay", "ISOLATED": "Per-tenant isolation",
             "SELECTIVE": "Selective sharing",
             "PROPOSED-FULL": "\\textbf{Proposed} ($\\alpha{=}1$)",
             "PROPOSED": "Proposed (budgeted)"}
    for x in d:
        g = dict(zip(h, x))
        p = g["policy"]
        f = lambda k, n=1: f"{float(g[k]):.{n}f}"
        if p == "NOCACHE":
            out.append(f"{names[p]} & \\multicolumn{{3}}{{c}}{{unstable "
                       f"($\\rho={float(g['util']):.2f}$)}} & 0.000 & "
                       f"{f('util', 2)} & -- & -- & -- \\\\")
            continue
        out.append(
            f"{names[p]} & {f('ttft_mean')} & {f('ttft_p95')} & "
            f"{f('ttft_p99')} & {f('hit', 3)} & {f('util', 2)} & "
            f"{f('shape', 2)} & {f('succ', 3)}$\\,\\pm\\,${f('succ_se', 3)} & "
            f"{f('auc', 3)} \\\\")
    return "\n".join(out) + "\n"


def f_dynamic():
    h, d = rows("e10_dynamic.csv")
    t = col(h, d, "t")
    body = ""
    for p in ("SHARED", "NOISE", "ISOLATED", "SELECTIVE", "PROPOSED-FULL"):
        st = STYLE[p].replace("mark=", "mark repeat=3, mark=")
        body += (f"\\addplot[{st}] coordinates {{{coords(t, col(h, d, p))}}};\n"
                 f"\\addlegendentry{{{LABEL[p]}}}\n")
    body += ("\\draw[black!40, dashed] (axis cs:200,90) -- (axis cs:200,4000);\n"
             "\\draw[black!40, dashed] (axis cs:400,90) -- (axis cs:400,4000);\n"
             "\\node[font=\\scriptsize, anchor=south] at (axis cs:300,2600) {30 req/s};\n")
    opts = (COMMON + ", ymode=log, ymin=100, ymax=4000, xmin=0, xmax=600, "
            "xlabel={Time (s); aggregate rate 18 $\\to$ 30 $\\to$ 18 req/s}, "
            "ylabel={Mean TTFT, 20-s window (ms)}, legend columns=3, "
            "legend style={font=\\tiny, at={(0.5,1.03)}, anchor=south}")
    return fig("fig:dynamic",
               "Response to a load burst (18~req/s, 30~req/s between 200 and "
               "400~s, then 18~req/s; 5 seeds). Per-tenant isolation builds a "
               "backlog that takes about 40~s to drain; the proposed scheme "
               "tracks the undefended cache.",
               axis(opts, body))


def f_goodput():
    h, d = rows("e11_goodput.csv")
    L = col(h, d, "load")
    body = ""
    for p in ("NOCACHE", "SHARED", "NOISE", "ISOLATED", "SELECTIVE",
              "PROPOSED-FULL"):
        body += (f"\\addplot[{STYLE[p]}] coordinates "
                 f"{{{coords(L, col(h, d, p + '_goodput'))}}};\n"
                 f"\\addlegendentry{{{LABEL[p]}}}\n")
    body += "\\addplot[black!40, thin, domain=9:35] {x};\n"
    opts = (COMMON + ", xmin=9, xmax=35, ymin=0, ymax=27, "
            "xlabel={Offered load $\\lambda$ (req/s)}, "
            "ylabel={Goodput (req/s with TTFT $\\le$ 300 ms)}, "
            "legend columns=3, "
            "legend style={font=\\tiny, at={(0.5,1.03)}, anchor=south}")
    return fig("fig:goodput",
               "Goodput, the rate of honest requests served within a 300-ms "
               "TTFT objective, versus offered load (5 seeds; the thin grey "
               "line is $y=\\lambda$).",
               axis(opts, body))


def table_confusion():
    h, d = rows("e12_confusion.csv")
    names = {"SHARED": "Shared (undefended)", "NOISE": "Shared + random delay",
             "ISOLATED": "Per-tenant isolation", "SELECTIVE": "Selective sharing",
             "PROPOSED-FULL": "\\textbf{Proposed} ($\\alpha{=}1$)"}
    out = []
    for x in d:
        g = dict(zip(h, x))
        if g["policy"] not in names:
            continue
        out.append(f"{names[g['policy']]} & {g['tp']} & {g['fp']} & {g['fn']} & "
                   f"{g['tn']} & {float(g['tpr']):.3f} & {float(g['fpr']):.3f} & "
                   f"{float(g['precision']):.3f} \\\\")
    return "\n".join(out) + "\n"


def table_emulator():
    h, d = rows("e13_emulator.csv")
    out = []
    for x in d:
        g = dict(zip(h, x))
        out.append(f"{float(g['nu']):.2f} & {float(g['kappa_emu']):.4f} & "
                   f"{float(g['kappa_emu_p95']):.4f} & {float(g['bound_n1']):.3f} & "
                   f"{float(g['bound_n16']):.3f} & {float(g['succ']):.3f}$\\,\\pm\\,$"
                   f"{float(g['succ_se']):.3f} \\\\")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    figs = dict(rounds=f_rounds, cdf=f_cdf, load=f_load, capacity=f_capacity,
                radio=f_radio, detector=f_detector, budget=f_budget,
                agent=f_agent, dynamic=f_dynamic, goodput=f_goodput)
    for k, fn in figs.items():
        try:
            txt = fn()
        except FileNotFoundError as e:
            print("skip", k, e)
            continue
        with open(os.path.join(RES, f"fig_{k}.tex"), "w") as f:
            f.write(txt)
    with open(os.path.join(RES, "tab_main_rows.tex"), "w") as f:
        f.write(table_main())
    with open(os.path.join(RES, "tab_confusion_rows.tex"), "w") as f:
        f.write(table_confusion())
    with open(os.path.join(RES, "tab_emulator_rows.tex"), "w") as f:
        f.write(table_emulator())
    print("figures written")
