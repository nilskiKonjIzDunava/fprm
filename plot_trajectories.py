"""Interactive plots from the CSVs written by `inference.py ... +trace_out=<prefix>`.

    python plot_trajectories.py results/sudoku                       # 30 evenly spaced sequences
    python plot_trajectories.py results/sudoku --seqs 3 17 250       # chosen sequences
    python plot_trajectories.py results/sudoku --only wrong --max-seqs 15

Reads <prefix>_trace.csv and <prefix>_summary.csv and writes four standalone HTML files
(plotly) into --out-dir:

    residue.html           residue vs iteration, one line per sequence (legend = seq_id)
    eta.html               step size eta_i vs iteration, same sequences
    first_step_hist.html   histogram of the first step below tolerance, % correct on each bar
    final_correct_hist.html  correctness at max_iter, split by converged / not converged

In the line plots click a legend entry to hide/show that sequence, double-click to isolate it.
Lines are green if the sequence is correct at max_iter, red otherwise; dotted if it never
went below the tolerance. Run it on any machine (no GPU needed): pip install pandas plotly.
"""
import argparse
import os

import numpy as np
import pandas as pd

GREEN, RED = "#1a9850", "#d73027"


def read_csv(path: str) -> pd.DataFrame:
    """Reads both flavours written by inference.py: ';' + decimal ',' (Italian Excel) or ',' + '.'."""
    with open(path) as f:
        it = ";" in f.readline()
    return pd.read_csv(path, sep=";" if it else ",", decimal="," if it else ".")


def select_seqs(summary: pd.DataFrame, seqs, only: str, max_seqs: int):
    if seqs:
        missing = sorted(set(seqs) - set(summary.seq_id))
        if missing:
            raise SystemExit(f"seq_id not found: {missing}")
        return [int(s) for s in seqs]
    s = summary
    if only == "correct":
        s = s[s.correct_at_max_iter == 1]
    elif only == "wrong":
        s = s[s.correct_at_max_iter == 0]
    elif only == "converged":
        s = s[s.first_step_below_tol.notna()]
    elif only == "unconverged":
        s = s[s.first_step_below_tol.isna()]
    ids = s.seq_id.to_numpy()
    if len(ids) > max_seqs:
        ids = ids[np.linspace(0, len(ids) - 1, max_seqs).round().astype(int)]
    return [int(i) for i in ids]


def line_fig(go, trace, summary, ids, col, title, ytitle, logy, hline=None):
    info = summary.set_index("seq_id")
    groups = dict(tuple(trace[trace.seq_id.isin(ids)].groupby("seq_id")))
    fig = go.Figure()
    for sid in ids:
        if sid not in groups:
            continue
        d = groups[sid].sort_values("step")
        ok = info.loc[sid, "correct_at_max_iter"] == 1
        conv = pd.notna(info.loc[sid, "first_step_below_tol"])
        fig.add_trace(go.Scatter(
            x=d["step"], y=d[col], mode="lines", name=str(sid),
            line=dict(color=GREEN if ok else RED, width=1.5, dash="solid" if conv else "dot"),
            legendgroup="corrette" if ok else "sbagliate",
            legendgrouptitle_text="corrette a max_iter" if ok else "sbagliate a max_iter",
            hovertemplate=f"seq {sid}<br>iterazione %{{x}}<br>{ytitle} %{{y:.4g}}<extra></extra>",
        ))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="gray", annotation_text=f"tol = {hline:g}")
    fig.update_layout(
        title=title, xaxis_title="iterazione", yaxis_title=ytitle,
        yaxis_type="log" if logy else "linear", template="plotly_white",
        legend=dict(title_text="seq_id", groupclick="toggleitem"),
    )
    return fig


def first_step_fig(go, summary, bin_width):
    conv = summary[summary.first_step_below_tol.notna()]
    never = summary[summary.first_step_below_tol.isna()]
    labels, n_ok, n_bad = [], [], []
    if len(conv):
        top = int(np.ceil(conv.first_step_below_tol.max() / bin_width) * bin_width)
        edges = np.arange(0, top + bin_width, bin_width)
        code = pd.cut(conv.first_step_below_tol, edges, labels=False)  # (0,bw], (bw,2bw], ...
        for k in range(len(edges) - 1):
            sel = conv[code == k]
            ok = int((sel.correct_at_natural_stop == 1).sum())
            labels.append(f"{k * bin_width + 1}-{(k + 1) * bin_width}")
            n_ok.append(ok)
            n_bad.append(len(sel) - ok)
    if len(never):  # no natural stop: correctness taken at max_iter
        ok = int((never.correct_at_max_iter == 1).sum())
        labels.append("non convergono (corrette a max_iter)")
        n_ok.append(ok)
        n_bad.append(len(never) - ok)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=n_ok, name="corrette", marker_color=GREEN))
    fig.add_trace(go.Bar(x=labels, y=n_bad, name="sbagliate", marker_color=RED))
    for lab, a, b in zip(labels, n_ok, n_bad):
        if a + b:
            fig.add_annotation(x=lab, y=a + b, text=f"{100 * a / (a + b):.0f}%", showarrow=False, yshift=10)
    fig.update_layout(
        barmode="stack", template="plotly_white",
        title="Primo passo sotto la tolleranza (sulla barra: % corrette all'arresto naturale)",
        xaxis_title="primo passo sotto la tolleranza", yaxis_title="numero di sequenze",
    )
    return fig


def final_fig(go, summary):
    conv = summary.first_step_below_tol.notna()
    ok = summary.correct_at_max_iter == 1
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["corrette", "sbagliate"], y=[int((ok & conv).sum()), int((~ok & conv).sum())],
                         name="convergono", marker_color="#4575b4"))
    fig.add_trace(go.Bar(x=["corrette", "sbagliate"], y=[int((ok & ~conv).sum()), int((~ok & ~conv).sum())],
                         name="non convergono", marker_color="#bbbbbb"))
    n = len(summary)
    for lab, tot in (("corrette", int(ok.sum())), ("sbagliate", int((~ok).sum()))):
        fig.add_annotation(x=lab, y=tot, text=f"{tot} ({100 * tot / n:.1f}%)", showarrow=False, yshift=10)
    fig.update_layout(
        barmode="stack", template="plotly_white",
        title="Correttezza all'ultimo passo (max_iter)", yaxis_title="numero di sequenze",
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="the --trace_out prefix used by inference.py")
    ap.add_argument("--out-dir", default=None, help="default: the folder of the prefix")
    ap.add_argument("--seqs", type=int, nargs="*", help="explicit seq_id list for the line plots")
    ap.add_argument("--only", choices=["all", "correct", "wrong", "converged", "unconverged"], default="all")
    ap.add_argument("--max-seqs", type=int, default=30, help="max lines in the line plots (evenly spaced)")
    ap.add_argument("--tol", type=float, default=0.1, help="tolerance line on the residue plot")
    ap.add_argument("--bin-width", type=int, default=10, help="bin width (iterations) of the first-step histogram")
    ap.add_argument("--logy", action=argparse.BooleanOptionalAction, default=True, help="log y-axis for the residue")
    ap.add_argument("--log-eta", action="store_true", help="log y-axis for eta")
    ap.add_argument("--offline", action="store_true", help="embed plotly.js in the HTML (default: load from CDN)")
    args = ap.parse_args()

    try:
        import plotly.graph_objects as go
    except ImportError:
        raise SystemExit("plotly is required: pip install plotly pandas")

    trace = read_csv(f"{args.prefix}_trace.csv")
    summary = read_csv(f"{args.prefix}_summary.csv")
    out_dir = args.out_dir or os.path.dirname(args.prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    n = len(summary)
    conv = summary.first_step_below_tol.notna()
    print(f"{n} sequences | converged {conv.mean():.1%} | "
          f"correct at natural stop {(summary.correct_at_natural_stop == 1).sum() / max(conv.sum(), 1):.1%} of converged | "
          f"correct at max_iter {(summary.correct_at_max_iter == 1).mean():.1%}")

    ids = select_seqs(summary, args.seqs, args.only, args.max_seqs)
    print(f"line plots: {len(ids)} sequences")
    incl = True if args.offline else "cdn"
    figs = {
        "residue": line_fig(go, trace, summary, ids, "residue", "Residuo vs iterazione", "residuo",
                            args.logy, hline=args.tol),
        "eta": line_fig(go, trace, summary, ids, "eta", "Step size vs iterazione", "eta",
                        args.log_eta),
        "first_step_hist": first_step_fig(go, summary, args.bin_width),
        "final_correct_hist": final_fig(go, summary),
    }
    for name, fig in figs.items():
        path = os.path.join(out_dir, f"{name}.html")
        fig.write_html(path, include_plotlyjs=incl)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()