# %% [markdown]
# # Paper ablations — figures
#
# Every figure for experiments 1–4 is drawn here; nothing here measures anything. Produce the data
# first:
#
# ```
# python experiments/harness/scrape.py baseline      # -> data/paper/baseline.npz
# python experiments/harness/scrape.py aux clone attention
# python experiments/harness/specialize.py <study> measure    # -> spec_<study>.npz
# ```
#
# Until those exist, `load.fixture(...)` re-labels finished past runs as a study's arms (real
# numbers, fictional conditions) and `load.synthetic(...)` fabricates the arrays whose producer does
# not exist yet — the spread ladder, the boundary fold, the attention map. Anything synthetic is
# stamped on the face of the panel it appears in.
#
# Terms: `experiments/CONTEXT.md`. Formulas: `docs/reference/Metrics.md`. Protocol: ADR-0021.

# %%
import builtins
import sys
from pathlib import Path

INTERACTIVE = hasattr(builtins, "__IPYTHON__")

if INTERACTIVE:
    # `figures.py` and `load.py` are libraries edited ALONGSIDE this notebook, and a kernel keeps the
    # module objects it first imported. Without this, editing a helper's signature surfaces as a
    # TypeError in a cell whose own source is already correct -- the traceback points at the call,
    # the call is right, and the stale import is invisible. autoreload 2 re-imports them per cell.
    from IPython import get_ipython

    _ip = get_ipython()
    _ip.run_line_magic("load_ext", "autoreload")
    _ip.run_line_magic("autoreload", "2")

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))            # so `python experiments/paper/analysis.py` works too

from experiments.paper import figures as F   # noqa: E402
from experiments.paper import load           # noqa: E402

RETENTION_F = 0.8            # metric 3's valid width, pre-registered (see load.retention_width)
RETENTION_SENS = (0.7, 0.8, 0.9)


def _rollups():
    """Every study on the fixture path, synthetic gaps filled. Swap for `load.study(s)` alone once
    the real scrapes and the ladder/spec passes exist."""
    return {s: load.synthetic(load.study(s)) for s in ("aux", "clone", "attention")}


# Built at import so every display cell below has it. Cheap: a few npz loads and some RNG.
R = _rollups()


def show(fig):
    """Display a figure in an interactive kernel; do nothing under plain `python analysis.py`.

    Returns NOTHING, on purpose. A kernel auto-displays a cell's last expression, so returning the
    figure renders it a SECOND time whenever a `show(...)` ends a cell -- which is every
    single-figure cell here, and the last figure of every multi-figure one. The explicit `fig.show()`
    is what stays, because auto-display alone would drop all but the last figure in a cell.

    The display calls below live at MODULE level, which is what makes this a notebook -- a `# %%`
    cell is just top-level code. That also means they run when the file is executed as a script, and
    an unguarded `fig.show()` there opens a browser tab per figure. The script path renders to PNG at
    the bottom instead, so this is the one place that has to know which mode it is in.

    If a figure comes up blank in an interactive window, the renderer is the usual cause:
        import plotly.io as pio; print(pio.renderers.default)
    VS Code's Interactive Window wants "vscode"; classic Jupyter wants "notebook". Plotly also needs
    `nbformat >= 4.2` to emit its mime bundle at all, which is why it is a project dependency.
    """
    if INTERACTIVE:
        fig.show()


def show_table(df, caption=None):
    """Render a DataFrame richly in a kernel, as text otherwise.

    `display` is injected into the interactive NAMESPACE by the kernel, not into builtins that an
    imported module can see -- so relying on the bare name works when a cell is pasted and breaks the
    moment the file is imported. Imported explicitly here for that reason.

    Returns nothing, for the same reason `show` does not: a returned frame is auto-displayed a
    second time by the cell that called this.
    """
    if not INTERACTIVE:
        return
    try:
        from IPython.display import display
    except ImportError:
        print(df.to_string())
    else:
        display(df)
    if caption:
        print(caption + "\n")


# %% [markdown]
# ## Return curve ∥ specialized return
#
# ADR-0021 says metric 5 is "plotted as markers on metric 1's axes" and also that "no figure may
# place [no-resample returns] on shared axes" with a resampling run — a direct contradiction, since
# the specialization pass runs `resample_interval: 0`. Resolved by sharing the **y** (both in raw
# return units) and not the x: same units, same glance, deliberately not the same series.
#
# Read direction differs by experiment. For `aux` the right panel is the decision metric and the
# left is training dynamics; for `clone` the **left** decides (metric 1 at the final window) and the
# right is the collapse check that catches a metric-1 win bought by locking onto one easy body.

# %%
def fig_return_and_spec(r, *, title="", seed_paths=False):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.74, 0.26],
                        horizontal_spacing=0.035,
                        subplot_titles=("return curve — window average, generator's own bodies",
                                        "specialized return"))
    # mean_ci's seed axis is 1 of an (arm, seed, ...) array -- reduce once, then index by arm.
    Rm, Rlo, Rhi = load.mean_ci(r.R_mean)
    F.arm_lines(fig, r.windows, Rm, Rlo, Rhi, r.arms, row=1, col=1)

    fig.add_vline(x=r.rl_start - 0.5, line=dict(color=F.INK_MUTED, width=1, dash="dot"),
                  annotation_text=f"pretrain → RL (w={r.rl_start})",
                  annotation_position="bottom right",
                  annotation=dict(font=dict(size=10, color=F.INK_MUTED)), row=1, col=1)

    # Right: one x position per checkpoint, seeds jittered behind the arm mean. The tick labels are
    # the checkpoints' OWN gen_window numbers; `ckpt_metric` is what aligns them to the left panel.
    n_ck = len(r.ckpt_gen)
    xs = np.arange(n_ck, dtype=float)
    Sm, Slo, Shi = load.mean_ci(r.spec)
    for j, arm in enumerate(r.arms):
        a = r.arm(arm)
        c = F.colour(arm)
        off = (j - (len(r.arms) - 1) / 2) * 0.16
        spec = r.spec[a]                                          # (seed, ckpt)
        jit = off + np.linspace(-0.045, 0.045, len(r.seeds))[:, None]
        fig.add_trace(go.Scatter(
            x=(xs[None, :] + jit).ravel(), y=spec.ravel(), mode="markers",
            marker=dict(color=F.rgba(c, 0.5), size=7, line=dict(width=1, color=F.SURFACE)),
            showlegend=False, legendgroup=arm, hoverinfo="skip"), row=1, col=2)
        if seed_paths:
            for s in range(len(r.seeds)):
                fig.add_trace(go.Scatter(x=xs + off, y=spec[s], mode="lines",
                                         line=dict(color=F.rgba(c, 0.25), width=1),
                                         showlegend=False, legendgroup=arm, hoverinfo="skip"),
                              row=1, col=2)
        m = Sm[a]
        fig.add_trace(go.Scatter(
            x=xs + off, y=m,
            error_y=dict(type="data", array=Shi[a] - m, arrayminus=m - Slo[a],
                         color=c, width=4, thickness=1.5),
            mode="lines+markers", line=dict(color=c, width=2),
            marker=dict(color=c, size=10, line=dict(width=1.5, color=F.SURFACE)),
            showlegend=False, legendgroup=arm,
            hovertemplate=f"{arm}: %{{y:.4g}}<extra></extra>"), row=1, col=2)

    fig.update_xaxes(title_text="resample window", row=1, col=1)
    fig.update_xaxes(title_text="checkpoint", tickmode="array", tickvals=xs,
                     ticktext=[f"w={w}" for w in r.ckpt_gen],
                     range=[-0.5, n_ck - 0.35], row=1, col=2)
    fig.update_yaxes(title_text="return (raw units)", row=1, col=1)
    F.style(fig, height=440, title=title)
    F.caption(fig,
              "Bands / whiskers: 95% CI across seeds (Student's t). Left is a joint body×control "
              "score on the generator's own bodies — not a control-quality curve.<br>"
              "Right shares the y-axis but not the x: the 250-epoch fine-tune never resamples, so "
              "it is the same unit and deliberately not the same series. Dots are individual seeds.")
    F.synthetic_badge(fig, r.synthetic & {"spec"}, r)
    return fig


# %%
show(fig_return_and_spec(R["aux"], title="Experiment 2 — aux: return and what the design was worth"))
show(fig_return_and_spec(R["clone"], title="Experiment 3 — clone: return and the collapse check"))


def final_window(r, key="R_mean"):
    """Each (arm, seed)'s value at its LAST logged window, not at index -1.

    A run that stopped short trails NaN on the study's declared axis (scrape pads to the budget, on
    purpose), so `[..., -1]` reads the padding rather than the run. Experiment 3's decision metric
    is metric 1 *at the final window*, so this is the number the whole experiment turns on.
    """
    a = r.arrays[key]
    out = np.full(a.shape[:2], np.nan)
    for i in np.ndindex(*a.shape[:2]):
        ok = np.flatnonzero(np.isfinite(a[i]))
        if ok.size:
            out[i] = a[i][ok[-1]]
    return out


# %% [markdown]
# ## Experiment 3 — the 2×2, as a 2×2
#
# `clone.md` predicts an ordering *and a mechanism*: `both ≥ mse_only > kl_only > none`, because a
# displaced actor is a one-time policy jump PPO walks back within a few epochs while a displaced
# critic poisons the advantages for every control step of the following window. That is a
# main-effects claim, and four coloured lines on a return curve cannot show it.
#
# Lines here carry the MSE level and are drawn in ink; the markers stay in their arm's colour, so
# the arm→colour contract survives a figure whose grouping variable is not the arm.

# %%
CLONE_CELLS = {"none": (0, 0), "kl_only": (1, 0), "mse_only": (0, 1), "both": (1, 1)}


def fig_clone_interaction(r, *, title=""):
    fin = final_window(r, "R_mean")
    m, lo, hi = load.mean_ci(fin, axis=1)
    fig = go.Figure()
    for mse_on, dash, nm in ((1, None, "MSE on (lam = 0.399)"), (0, "dash", "MSE off (lam = 0)")):
        cells = [(kl, arm) for arm, (kl, ms) in CLONE_CELLS.items() if ms == mse_on]
        cells.sort()
        xs = [kl for kl, _ in cells]
        ys = [m[r.arm(arm)] for _, arm in cells]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=nm,
                                 line=dict(color=F.INK_2, width=2, dash=dash),
                                 hoverinfo="skip"))
        F.direct_label(fig, xs[-1], ys[-1], nm, F.INK_2, dx=10)
    for arm, (kl, _) in CLONE_CELLS.items():
        a = r.arm(arm)
        c = F.colour(arm)
        fig.add_trace(go.Scatter(
            x=[kl], y=[m[a]],
            error_y=dict(type="data", array=[hi[a] - m[a]], arrayminus=[m[a] - lo[a]],
                         color=c, width=5, thickness=1.5),
            mode="markers", marker=dict(color=c, size=13, line=dict(width=2, color=F.SURFACE)),
            name=arm, showlegend=True,
            hovertemplate=f"{arm}: %{{y:.4g}}<extra></extra>"))
    fig.update_xaxes(title_text="actor clone (KL)", tickmode="array", tickvals=[0, 1],
                     ticktext=["off  (beta = 0)", "on  (beta = 0.119)"], range=[-0.35, 1.55])
    fig.update_yaxes(title_text="return at final window (raw units)")
    F.style(fig, height=440, title=title)
    F.caption(fig,
              "Whiskers: 95% CI across seeds. Vertical gap between the lines is the CRITIC main "
              "effect; slope along x is the ACTOR main effect; non-parallelism is the interaction.<br>"
              "clone.md predicts the gap exceeds the slope — a poisoned critic biases advantages for "
              "a whole window, while a displaced actor decays within a few epochs.")
    return fig


# %%
show(fig_clone_interaction(R["clone"], title="Experiment 3 — the 2×2"))


# %% [markdown]
# ## Experiment 3 — the boundary-recovery trace
#
# `control/r_step` folded on the RL resample boundaries. Offset 0 is the last epoch of the closing
# window, so the resample fires between 0 and +1 and the dip belongs to the new bodies. Normalised
# to each arm's own pre-boundary level: the arms sit at different absolute competence, which is the
# left panel of Fig 3.1's job, and mixing it in here would conflate *worse overall* with *dips
# deeper*. What is under test is the transient alone.
#
# The dip mixes two causes — the bodies are new, and the trunk moved under control — and only the
# clone scalars in the next figure isolate the second. Read them together.

# %%
def fig_boundary_fold(r, *, title=""):
    off = np.arange(-8, r.rew_fold.shape[-1] - 8)
    pre = np.nanmean(r.rew_fold[..., off <= 0], axis=-1, keepdims=True)
    norm = r.rew_fold / pre
    m, lo, hi = load.mean_ci(norm)
    fig = go.Figure()
    F.arm_lines(fig, off, m, lo, hi, r.arms)
    F.hline(fig, 1.0, "pre-boundary level")
    fig.add_vline(x=0.5, line=dict(color=F.INK_2, width=1.5),
                  annotation_text="resample", annotation_position="top left",
                  annotation=dict(font=dict(size=10, color=F.INK_2)))
    if r.epochs_per_window <= off[-1]:
        fig.add_vline(x=r.epochs_per_window, line=dict(color=F.INK_MUTED, width=1, dash="dot"),
                      annotation_text="next boundary", annotation_position="top right",
                      annotation=dict(font=dict(size=10, color=F.INK_MUTED)))
    fig.update_xaxes(title_text="epochs from the resample boundary")
    fig.update_yaxes(title_text="r_step ÷ pre-boundary level")
    F.style(fig, height=430, title=title)
    F.caption(fig,
              f"Mean over the {r.n_windows - r.rl_start} RL boundaries, then across seeds; band is "
              f"the 95% CI across seeds. Normalised per arm, so absolute competence is NOT shown "
              f"here — see Fig 3.1.<br>A window is {r.epochs_per_window} epochs, so an arm still "
              f"below 1.0 at the right-hand edge has not recovered before the next resample "
              f"arrives.")
    F.synthetic_badge(fig, r.synthetic & {"rew_fold", "r_step"}, r)
    return fig


# %%
show(fig_boundary_fold(R["clone"], title="Experiment 3 — boundary recovery"))


# %% [markdown]
# ## Experiment 3 — uncorrected drift, against a yardstick
#
# `kl` and `crit` are computed *before* being scaled by `beta`/`lam` and logged unconditionally, so
# the `none` arm reports its own counterfactual for free: exactly how far the trunk moves control
# per window with no correction applied. `clone.md`'s falsifier turns on reading that as large or
# small — small means there was nothing to preserve (delete both terms), large with no return gap
# means control absorbs it unaided (the clone is unnecessary, not inert).
#
# Which is unreadable without a reference, so `info/kl` — rl_games' own per-epoch policy KL — is
# drawn as a band behind it. Drift above that band means one resample displaces control further
# than a whole epoch of learning moves it.
#
# **Caveat carried in the caption:** `codesign_agent.py:771` reduces these with a mean over every
# minibatch of all 16 generator epochs, so they are a *within-update mean*, not the end-of-update
# residual — which understates the displacement control actually carries into the next window.

# %%
def fig_clone_drift(r, *, title=""):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("actor displacement — clone/actor_kl (nats)",
                                        "critic displacement — clone/critic_mse"))
    for row, key in ((1, "clone_kl"), (2, "clone_mse")):
        m, lo, hi = load.mean_ci(r.arrays[key])
        # A KL and a squared error are both non-negative; only the CI's normal approximation dips
        # below zero, and letting it set the axis wastes a third of the panel. `rangemode="tozero"`
        # only guarantees zero is INCLUDED, so the floor has to be set explicitly.
        top = float(np.nanmax(hi)) * 1.06
        F.arm_lines(fig, r.windows, m, lo, hi, r.arms, row=row, col=1,
                    showlegend=(row == 1), rng=(0.03 * top, top))
        fig.update_yaxes(range=[0, top], row=row, col=1)
    if r.has("ppo_kl"):
        lo, mid, hi = np.nanpercentile(r.ppo_kl, [10, 50, 90])
        fig.add_hrect(y0=lo, y1=hi, fillcolor=F.rgba(F.INK_MUTED, 0.13), line_width=0,
                      row=1, col=1)
        fig.add_hline(y=mid, line=dict(color=F.INK_MUTED, width=1, dash="dot"),
                      annotation_text="info/kl p10–p90 (one PPO epoch)",
                      annotation_position="bottom right",
                      annotation=dict(font=dict(size=10, color=F.INK_MUTED)), row=1, col=1)
    for row in (1, 2):
        fig.add_vline(x=r.rl_start - 0.5, line=dict(color=F.INK_MUTED, width=1, dash="dot"),
                      row=row, col=1)
    fig.update_xaxes(title_text="resample window", row=2, col=1)
    F.style(fig, height=620, title=title)
    F.caption(fig,
              "Bands: 95% CI across seeds. Both terms are computed BEFORE being scaled by beta/lam "
              "and logged in every arm, so `none` measures the drift it is not correcting.<br>"
              "Grey band is the run's own info/kl p10–p90 — the yardstick. Caveat: these are a mean "
              "over all 16 generator epochs of the update, so they understate the end-of-window "
              "displacement.")
    return fig


# %%
show(fig_clone_drift(R["clone"], title="Experiment 3 — the displacement the clone corrects"))


# %% [markdown]
# ## Metrics 3 and 4 — the control-generalization curve and the GenCrit overlay
#
# The x-axis is **perturbation distance** in modules, not temperature: levels are found by bisecting
# `T` until the population's mean `d_struct` from the committed body hits an integer, so the axis is
# identical across every run and seed and its top end is set by the grammar rather than by the run.
#
# GenCrit's prediction is an **overlay, not a second panel** — the actual line *is* metric 3's curve,
# so the vertical gap between them is the bias, read directly. The reported number is that bias with
# level 0 subtracted: GenCrit regresses returns collected under the sampled policy by earlier,
# weaker controllers, while the ladder rolls out the final policy at μ, and level 0 is the
# generator's own mode — so `bias(0)` *is* the in-distribution offset and subtracting it removes
# both confounds at once. A per-level correlation is the wrong statistic and is not reported.
#
# The bottom strip is the guard: per the `free_entropy` finding the cheap subtype axis moves first,
# so a flat curve with near-zero **skeleton share** means the ladder never tested a new body plan.

# %%
def fig_ladder(r, *, title=""):
    C, K = r.G.shape[2], r.G.shape[3]
    fig = make_subplots(
        rows=3, cols=C, shared_xaxes=True, vertical_spacing=0.07, horizontal_spacing=0.045,
        row_heights=[0.5, 0.28, 0.22],
        subplot_titles=[f"checkpoint w={w}" for w in r.ckpt_gen] + [""] * (2 * C))
    # `nested_bands` is the harness's, so the inner CI and the outer pooled across-body sd are the
    # same two quantities every other reader of this sidecar gets.
    Gm, Glo, Ghi, Olo, Ohi = load.nested_bands(r.G, r.G_sd)
    Pm, Plo, Phi = load.mean_ci(r.pred)
    Sm, _, _ = load.mean_ci(r.skel_share)
    bias = r.pred - r.G
    Xm, Xlo, Xhi = load.mean_ci(bias - bias[..., :1])       # excess bias, anchored at level 0
    Dm = np.nanmean(r.dist, axis=1)                          # the MEASURED per-level distance
    # An arm whose ladder pass has not run yet is ABSENT, not zero. Drawing its GenCrit trace anyway
    # registers a legend entry for a line that was never measured, which reads as a null result.
    measured = [i for i, _ in enumerate(r.arms) if np.isfinite(r.G[i]).any()]
    pending = [a for i, a in enumerate(r.arms) if i not in measured]
    for c in range(C):
        for i in measured:
            arm = r.arms[i]
            x = Dm[i, c]
            F.add_ci(fig, x, Gm[i, c], Olo[i, c], Ohi[i, c], arm=arm, row=1, col=c + 1,
                     showlegend=False, label=False, alpha=0.07, width=0)
            F.add_ci(fig, x, Gm[i, c], Glo[i, c], Ghi[i, c], arm=arm, row=1, col=c + 1,
                     showlegend=(c == 0), label=False)
            fig.add_trace(go.Scatter(
                x=x, y=Pm[i, c], mode="lines", line=dict(color=F.colour(arm), width=2, dash="dot"),
                name=f"{arm} (GenCrit)", showlegend=(c == 0), legendgroup=arm,
                hovertemplate=f"{arm} predicted: %{{y:.4g}}<extra></extra>"), row=1, col=c + 1)
            F.add_ci(fig, x, Xm[i, c], Xlo[i, c], Xhi[i, c], arm=arm, row=2, col=c + 1,
                     showlegend=False, label=False)
            fig.add_trace(go.Scatter(x=x, y=Sm[i, c], mode="lines",
                                     line=dict(color=F.colour(arm), width=2),
                                     showlegend=False, legendgroup=arm, hoverinfo="skip"),
                          row=3, col=c + 1)
        if c == C - 1 and measured:
            span = np.nanmax(Ghi[measured, c]) - np.nanmin(Olo[measured, c])
            ends = np.array([Gm[i, c][np.isfinite(Gm[i, c])][-1] for i in measured])
            F.label_series(fig, np.nanmax(Dm[measured, c]), ends, [r.arms[i] for i in measured],
                           span=span, rng=(np.nanmin(ends), np.nanmax(Ghi[measured, c])),
                           row=1, col=c + 1)
        fig.add_hline(y=0, line=dict(color=F.INK_MUTED, width=1), row=2, col=c + 1)
        # Excess bias is `pred - G` at the far end of the ladder, where GenCrit's per-seed
        # predictions diverge wildly -- one seed's blow-up at d~22 set a +/-20k axis on a signal of a
        # few hundred and flattened the panel to a line at zero. Scale from the MEAN curve and let
        # the band clip; the caption says so.
        fin = np.isfinite(Xm[measured, c])
        if fin.any():
            v = Xm[measured, c][fin]
            lo_, hi_ = float(np.min(v)), float(np.max(v))
            pad = 0.25 * max(hi_ - lo_, 1e-9)
            fig.update_yaxes(range=[min(lo_ - pad, -pad), max(hi_ + pad, pad)], row=2, col=c + 1)
        # The generator's default spread sits at T = 1; `T` is 1/beta, so interpolate on it.
        t1 = np.nanmean([np.interp(1.0, r.T[i, s, c], r.dist[i, s, c])
                         for i in range(len(r.arms)) for s in range(len(r.seeds))])
        for row in (1, 2, 3):
            fig.add_vline(x=t1, line=dict(color=F.INK_MUTED, width=1, dash="dash"), row=row, col=c + 1)
        fig.update_xaxes(title_text="perturbation distance (modules)" if c == C // 2 else "",
                         row=3, col=c + 1)
    fig.update_yaxes(title_text="return (raw)", row=1, col=1)
    fig.update_yaxes(title_text="excess bias", row=2, col=1)
    fig.update_yaxes(title_text="skeleton share", range=[0, 1.05], row=3, col=1)
    if pending:
        fig.add_annotation(xref="paper", yref="paper", x=0.0, y=1.06, xanchor="left",
                           text=f"not yet measured: {', '.join(pending)} (ladder pass not run)",
                           showarrow=False, font=dict(size=11, color=F.INK_MUTED))
    F.style(fig, height=800, title=title)
    F.caption(fig,
              "Top: solid = actual return at μ; dotted = GenCrit's prediction for the same bodies. "
              "Inner band 95% CI across seeds; outer band the pooled across-BODY sd. At distance 0 "
              "every body is the identical committed body, so the outer band there is pure episode "
              "noise — this metric's own floor.<br>"
              "Middle: bias with level 0 subtracted. Anchoring removes the μ-vs-sampled and "
              "staleness confounds, neither constant across arms; above 0 = over-optimism about "
              "unfamiliar designs. Dashed rule = the generator's own spread (T=1).<br>"
              "Bottom: skeleton share — a flat top curve with a near-zero share means the ladder "
              "only ever swapped subtypes, never a body plan. x is the MEASURED mean d_struct per "
              "level, not the integer it was bisected onto.<br>"
              "The excess-bias row is scaled to its mean curve, so a far-ladder seed whose GenCrit "
              "prediction diverges clips rather than flattening the panel.")
    F.synthetic_badge(fig, r.synthetic & {"G", "G_sd", "pred", "skel_share", "T", "dist"}, r)
    return fig


# %%
show(fig_ladder(R["aux"], title="Experiment 2 — control generalization and GenCrit's judgement"))
show(fig_ladder(R["clone"], title="Experiment 3 — control generalization and GenCrit's judgement"))


# %% [markdown]
# ## Metric 4 — exploration, as a joint rather than two marginals
#
# Breadth and travel are both required and neither is sufficient, and they come apart in two named,
# opposite ways: a generator holding one design that marches across design space explores while
# reading as fully collapsed, and one holding the same three designs forever reads as healthy while
# exploring nothing. Both are *corners of the joint*, so the panel is the plane rather than its two
# marginals. Colour is the window index here — the trajectory is the whole point — so arm identity
# moves to the panel title and the final-window marker.
#
# Travel is read against its **measured split-half null**, never against zero: at the null the
# energy distance is as often slightly negative as slightly positive, and a small negative value is
# the estimator working. The panel starts at the pretrain→RL boundary because `build/n_modes` is not
# logged during pretrain.

# %%
def fig_exploration(r, *, title=""):
    n = len(r.arms)
    fig = make_subplots(rows=2, cols=n, vertical_spacing=0.17, horizontal_spacing=0.05,
                        row_heights=[0.62, 0.38],
                        specs=[[{} for _ in range(n)],
                               [{"colspan": n}] + [None] * (n - 1)],
                        subplot_titles=[f"{a}" for a in r.arms] + ["mode coverage — cumulative"])
    breadth = np.nanmean(r.n_modes, axis=1)
    excess = np.nanmean(r.energy - r.energy_null, axis=1)
    for i, arm in enumerate(r.arms):
        ok = np.isfinite(breadth[i]) & np.isfinite(excess[i])
        w = r.windows[ok]
        fig.add_trace(go.Scatter(
            x=breadth[i][ok], y=excess[i][ok], mode="lines+markers",
            line=dict(color=F.rgba(F.INK_MUTED, 0.45), width=1),
            marker=dict(size=7, color=w, colorscale=F.SEQ, showscale=(i == n - 1),
                        colorbar=dict(title="window", len=0.5, y=0.76, thickness=10)),
            showlegend=False, name=arm,
            hovertemplate="w=%{marker.color}<br>breadth %{x:.2f}<br>travel %{y:.3f}<extra></extra>"),
            row=1, col=i + 1)
        if ok.any():                                  # the arm's own colour, on its endpoint only
            fig.add_trace(go.Scatter(
                x=[breadth[i][ok][-1]], y=[excess[i][ok][-1]], mode="markers",
                marker=dict(size=15, color=F.colour(arm), symbol="circle-open",
                            line=dict(width=3, color=F.colour(arm))),
                showlegend=False, hoverinfo="skip"), row=1, col=i + 1)
        fig.add_hline(y=0, line=dict(color=F.INK_MUTED, width=1, dash="dot"), row=1, col=i + 1)
        fig.update_xaxes(title_text="breadth (n_modes)", row=1, col=i + 1)
        fig.layout.annotations[i].font.color = F.colour(arm)
    fig.update_yaxes(title_text="travel above split-half null", row=1, col=1)
    Cm, Clo, Chi = load.mean_ci(r.coverage)
    F.arm_lines(fig, r.windows, Cm, Clo, Chi, r.arms, row=2, col=1)
    fig.update_xaxes(title_text="resample window", row=2, col=1)
    fig.update_yaxes(title_text="distinct designs found", row=2, col=1)
    F.style(fig, height=740, title=title)
    F.caption(fig,
              "Top: one point per window, colour = window index; the ring is the final window in the "
              "arm's colour. Upper-left = ES-like hill-climbing (travelling while collapsed); "
              "lower-right = breadth without exploration. Travel is read against the MEASURED "
              "split-half null (the zero line here), never against 0 in absolute terms.<br>"
              "Bottom: cumulative greedy cover at τ=1 — monotone by construction, so its SLOPE is "
              "the discovery rate and a plateau means finding stopped, not that motion stopped. "
              "Both panels start at the pretrain→RL boundary; build/* is not logged before it.")
    F.synthetic_badge(fig, r.synthetic & {"n_modes", "energy", "energy_null", "coverage"}, r)
    return fig


# %%
show(fig_exploration(R["aux"], title="Experiment 2 — exploration: breadth and travel together"))


# %% [markdown]
# ## Experiment 2 — the FK hazard, as a joint
#
# `aux.md` names this as a check to run *before* reading a null: at a rest pose the torso-frame
# target is nearly a deterministic function of the morphology, which design mode already reads as
# tokens, so FK may have had nothing to teach. The doc's test is "a flat FK loss" — but a loss that
# crashes to the floor and stays there is equally consistent with the hazard, because a target the
# tokens already determine is learned once and then free.
#
# What discriminates is the loss **against body diversity**: FK pinned at its floor while the
# population spreads means the target was determined by the tokens. Single-arm by construction —
# `losses/fd|fk` are NaN in the `none` arm, whose heads exist but are never armed.

# %%
def fig_aux_hazard(r, arm=None, *, title=""):
    arm = arm or F.CONTROL_ARM[r.study]
    a = r.arm(arm)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        row_heights=[0.58, 0.42],
                        subplot_titles=("auxiliary losses — did the heads do any work?",
                                        "body diversity — was there anything to learn from?"))
    ep = np.arange(1, r.arrays["fd"].shape[-1] + 1)
    for key, c, nm in (("fd", F.BLUE, "losses/fd"), ("fk", F.AQUA, "losses/fk")):
        m = np.nanmean(r.arrays[key][a], axis=0)
        ok = np.isfinite(m) & (m > 0)
        fig.add_trace(go.Scatter(x=ep[ok], y=m[ok], mode="lines", name=nm,
                                 line=dict(color=c, width=2),
                                 hovertemplate=f"{nm}: %{{y:.3g}}<extra></extra>"), row=1, col=1)
        if ok.any():
            F.direct_label(fig, ep[ok][-1], m[ok][-1], nm, c, row=1, column=1, log=True)
    fig.update_yaxes(type="log", title_text="loss (log)", row=1, col=1)
    # Window-cadence diversity onto the same EPOCH axis: window w closes at epoch (w+1)*per_window.
    dv = np.nanmean(r.div_struct[a], axis=0)
    okd = np.isfinite(dv)
    wep = (r.windows + 1) * r.epochs_per_window
    fig.add_trace(go.Scatter(x=wep[okd], y=dv[okd], mode="lines",
                             line=dict(color=F.INK_2, width=2), name="build/div_struct",
                             hovertemplate="div_struct: %{y:.3g}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="mean pairwise d_struct", row=2, col=1)
    fig.update_xaxes(title_text="epoch", row=2, col=1)
    F.style(fig, height=620, title=f"{title} — arm `{arm}`" if title else "")
    F.caption(fig,
              "Mean across seeds of one arm; the ablated arm's aux losses are NaN by construction "
              "(the heads exist but are never armed), so this panel has no comparison and is not "
              "meant to.<br>Read jointly: FK pinned at its floor WHILE div_struct rises means the "
              "target was already determined by the morphology tokens — the hazard — which is a "
              "different finding from 'kinematic grounding does not help'. The tag is `losses/fk`; "
              "there is no `gen/fk`.")
    F.synthetic_badge(fig, r.synthetic & {"div_struct"}, r)
    return fig


# %%
show(fig_aux_hazard(R["aux"], title="Experiment 2 — the FK hazard"))


# %% [markdown]
# ## Experiment 4 — return, asymptote and sample efficiency, in one panel
#
# Off-protocol by construction: one fixed body, no generator, no resampling, no aux heads. ADR-0021's
# metrics 2–5 need a generator and metric 1's `quality/R_mean` is never written, so this experiment
# defines its own measurements — and **none of them may share axes with experiments 1–3**, because
# `resample_interval: 0` silently disables LR warmup and these runs are on a different schedule over
# a different body distribution.
#
# A, B and C are one figure because B is the mean of A over its last 200 epochs and C is where A
# crosses a threshold — drawing them apart would plot the same curve three times. `attention.md` sets
# the threshold at `self_cls`'s asymptote, which makes C degenerate for `self_cls` itself (it reaches
# its own asymptote by definition, at an epoch set by curve noise) and undefined for `self`, which is
# expected never to get there. Ruled at **90%** of that asymptote instead, so every arm that arrives
# has a real learning-speed number; arms that never arrive are annotated rather than dropped.

# %%
ASYMPTOTE_EPOCHS = 200
THRESHOLD_FRAC = 0.90
SMOOTH = 25              # epochs; the crossing is read off the smoothed curve, not a noisy sample


def _smooth(y, n=SMOOTH):
    """Centred moving average that does NOT fall off at the edges.

    A plain `np.convolve(..., "same")` zero-pads, so the last n/2 points are dragged toward zero --
    which on a return curve renders as a collapse in exactly the window measurement B reads. Dividing
    by the convolved validity mask normalises each output by how many real samples went into it, so
    the ends are a shorter average rather than a fabricated cliff.
    """
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return y
    ker = np.ones(min(n, ok.sum()))
    num = np.convolve(np.where(ok, y, 0.0), ker, mode="same")
    den = np.convolve(ok.astype(float), ker, mode="same")
    out = np.full_like(y, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def fig_attention_return(r, *, title=""):
    m, lo, hi = load.mean_ci(r.r_step)
    ep = np.arange(1, m.shape[-1] + 1)
    ref_arm = "self_cls" if "self_cls" in r.arms else r.arms[-1]
    # B and C come from the harness so they are the same numbers the tables report. The threshold is
    # fixed from `self_cls` BEFORE `full` is looked at -- attention.md makes that ordering the
    # caller's job, so it is done here explicitly and once.
    asym = load.asymptote(r.r_step, tail=ASYMPTOTE_EPOCHS)                 # (arm, seed)
    thresh = THRESHOLD_FRAC * float(np.nanmean(asym[r.arm(ref_arm)]))
    cross = load.crossing(r.r_step, thresh, smooth=SMOOTH)                 # (arm, seed)

    fig = go.Figure()
    F.arm_lines(fig, ep, np.array([_smooth(row) for row in m]), lo, hi, r.arms)
    fig.add_vrect(x0=ep[-ASYMPTOTE_EPOCHS], x1=ep[-1], fillcolor=F.rgba(F.INK_MUTED, 0.10),
                  line_width=0, annotation_text=f"B: final {ASYMPTOTE_EPOCHS} epochs",
                  annotation_position="top left",
                  annotation=dict(font=dict(size=10, color=F.INK_MUTED)))
    fig.add_hline(y=thresh, line=dict(color=F.INK_MUTED, width=1, dash="dot"),
                  annotation_text=f"C: {THRESHOLD_FRAC:.0%} of `{ref_arm}` asymptote",
                  annotation_position="top left",
                  annotation=dict(font=dict(size=10, color=F.INK_MUTED)))
    for arm in r.arms:
        c, xs = F.colour(arm), cross[r.arm(arm)]
        if np.isfinite(xs).any():
            x = float(np.nanmean(xs))
            fig.add_trace(go.Scatter(x=[x, x], y=[0, thresh], mode="lines",
                                     line=dict(color=c, width=1.5, dash="dot"),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_annotation(x=x, y=0, text=f"{x:.0f}", showarrow=False, yanchor="top",
                               font=dict(size=10, color=F.INK), yshift=-2)
        n_never = int(np.isnan(xs).sum())
        if n_never:
            fig.add_annotation(x=ep[-1], y=thresh, text=f"{arm}: {n_never}/{len(xs)} never",
                               showarrow=False, xanchor="right", yanchor="bottom",
                               font=dict(size=10, color=c), yshift=4)
    fig.update_xaxes(title_text="epoch")
    fig.update_yaxes(title_text="control/r_step  (mean raw reward per env-step)", rangemode="tozero")
    F.style(fig, height=470, title=title)
    F.caption(fig,
              f"Curves smoothed over {SMOOTH} epochs for display; B and C come from "
              f"`harness.stats` on the raw series (C uses a TRAILING mean, so a crossing never "
              f"depends on the future). Band is the 95% CI across seeds. Shaded strip = "
              f"measurement B. Rule = C's threshold, {THRESHOLD_FRAC:.0%} of `{ref_arm}`'s "
              f"asymptote, fixed before the other arms were read; drop-lines are the across-seed "
              f"mean crossing.<br>"
              f"These runs never resample, so LR warmup never applies — their returns are NOT "
              f"comparable to experiments 1–3 and must never share an axis with them.")
    F.synthetic_badge(fig, r.synthetic & {"r_step"}, r)
    return fig


# %%
show(fig_attention_return(R["attention"], title="Experiment 4 — return, asymptote and sample efficiency"))


# %%
def fig_attention_gait(r, *, title=""):
    rows = [("ep_len", "episode length — falling over"),
            ("sigma", "control/sigma_mean — entropy collapse"),
            ("adv_std", "control/adv_std — advantage scale")]
    rows = [(k, t) for k, t in rows if r.has(k)]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[t for _, t in rows])
    for i, (key, _) in enumerate(rows, start=1):
        m, lo, hi = load.mean_ci(r.arrays[key])
        ep = np.arange(1, m.shape[-1] + 1)
        F.arm_lines(fig, ep, np.array([_smooth(row) for row in m]), lo, hi, r.arms,
                    row=i, col=1, showlegend=(i == 1))
    fig.update_xaxes(title_text="epoch", row=len(rows), col=1)
    F.style(fig, height=240 * len(rows) + 160, title=title)
    F.caption(fig,
              f"Mean across seeds, smoothed over {SMOOTH} epochs; bands are the 95% CI across seeds. "
              f"`attention.md` predicts the gap shows here too, as shorter episodes (falling) in the "
              f"masked arms.<br>Same no-resample caveat as the return panel: not comparable to "
              f"experiments 1–3.")
    return fig


# %%
show(fig_attention_gait(R["attention"], title="Experiment 4 — gait diagnostics"))


# %% [markdown]
# ## Experiment 4 — attention structure
#
# What makes a positive result interpretable: if `full` beats `self_cls`, the learned map must
# actually attend across limbs, and a near-diagonal map paired with a return gap means the gap came
# from something else and the result is not yet explained.
#
# The headline is the **mass split**, not the map. Attention mass is linear in the weights, so
# cross-limb mass survives the state-averaging intact (mean-of-mass = mass-of-mean) — but the
# *pattern* does not: a limb attending to its contralateral partner at one gait phase and another at
# a different phase averages into diffuse mass across both. So the fraction is the robust reading
# and the map is the supporting evidence.
#
# Token layout is derived from the run's own ModuleLibrary, never hardcoded:
# `n_tokens = 1 + n_slots + n_slots·max_depth` and `_content_start = 1 + n_slots`
# (`architectures.py:265`). For the ant's `simple` library that is 8 slots × depth 4 → **41 tokens**,
# `_content_start = 9`. The start tokens are batch-independent learned anchors carrying no state.

# %%
def fig_attention_structure(r, arm=None, *, title=""):
    """Measurement E. `attn_offdiag` is the decisive number; the map is the evidence behind it."""
    arm = arm or F.CONTROL_ARM[r.study]
    n_ck = r.attn_offdiag.shape[-1]
    x = np.arange(n_ck)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.54, 0.46], horizontal_spacing=0.11,
                        subplot_titles=("cross-limb attention share (attn_offdiag)",
                                        f"state-averaged map, final checkpoint — `{arm}`"))
    m, lo, hi = load.mean_ci(r.attn_offdiag)
    F.arm_lines(fig, x, m, lo, hi, r.arms, row=1, col=1)
    fig.add_hline(y=0, line=dict(color=F.INK_MUTED, width=1, dash="dot"),
                  annotation_text="0 — what a self/self_cls mask structurally cannot exceed",
                  annotation_position="bottom left",
                  annotation=dict(font=dict(size=10, color=F.INK_MUTED)), row=1, col=1)
    fig.update_xaxes(title_text="checkpoint", tickmode="array", tickvals=x,
                     ticktext=[str(w) for w in (r.ckpt_gen or x)], range=[-0.2, n_ck - 0.5],
                     row=1, col=1)
    fig.update_yaxes(title_text="share of a module token's mass on OTHER module tokens",
                     rangemode="tozero", row=1, col=1)

    M = np.nanmean(r.attn_map[r.arm(arm), :, -1], axis=(0, 1))     # over seeds and heads
    cs, n_tok = r.content_start, M.shape[-1]
    labels = ["CLS"] + [f"s{i}" for i in range(1, cs)] + [f"m{i}" for i in range(n_tok - cs)]
    fig.add_trace(go.Heatmap(z=M, x=labels, y=labels, colorscale=F.SEQ,
                             colorbar=dict(title="mass", len=0.72, thickness=10),
                             hovertemplate="query %{y} → key %{x}: %{z:.3f}<extra></extra>"),
                  row=1, col=2)
    for b in (0.5, cs - 0.5):
        fig.add_vline(x=b, line=dict(color=F.INK, width=1), row=1, col=2)
        fig.add_hline(y=b, line=dict(color=F.INK, width=1), row=1, col=2)
    fig.update_yaxes(autorange="reversed", title_text="query", row=1, col=2)
    fig.update_xaxes(title_text="key", row=1, col=2)
    F.style(fig, height=580, title=title)
    F.caption(fig,
              "Left: the decisive number — share of a present module token's attention landing on "
              "the OTHER present module tokens, averaged over heads, query tokens and seeds. Run on "
              "the ablated arms too, where it is also the check that the mask does what it claims: "
              "a self/self_cls mask cannot exceed 0.<br>"
              f"Right: mean over seeds and all {r.attn_map.shape[3]} heads at the final checkpoint. "
              f"Rules mark the [CLS] / [start × {r.n_slots}] / [module × {r.n_slots * r.max_depth}] "
              f"blocks. Cross-limb MASS survives state-averaging (mass is linear in the weights); "
              f"the map's PATTERN does not, so the fraction is the reading and the map the evidence.")
    F.synthetic_badge(fig, r.synthetic & {"attn_map", "attn_offdiag"}, r)
    return fig


# %%
show(fig_attention_structure(R["attention"], title="Experiment 4 — is cross-limb attention used?"))


# %% [markdown]
# ## Metric 2 × metric 3 — the pair that is the finding
#
# `backbone.md` is explicit that neither number is the result on its own: an arm that wins
# specialized return while its control-generalization curve collapses got a good design without
# getting good at codesign — it found one body and specialised, which is the local-search story the
# paper argues against. So the two are one panel, with the readings as named quadrants.

# %%
def fig_design_vs_generalization(r, *, f=RETENTION_F, ckpt=-1, title=""):
    dist = np.nanmean(r.dist[:, :, ckpt, :], axis=(0, 1))
    W = load.retention_width(r.G[:, :, ckpt, :], dist, f)         # (arm, seed)
    S = r.spec[:, :, ckpt]
    wm, wlo, whi = load.mean_ci(W, axis=1)
    sm, slo, shi = load.mean_ci(S, axis=1)
    fig = go.Figure()
    fig.add_vline(x=float(np.nanmean(wm)), line=dict(color=F.GRID, width=1))
    fig.add_hline(y=float(np.nanmean(sm)), line=dict(color=F.GRID, width=1))
    for txt, xa, ya, xanc, yanc in (("local search", 0.02, 0.97, "left", "top"),
                                    ("general AND good — the claim", 0.98, 0.97, "right", "top"),
                                    ("weak", 0.02, 0.03, "left", "bottom"),
                                    ("general but poor design", 0.98, 0.03, "right", "bottom")):
        fig.add_annotation(xref="paper", yref="paper", x=xa, y=ya, text=txt, showarrow=False,
                           xanchor=xanc, yanchor=yanc, font=dict(size=11, color=F.INK_MUTED))
    for i, arm in enumerate(r.arms):
        c = F.colour(arm)
        fig.add_trace(go.Scatter(x=W[i], y=S[i], mode="markers", showlegend=False,
                                 marker=dict(color=F.rgba(c, 0.35), size=8,
                                             line=dict(width=1, color=F.SURFACE)),
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[wm[i]], y=[sm[i]],
            error_x=dict(type="data", array=[whi[i] - wm[i]], arrayminus=[wm[i] - wlo[i]],
                         color=c, thickness=1.5, width=5),
            error_y=dict(type="data", array=[shi[i] - sm[i]], arrayminus=[sm[i] - slo[i]],
                         color=c, thickness=1.5, width=5),
            mode="markers", marker=dict(color=c, size=15, line=dict(width=2, color=F.SURFACE)),
            name=arm, showlegend=True,
            hovertemplate=f"{arm}: width %{{x:.2f}}, spec %{{y:.4g}}<extra></extra>"))
    F.label_series(fig, np.nanmax(wm), sm, list(r.arms),
                   span=np.nanmax(shi) - np.nanmin(slo),
                   rng=(np.nanmin(slo), np.nanmax(shi)))
    fig.update_xaxes(title_text=f"retention width at f={f} (modules)")
    fig.update_yaxes(title_text="specialized return (raw units)")
    F.style(fig, height=520, title=title)
    F.caption(fig,
              f"Checkpoint w={r.ckpt_gen[ckpt]}. Large marker = arm mean with 95% CI on both axes; "
              f"small markers are individual seeds. Quadrant lines are the across-arm means, so the "
              f"regions are relative to this study, not absolute.<br>"
              f"Width = perturbation distance at which return first falls below {f:.0%} of its "
              f"level-0 value — the definition backbone.md and aux.md pre-register a falsifier "
              f"against but never give. Sensitivity over f ∈ {RETENTION_SENS} is in the table.")
    F.synthetic_badge(fig, r.synthetic & {"G", "dist", "spec"}, r)
    return fig


# %%
show(fig_design_vs_generalization(R["clone"], title="Metric 2 × metric 3 — the pair that is the finding"))


# %% [markdown]
# ## Tables
#
# Numbers only — no verdict column. Each experiment pre-registers a decision metric and a falsifier,
# but with 8 seeds and ADR-0018's measured seed CV of 9–48% on `quality/R_mean` the smallest
# detectable arm difference is roughly one seed-sd, so a mechanical call would frequently be
# reporting an underpowered measurement as a null. The table puts the effect and the floor side by
# side and leaves the reading where it belongs.
#
# The floor here is the **control arm's across-seed sd** — ADR-0018's noise floor, and the only one
# of the four quantities these docs call a "noise floor" that gates an arm *comparison*. It needs no
# separate calibration wave: eight seeds at one fixed config is that wave, run on exactly the
# configuration every ablation is a delta off.

# %%
import pandas as pd

# Each study's pre-registered decision metric: (label, how to extract it from a rollup).
DECISION = {
    "aux":       ("specialized return @ final checkpoint", lambda r: r.spec[:, :, -1], "spec"),
    "backbone":  ("specialized return @ final checkpoint", lambda r: r.spec[:, :, -1], "spec"),
    "clone":     ("return @ final window", lambda r: final_window(r, "R_mean"), "R_mean"),
    "attention": (f"asymptotic return (final {ASYMPTOTE_EPOCHS} epochs)",
                  lambda r: np.nanmean(r.r_step[..., -ASYMPTOTE_EPOCHS:], axis=-1), "r_step"),
}

SYNTH_MARK = "◆"   # appended to any column whose SOURCE ARRAY is fabricated


def _col(r, label, *keys):
    """Column label, marked when it is computed from synthetic data.

    Listing the fabricated arrays in a footnote stopped being enough once most of the data became
    real: the one remaining fake column was the DECISION METRIC, and a bare number in a table of
    otherwise-measured numbers reads as measured. The mark travels with the value.
    """
    return f"{label} {SYNTH_MARK}" if any(k in r.synthetic for k in keys) else label


def _pm(m, lo, hi, sig=4):
    """`mean ± half-width`, where the half-width is the CI's own, not a symmetrised guess."""
    if not np.isfinite(m):
        return "—"
    half = np.nanmax([hi - m, m - lo])
    return f"{m:.{sig}g} ± {0 if not np.isfinite(half) else half:.{sig - 1}g}"


def table_arms(r) -> pd.DataFrame:
    """One row per arm: the decision metric, the supporting metrics, and the floor beside them.

    The Δ column is **seed-paired** (`stats.paired_ci`), not a difference of two independent means.
    Every arm runs the same seeds 42–49 and a seed fixes the initialisation and the env stream, so
    "seed 44 was a bad seed" is a shared term that cancels in the paired difference and does not
    cancel in two independent means. ADR-0018 measured seed-only CV at 9–48% against a hyperparameter
    spread of 14% — pairing is what buys that back, and it is why the series runs 8 seeds at all.
    """
    ctrl = F.CONTROL_ARM[r.study]
    ci = r.arm(ctrl)
    label, get, src = DECISION[r.study]
    label = _col(r, label, src)
    D = get(r)
    dm, dlo, dhi = load.mean_ci(D, axis=1)
    floor = load.noise_floor(D[ci], axis=-1)

    rows = []
    for i, arm in enumerate(r.arms):
        pm, plo, phi, pn = load.paired_ci(D[i], D[ci])
        row = {"arm": arm, "n": int(np.isfinite(D[i]).sum()),
               label: _pm(dm[i], dlo[i], dhi[i]),
               f"Δ vs {ctrl} (paired)": "—" if i == ci else f"{pm:+.4g} [{plo:+.3g}, {phi:+.3g}]"}
        if r.has("spec") and r.study != "aux":
            sm, slo, shi = load.mean_ci(r.spec[:, :, -1], axis=1)
            row[_col(r, "specialized return", "spec")] = _pm(sm[i], slo[i], shi[i])
        if r.n_windows:
            fm, flo, fhi = load.mean_ci(final_window(r, "R_mean"), axis=1)
            row["return @ final window"] = _pm(fm[i], flo[i], fhi[i])
        if r.has("G"):
            dist = np.nanmean(r.dist[:, :, -1, :], axis=(0, 1))
            for f in RETENTION_SENS:
                w = load.retention_width(r.G[:, :, -1, :], dist, f)
                wm, wlo, whi = load.mean_ci(w, axis=1)
                row[_col(r, f"width f={f}", "G", "dist")] = _pm(wm[i], wlo[i], whi[i], 3)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("arm")
    df.attrs["caption"] = (
        f"{r.study}: decision metric is **{label}**. Floor = the `{ctrl}` arm's across-seed sd "
        f"= {floor['sd']:.4g} (CV {floor['cv']:.0%}, n={floor['n']}) — ADR-0018's noise floor, the "
        f"only one of the four quantities these docs call a 'noise floor' that gates an ARM "
        f"comparison. A |Δ| below it is not resolvable at this seed count. "
        f"± and [..] are 95% intervals across seeds (Student's t); Δ is seed-paired. "
        f"Retention width is where return first falls below f×G(0); f={RETENTION_F} is "
        f"pre-registered, the others are the sensitivity check."
        + (f"  {SYNTH_MARK} = computed from SYNTHETIC data (no producer has run): "
           f"{', '.join(sorted(r.synthetic))}." if r.synthetic else ""))
    return df


def table_series(rollups: dict) -> pd.DataFrame:
    """One row per experiment — the table the paper carries."""
    rows = []
    for name, r in rollups.items():
        ctrl = F.CONTROL_ARM[r.study]
        ci = r.arm(ctrl)
        label, get, src = DECISION[name]
        D = get(r)
        m, _, _ = load.mean_ci(D, axis=1)
        floor = load.noise_floor(D[ci], axis=-1)
        others = [a for a in r.arms if a != ctrl]
        worst = max(others, key=lambda a: abs(m[r.arm(a)] - m[ci])) if others else None
        if worst is None:
            d = lo = hi = np.nan
        else:
            d, lo, hi, _ = load.paired_ci(D[r.arm(worst)], D[ci])
        sd = floor["sd"]
        # The control arm's NAME is a column value, not part of a column header -- putting it in the
        # header gives each study its own column and the rest of the table NaN.
        rows.append({
            "experiment": name, "arms": len(r.arms), "seeds": len(r.seeds),
            "decision metric": _col(r, label, src), "control arm": ctrl,
            "control": f"{m[ci]:.4g}",
            "largest Δ (paired)": "—" if worst is None else f"{d:+.4g} ({worst})",
            "95% CI on Δ": "—" if worst is None else f"[{lo:+.3g}, {hi:+.3g}]",
            "seed floor (sd)": f"{sd:.4g}", "seed CV": f"{floor['cv']:.0%}",
            "|Δ| / floor": "—" if worst is None or not sd else f"{abs(d) / sd:.2f}",
            "synthetic": f"{len(r.synthetic)} arrays" if r.synthetic else "—",
        })
    return pd.DataFrame(rows).set_index("experiment")


# %%
show_table(table_series(R))

# %%
for _name in ("aux", "clone", "attention"):
    _tab = table_arms(R[_name])
    print(f"=== {_name} ===")
    show_table(_tab, _tab.attrs["caption"])


# %% [markdown]
# ## Rendering to files
#
# Running this file as a *script* (`python experiments/paper/analysis.py`) writes every figure to
# `data/paper/figures/` instead of displaying them — a smoke test for the whole notebook. Cells above
# are the interactive path.

# %%
if __name__ == "__main__":
    # Running this file as a script renders every figure to PNG -- a smoke test for the whole
    # notebook. Interactively you run the `# %%` cells instead and the figures display inline.
    import os
    import pathlib
    OUT = pathlib.Path(os.environ.get("PAPER_FIGDIR", "data/paper/figures"))
    OUT.mkdir(parents=True, exist_ok=True)
    aux, clone, attn = R["aux"], R["clone"], R["attention"]
    # Named for what they SHOW, prefixed so they sort into reading order. `fig22.png` told a reader
    # nothing and told the next person to touch this file even less.
    jobs = [
        ("exp2-aux_1_return-and-specialized",        fig_return_and_spec(aux, title="Experiment 2 — aux: return and what the design was worth"), 1180, 520),
        ("exp2-aux_2_control-generalization-gencrit", fig_ladder(aux, title="Experiment 2 — control generalization and GenCrit's judgement"), 1400, 860),
        ("exp2-aux_3_exploration-breadth-travel",    fig_exploration(aux, title="Experiment 2 — exploration: breadth and travel together"), 1200, 760),
        ("exp2-aux_4_fk-hazard-loss-vs-diversity",   fig_aux_hazard(aux, title="Experiment 2 — the FK hazard"), 1180, 640),
        ("exp3-clone_1_return-and-collapse-check",   fig_return_and_spec(clone, title="Experiment 3 — clone: return and the collapse check"), 1180, 520),
        ("exp3-clone_2_interaction-2x2",             fig_clone_interaction(clone, title="Experiment 3 — the 2×2"), 1100, 520),
        ("exp3-clone_3_boundary-recovery",           fig_boundary_fold(clone, title="Experiment 3 — boundary recovery"), 1180, 500),
        ("exp3-clone_4_uncorrected-drift",           fig_clone_drift(clone, title="Experiment 3 — the displacement the clone corrects"), 1180, 700),
        ("exp3-clone_5_control-generalization-gencrit", fig_ladder(clone, title="Experiment 3 — control generalization and GenCrit's judgement"), 1400, 860),
        ("exp4-attention_1_return-asymptote-efficiency", fig_attention_return(attn, title="Experiment 4 — return, asymptote and sample efficiency"), 1180, 500),
        ("exp4-attention_2_gait-diagnostics",        fig_attention_gait(attn, title="Experiment 4 — gait diagnostics"), 1180, 880),
        ("exp4-attention_3_attention-structure",     fig_attention_structure(attn, title="Experiment 4 — is cross-limb attention used?"), 1320, 600),
        ("cross_design-vs-generalization",           fig_design_vs_generalization(clone, title="Metric 2 × metric 3 — the pair that is the finding"), 1100, 560),
    ]
    for nm, fig, w, hgt in jobs:
        fig.write_image(OUT / f"{nm}.png", width=w, height=hgt, scale=2)
    print(f"{len(jobs)} figures -> {OUT}")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== table_series ===")
    print(table_series(R).to_string())
    for name in ("aux", "clone", "attention"):
        print(f"\n=== table_arms({name}) ===")
        tab = table_arms(R[name])
        print(tab.to_string())
        print("\n" + tab.attrs["caption"])
