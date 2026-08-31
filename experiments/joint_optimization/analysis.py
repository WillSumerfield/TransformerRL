# %% [markdown]
# # Joint optimization — figures
#
# Every figure is drawn here; the `.py` scripts only produce data. Run them first:
#
# ```
# python pilot.py                 # chooses the pins for exps 1-3
# python exp1_spread.py
# python exp2_exploration.py
# python exp3_generalization.py
# python exp4_ratios.py           # ~30 min
# python paths.py
# ```
#
# Vocabulary is in `experiments/CONTEXT.md`.

# %%
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

import landscape as L
import sweep

DATA = sweep.DATA
SEQ = "viridis"


def load(name):
    return np.load(DATA / f"{name}.npz")


def surface_pair(z_speed, z_best, x, y, xlab, ylab, title, logaxes=True):
    """The two panels every experiment produces: convergence speed and best fitness.

    Height is the metric, colour is the across-seed std dev -- so for the fitness panel a high,
    evenly-coloured region is both good and reliable, while a high but hot region is a lottery.
    """
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("convergence speed (evals)", "best design fitness"),
        horizontal_spacing=0.06,
    )
    for c, (m, s) in enumerate([z_speed, z_best], start=1):
        fig.add_trace(
            go.Surface(x=x, y=y, z=m.T, surfacecolor=s.T, colorscale=SEQ,
                       colorbar=dict(title="seed std", x=0.44 if c == 1 else 1.02, len=0.8)),
            row=1, col=c,
        )
    axes = dict(type="log", dtick=1) if logaxes else {}
    for scene in ("scene", "scene2"):
        fig.update_layout(**{scene: dict(
            xaxis=dict(title=xlab, **axes), yaxis=dict(title=ylab, **axes),
            zaxis=dict(title=""), camera=dict(eye=dict(x=1.6, y=-1.6, z=1.1)))})
    fig.update_layout(title=title, height=560, margin=dict(l=0, r=0, t=80, b=0))
    return fig


def metrics(z, ax0, ax1):
    """(speed, best) each as (mean, std) over seeds, on the two swept axes."""
    curves = torch.as_tensor(z["curves"]).squeeze()          # (n0, n1, seeds, C)
    best = torch.as_tensor(z["best"]).squeeze()              # (n0, n1, seeds)
    n0, n1, s, _ = curves.shape
    speed = sweep.convergence_evals(curves.reshape(-1, curves.shape[-1]), z["evals"])
    speed = speed.reshape(n0, n1, s).float()
    pack = lambda t: (t.mean(-1).numpy(), t.std(-1).numpy())
    return pack(speed), pack(best), z[f"axis_{ax0}"], z[f"axis_{ax1}"]

# %% [markdown]
# ## The landscape
#
# Both optimizers maximize reward on this one surface. Neither ever sees it whole: the designer only observes a
# design's realized value, the controller only the slice for the design in front of it.
#
# The second panel is the distinction that drives everything. **What the designer is judged on** is the
# marginal — the best value reachable on each design — but **what it can model** is a single slice at
# the controller's current mean action. Where those two curves disagree, the designer's model of
# design quality is wrong no matter how large its generalization radius is.

# %%
n = 401
Zg = L.f(L.grid(n)).numpy()
ax = L.axis(n).numpy()

fig = make_subplots(
    rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "xy"}]],
    column_widths=[0.58, 0.42], horizontal_spacing=0.07,
    subplot_titles=("f(design, action)", "what the designer is judged on vs. what it can model"),
)
fig.add_trace(go.Surface(x=ax, y=ax, z=Zg.T, colorscale=SEQ,
                         colorbar=dict(title="value", x=0.52, len=0.75)), row=1, col=1)

marg = Zg.max(axis=1)
fig.add_trace(go.Scatter(x=ax, y=marg, name="marginal  max_a f(d,a)",
                         line=dict(width=3, color="#111")), row=1, col=2)
for a_fixed in (-0.5, 0.0, 0.5):
    j = int(np.abs(ax - a_fixed).argmin())
    fig.add_trace(go.Scatter(x=ax, y=Zg[:, j], name=f"slice at a={a_fixed:+.1f}",
                             line=dict(width=1.5, dash="dot")), row=1, col=2)

fig.update_layout(
    scene=dict(xaxis_title="design", yaxis_title="action", zaxis_title="value",
               camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0))),
    height=520, margin=dict(l=0, r=0, t=60, b=0),
    legend=dict(title="curve", orientation="h", y=-0.12, x=0.62, xanchor="center"),
)
fig.update_xaxes(title="design", row=1, col=2)
fig.show()

# %%
# How rugged is each optimizer's problem, and does a slice predict the marginal?
# Derived from the display grid (BOUNDS), not the padded peak tables, so these numbers describe
# the region runs actually live in rather than the padded lookup range.
Z = torch.as_tensor(Zg)
T_a = L._ascend(Z)
T_d = L._ascend(Z.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()
n_a = [T_a[i].unique().numel() for i in range(n)]
n_d = [T_d[:, j].unique().numel() for j in range(n)]
print(f"value range          [{Zg.min():+.4f}, {Zg.max():+.4f}]")
print(f"action peaks / slice   min {min(n_a)}  median {sorted(n_a)[n // 2]}  max {max(n_a)}")
print(f"design peaks / slice   min {min(n_d)}  median {sorted(n_d)[n // 2]}  max {max(n_d)}")

# the designer climbs a slice but is judged on the marginal -- how well does that transfer?
realized = torch.gather(Z, 1, T_a).numpy()      # reward once the controller climbs from action j
corr = lambda u, v: float(np.corrcoef(u, v)[0, 1])
print("\n     a      corr(slice, marginal)   corr(realized, marginal)")
for a_fixed in (-0.75, -0.25, 0.25, 0.75):
    j = int(np.abs(ax - a_fixed).argmin())
    print(f"  {a_fixed:+.2f}   {corr(Zg[:, j], marg):>18.3f}   {corr(realized[:, j], marg):>21.3f}")

# %% [markdown]
# **Metrics.** *peaks / slice* is the number of distinct local maxima a single-axis climb can land
# on, so it is how rugged that optimizer's problem is. The two correlations compare predictors of the
# marginal `max_a f(d,.)`: *slice* is `f(d, a_fixed)`, which is what the designer actually climbs,
# and *realized* is the value after the controller climbs from `a_fixed`. Where `corr(slice,
# marginal)` is near zero or negative, the designer's model of design quality is not merely noisy but
# actively misleading.

# %% [markdown]
# ## Experiment 1 — Spread
#
# Read the speed panel knowing that the update steps toward the sample cloud, so wider spread
# mechanically produces larger steps. That coupling is deliberate: wide sampling buying
# faster coarse progress *is* the trade-off being measured.

# %%
sp, bt, x, y = metrics(load("exp1_spread"), "sig_d", "sig_c")
surface_pair(sp, bt, x, y, "designer spread", "controller spread", "Experiment 1 — Spread").show()

# %% [markdown]
# **What this measures.** Whether designer and controller spread trade off against each other, and
# whether either has an interior optimum rather than "wider is always better". Spread is the standard
# deviation of each side's Gaussian proposal, in domain units.
#
# **Metrics** (both panels recur in Experiments 1-3):
#
# - **convergence speed (evals)** — `f` evaluations before best-so-far first comes within 1% of that
#   run's *own* final value. Lower is faster. It says nothing about *where* a run converged, so a run
#   that collapses instantly onto a bad design scores as maximally fast; only read it against the
#   right panel.
# - **best design fitness** — the highest design fitness reached at any point in the run, where a
#   design's fitness is its mean reward over the `k * P_a` actions the controller played on it.
#   Higher is better.
# - **colour (both panels)** — standard deviation across seeds. High and evenly coloured is good
#   *and* reliable; high but hot is a lottery.

# %% [markdown]
# ## Experiment 2 — Exploration
#
# The `e = 1` edges are degenerate by construction: nothing ever climbs there, so the pinned
# generalization values have no effect at all. That flat edge is a finding, not an artifact.

# %%
sp, bt, x, y = metrics(load("exp2_exploration"), "e_d", "e_c")
surface_pair(sp, bt, x, y, "designer exploration", "controller exploration",
             "Experiment 2 — Exploration", logaxes=False).show()

# %% [markdown]
# **What this measures.** How much undirected sampling each side needs. Exploration `e` is the
# probability that a sample skips the climb and is kept where it landed, so `e = 0` is pure
# hill-climbing and `e = 1` is blind random search. The question is whether the designer — which
# climbs a surrogate that can be actively misleading — needs more randomness than the controller,
# which climbs the true slice it is scored on.

# %% [markdown]
# ## Experiment 3 — Generalization
#
# This is where the matching-radii hypothesis lives. The controller's radius is measured in the joint
# space, so a designer ranging beyond it gets its good designs played badly — if the hypothesis holds,
# the off-diagonal cells are worse than the diagonal at equal total radius.

# %%
sp, bt, x, y = metrics(load("exp3_generalization"), "g_d", "g_c")
surface_pair(sp, bt, x, y, "designer generalization", "controller generalization",
             "Experiment 3 — Generalization").show()

# %% [markdown]
# **What this measures.** The matching-radii hypothesis. Generalization `g` is the radius of the
# Gaussian kernel weighting how far a sample is dragged toward its local peak, i.e. how far a climb
# carries from where that optimizer currently sits. The controller measures that distance *jointly*
# over (design, action), so a designer ranging beyond the controller's radius should have its good
# designs played badly.

# %%
# Is the diagonal actually better? Compare matched radii against mismatched ones of equal sum.
z = load("exp3_generalization")
best = torch.as_tensor(z["best"]).squeeze().mean(-1).numpy()
g = z["axis_g_d"]
lg = np.log10(g)
S = lg[:, None] + lg[None, :]                    # total radius (log)
D = np.abs(lg[:, None] - lg[None, :])            # mismatch
bins = np.digitize(S, np.quantile(S, np.linspace(0, 1, 7)[1:-1]))
rows = []
for b in np.unique(bins):
    m = bins == b
    matched, mismatched = best[m & (D <= 0.4)], best[m & (D >= 1.2)]
    if matched.size and mismatched.size:
        rows.append((f"{S[m].mean():+.2f}", matched.mean(), mismatched.mean(),
                     matched.mean() - mismatched.mean()))
print(f"{'total radius':>12} {'matched':>9} {'mismatch':>9} {'penalty':>9}")
for r in rows:
    print(f"{r[0]:>12} {r[1]:9.4f} {r[2]:9.4f} {r[3]:+9.4f}")
print("\npositive penalty = mismatched radii are worse at the same total radius")

# %% [markdown]
# **Metrics.** Configs are binned by **total radius** `log10 g_d + log10 g_c`; inside each bin,
# *matched* means `|log10 g_d - log10 g_c| <= 0.4` and *mismatch* means `>= 1.2`, and **penalty** is
# matched minus mismatch mean best fitness. Holding the total fixed is what isolates *mismatch*
# itself from simply having more generalization, so a positive penalty is the hypothesis confirmed.
#
# The 0.4 / 1.2 thresholds were set for a two-decade `g` sweep; `GEN` now spans 1.3 decades, so only 2 of 121 cells clear the mismatch cut and the table collapses to a single bin. Widen `GEN` before reading anything into it.

# %% [markdown]
# ## Path taken
#
# Each panel is its own window on the landscape, centred on that pair's starting `(mu_d, mu_a)` and
# widened just enough to contain everywhere it went — so the panel size itself reports how far the
# pair travelled, and the centre is always the origin. The surface is shaded by height where the
# pair never went and red where it did, white early to red late, so one static frame carries the
# ordering. Green points are the evaluated samples, subsampled.

# %%
import plotly.colors as pc

n = 224
HALF_MIN = 1.5      # never zoom in tighter than this, however still the pair sat
MARGIN = 1.12        # padding around the visited extent
DILATE = 3           # widen the rasterised path by this many cells, so it is visible

# One scale doing two jobs: [0, 0.5) shades the landscape by height, [0.5, 1] shades the path by
# recency. Splitting the range is what lets the mesh itself carry the trajectory.
base = pc.sample_colorscale("Blues_r", np.linspace(0.15, 0.95, 9))
hot = pc.sample_colorscale([[0, "#ff9100"], [1, "#7f0000"]], np.linspace(0, 1, 9))
PATH_SCALE = ([[0.5 * x, c] for x, c in zip(np.linspace(0, 1, 9), base)]
              + [[0.5 + 0.5 * x, c] for x, c in zip(np.linspace(0, 1, 9), hot)])


def dilate(v, times):
    """Grow the rasterised path by `times` cells, keeping the latest visit. -1 means unvisited."""
    for _ in range(times):
        v = np.maximum.reduce([
            v,
            np.pad(v, ((1, 0), (0, 0)), constant_values=-1.0)[:-1],
            np.pad(v, ((0, 1), (0, 0)), constant_values=-1.0)[1:],
            np.pad(v, ((0, 0), (1, 0)), constant_values=-1.0)[:, :-1],
            np.pad(v, ((0, 0), (0, 1)), constant_values=-1.0)[:, 1:],
        ])
    return v


z = load("paths")
names = [str(s) for s in z["names"]]

fig = make_subplots(rows=2, cols=2, specs=[[{"type": "surface"}] * 2] * 2,
                    subplot_titles=names, vertical_spacing=0.06, horizontal_spacing=0.04)
for i, name in enumerate(names):
    r, c = divmod(i, 2)
    d0, a0 = float(z["start_d"][i]), float(z["start_a"][i])
    # mu_a is recombined before step 0 is recorded, so prepend the true start: otherwise the path
    # visibly begins one jump away from the centre of its own window
    md_ = np.concatenate([[d0], z["mu_d"][:, i]])
    ma = np.concatenate([[a0], z["mu_a"][:, i]])

    # the window is centred on where this pair started, then widened until it contains
    # everywhere the pair went -- so the origin is always the middle of the panel
    half = max(np.abs(md_ - d0).max(), np.abs(ma - a0).max(), HALF_MIN) * MARGIN
    axd = np.linspace(d0 - half, d0 + half, n)
    axa = np.linspace(a0 - half, a0 + half, n)
    dd, aa = np.meshgrid(axd, axa, indexing="ij")
    Zw = L.f(torch.tensor(np.stack([dd, aa], -1), dtype=torch.float32)).numpy()

    # rasterise the trajectory, keeping the most recent visit in each cell
    t = np.linspace(0.0, 1.0, len(md_))
    gi = np.rint((md_ - axd[0]) / (2 * half) * (n - 1)).astype(int)
    gj = np.rint((ma - axa[0]) / (2 * half) * (n - 1)).astype(int)
    ok = (gi >= 0) & (gi < n) & (gj >= 0) & (gj < n)
    vis = np.full((n, n), -1.0)
    np.maximum.at(vis, (gi[ok], gj[ok]), t[ok])
    vis = dilate(vis, DILATE)

    # unvisited cells sit strictly below 0.5 so they can never read as path
    shade = 0.499 * (Zw - Zw.min()) / (Zw.max() - Zw.min())
    fig.add_trace(
        go.Surface(x=axd, y=axa, z=Zw.T,
                   surfacecolor=np.where(vis >= 0, 0.5 + 0.5 * vis, shade).T,
                   colorscale=PATH_SCALE, cmin=0.0, cmax=1.0, showscale=i == 0,
                   colorbar=dict(tickvals=[0.02, 0.47, 0.6, 0.98], len=0.4, y=0.78,
                                 ticktext=["value low", "value high", "path early", "path late"])),
        row=r + 1, col=c + 1)

    # actions are stored (step, P_d * P_a) against (step, P_d) designs, so each design has to be
    # repeated P_a times to pair a sample with the design it was actually played on
    a = z["a"][:, i]
    d = np.repeat(z["d"][:, i], a.shape[-1] // z["d"].shape[-1], axis=1).ravel()
    a = a.ravel()
    inside = (np.abs(d - d0) <= half) & (np.abs(a - a0) <= half)
    d, a = d[inside], a[inside]
    k = np.random.default_rng(0).choice(d.size, min(1200, d.size), replace=False)
    zs = L.f(torch.tensor(np.stack([d[k], a[k]], -1), dtype=torch.float32)).numpy() + 0.01
    fig.add_trace(go.Scatter3d(x=d[k], y=a[k], z=zs, mode="markers",
                               marker=dict(size=1.3, color="#2ecc40", opacity=0.4),
                               showlegend=False), row=r + 1, col=c + 1)

for s in ("scene", "scene2", "scene3", "scene4"):
    fig.update_layout(**{s: dict(xaxis_title="design", yaxis_title="action", zaxis_title="value",
                                 camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)))})
fig.update_layout(height=1000, margin=dict(l=0, r=0, t=60, b=0),
                  title="Path taken — surface shaded red where the pair went, white early to red "
                        "late; green = samples. Each panel is centred on its own starting point.")
fig.show()

# %% [markdown]
# **What this shows.** Four hand-picked configurations run to completion with trajectories recorded,
# to show *how* each archetype succeeds or fails rather than only how well.
#
# **Mismatched radii** The designers spread is increased, while the controller's generalization is decreased.
#
# **Encoding.** The surface colour is one scale split in half: the lower half shades unvisited mesh
# by landscape value, the upper half shades visited mesh by recency, white early to red late. The
# trajectory is rasterised onto the mesh and each cell keeps its *latest* visit, so a region the
# pair returned to reads as late. Each recorded mean is one dot, widened a few cells, so a gap in
# the trail means the mean jumped rather than walked. Panels are not on a common window — each is
# centred on its own start and sized to its own travel — so compare shapes across panels, never
# extents.

# %% [markdown]
# ## Experiment 4 — Sampling ratios
#
# Four artifacts, each answering one question. The evaluation budget is identical across ratios by
# construction, so any difference is structural rather than bought with compute.

# %%
z4 = load("exp4_ratios")
PARAMS = [str(p) for p in z4["param_order"]]
ratios = z4["ratios"]
best4 = z4["best"]          # (5,5,5,5,5,5, ratio, seed)
speed4 = z4["speed"]
axes4 = {p: z4[f"axis_{p}"] for p in PARAMS}
mean4 = best4.mean(-1)      # over seeds -> (..., ratio)
print("grid", best4.shape, " ratios", ratios)

# %% [markdown]
# ### 1. Main effects — how much does each param move the metric, at each ratio?

# %%
# marginalize over the other five params, then take the range the swept one spans, signed by
# whether its best setting sits above or below its worst on the axis
sens = np.zeros((len(PARAMS), len(ratios)))
for i in range(len(PARAMS)):
    other = tuple(j for j in range(len(PARAMS)) if j != i)
    marg = mean4.mean(axis=other)               # (5, ratio)
    sens[i] = (marg.max(0) - marg.min(0)) * np.sign(marg.argmax(0) - marg.argmin(0))

lim = np.abs(sens).max()
go.Figure(go.Heatmap(z=sens, x=[f"1:{r}" for r in ratios], y=PARAMS, colorscale="RdBu_r",
                     zmid=0.0, zmin=-lim, zmax=lim,
                     colorbar=dict(title="Δ best fitness<br>(+ = larger<br>setting wins)"))
          ).update_layout(
    title="Signed main effect of each parameter, by sampling ratio", height=420,
    xaxis_title="sampling ratio", yaxis_title="").show()

# %% [markdown]
# **Metric.** *Main effect* is how far mean best design fitness moves between a parameter's best and
# worst setting once the other five are averaged over — the range of its marginal, in fitness units.
# It is **signed** by which end of the axis the best setting sits on: positive means a larger value of
# that parameter scores better, negative means a smaller one does. Magnitude is unchanged by the sign,
# so zero still means the parameter does not matter at that ratio, and the row-to-row pattern still
# says which knobs start or stop mattering as the designer trades many noisy updates for few
# well-measured ones. Cell values are printed because `sig_c` saturates the colour range: four of the
# six rows would otherwise read as blank white.
#
# **Two readings to avoid.** First, a sign is only a direction, not a shape — `sig_d` is positive at
# every ratio but its marginal actually turns over around 0.6–1.6 and falls again, so "larger wins"
# here means "larger up to a point". Second, both controller rows darken as the ratio grows. That is
# mechanical rather than substantive: the budget is fixed, so at 1:2048 the designer gets only 25
# updates, and with the designer nearly frozen every controller parameter must matter more. This
# experiment therefore cannot test whether a high ratio substitutes for controller generalization —
# a starved designer produces the same signature.

# %% [markdown]
# ### 2. Where the optimum sits, as the ratio changes
#
# The headline figure. Uses the centroid of the top-N configs rather than the raw argmin, which jumps
# discontinuously under seed noise; the centroid honestly represents *the region of good settings*.

# %%
TOP_N = 64
flat = mean4.reshape(-1, len(ratios))
opt = np.zeros((len(PARAMS), len(ratios)))
for r in range(len(ratios)):
    idx = np.unravel_index(np.argsort(-flat[:, r])[:TOP_N], mean4.shape[:-1])
    for i, p in enumerate(PARAMS):
        v = axes4[p][idx[i]]
        lo, hi = axes4[p].min(), axes4[p].max()
        opt[i, r] = (np.mean(v) - lo) / (hi - lo)      # normalized to its own sweep range

fig = go.Figure()
for i, p in enumerate(PARAMS):
    fig.add_trace(go.Scatter(x=ratios, y=opt[i], mode="lines+markers", name=p))
fig.update_layout(title=f"Optimal setting vs sampling ratio (centroid of top {TOP_N})",
                  xaxis=dict(title="sampling ratio (controller updates per designer update)",
                             type="log", tickvals=ratios, ticktext=[f"1:{r}" for r in ratios]),
                  yaxis_title="normalized optimum", height=480,
                  legend=dict(title="parameter")).show()

# %% [markdown]
# **What this measures.** *Where* the good region sits, not how good it is. Each parameter's optimum
# is the mean of the top-64 configs' values for it, rescaled to 0-1 across its own sweep range so all
# six share one axis; a rising line means that parameter's best setting grows with the ratio.

# %% [markdown]
# ### 3. Strongest interactions — only the few worth looking at

# %%
# second-order interaction: how much the joint effect departs from the sum of the marginals
pairs = []
for i in range(len(PARAMS)):
    for j in range(i + 1, len(PARAMS)):
        other = tuple(k for k in range(len(PARAMS)) if k not in (i, j))
        joint = mean4.mean(axis=other).mean(-1)                     # (5,5)
        add = joint.mean(1)[:, None] + joint.mean(0)[None, :] - joint.mean()
        pairs.append((np.abs(joint - add).mean(), PARAMS[i], PARAMS[j], i, j))
pairs.sort(reverse=True)

# the spread and generalization axes are geomspace, so they need a log axis to plot evenly; the two
# exploration axes are linspace and must stay linear. detect rather than hardcode, so a change to the
# axis definitions in exp4_ratios.py cannot silently mislabel a panel.
def is_log(v):
    return v.min() > 0 and np.allclose(np.diff(np.log(v)), np.diff(np.log(v))[0])

def scale(p):
    v = axes4[p]
    t = dict(tickvals=v, ticktext=[f"{x:.2f}" for x in v])
    return dict(type="log", **t) if is_log(v) else t

fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{a} x {b}" for _, a, b, _, _ in pairs[:3]])
for c, (_, a, b, i, j) in enumerate(pairs[:3], start=1):
    other = tuple(k for k in range(len(PARAMS)) if k not in (i, j))
    joint = mean4.mean(axis=other).mean(-1)
    fig.add_trace(go.Heatmap(z=joint.T, x=axes4[a], y=axes4[b], colorscale=SEQ,
                             showscale=c == 3,
                             colorbar=dict(title="mean best<br>fitness")), row=1, col=c)
    fig.update_xaxes(title=a, **scale(a), row=1, col=c)
    fig.update_yaxes(title=b, **scale(b), row=1, col=c)
fig.update_layout(title="Three strongest parameter interactions", height=420).show()
for s, a, b, _, _ in pairs:
    print(f"{a:>6} x {b:<6}  {s:.4f}")

# %% [markdown]
# **Metric.** *Interaction strength* is the mean absolute departure of the joint fitness surface from
# the additive prediction built from its two marginals. Large means the pair must be tuned together;
# near zero means they can be tuned independently. The three strongest are plotted, all 15 printed.
# Spread and generalization axes are drawn on a log scale, matching the geometric spacing they were
# swept on; the two exploration axes were swept linearly and stay linear, so a panel may be mixed.

# %% [markdown]
# ### 4. Named archetypes across ratios

# %%
def nearest(p, v):
    return int(np.abs(axes4[p] - v).argmin())

import paths as P
fig = go.Figure()
for name, cfg in P.ARCHETYPES.items():
    idx = tuple(nearest(p, cfg[p]) for p in PARAMS)
    fig.add_trace(go.Scatter(x=ratios, y=mean4[idx], mode="lines+markers", name=name))
fig.update_layout(title="Archetypes across sampling ratios (nearest grid cell)",
                  xaxis=dict(title="sampling ratio", type="log", tickvals=ratios,
                             ticktext=[f"1:{r}" for r in ratios]),
                  yaxis_title="best design fitness", height=460,
                  legend=dict(title="archetype")).show()

# %% [markdown]
# **What this measures.** Whether a configuration's ranking is stable across the ratio axis or the
# best archetype changes with it. The four named configs from `paths.py` are each snapped to their
# nearest cell on Experiment 4's coarse grid, so these are approximations of those configs.
