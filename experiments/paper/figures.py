"""Shared plotting primitives: the palette, the arm->colour contract, and the panel furniture.

`analysis.py` draws; this decides what things look like. Two rules here are load-bearing rather than
cosmetic, and both come from constraints the figures cannot satisfy on their own.

**Colour follows the ARM, never its position.** `ARM_COLOUR` maps an arm name to a hex, and the
control arm of every study takes gold: `tuned` in `baseline`, `both` in `clone`, `aux` in `aux`,
`single` in `backbone` (the shipped configuration the other two rungs are read against) and `full`
in `attention`. Filtering a study to a subset of arms must not repaint the survivors, which is why
this is a dict and not an index into a list.

**Light surface only, deliberately.** These are paper figures on white. The four-arm palette was
validated with the data-viz validator at `--pairs all` (the scatter forms need it) against the light
surface and PASSES; no four-hue dark set containing gold clears the all-pairs floors, so rather than
ship an unvalidated dark mode this commits to one surface and says so.

**Two validator WARNs are discharged by furniture, not ignored.** Red<->aqua sits at CVD dE 6.9,
inside the 6-8 band that is legal ONLY with secondary encoding; and gold and aqua fall below 3:1
contrast on the light surface, which triggers the relief rule. Both are answered by the same thing:
`direct_label` on every series, plus the numbers table each experiment carries. `add_ci` therefore
labels by default, and turning it off is a decision to break the palette's validation.

**Every panel names the floor it draws.** The grill settled that there is no shared floor
vocabulary -- four different quantities in these docs are called "noise floor" and only the
across-seed one gates an arm comparison -- so `caption` is required rather than optional and says
which one is on the page.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# ── tokens ────────────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e8e7e3"

GOLD, BLUE, AQUA, RED = "#eda100", "#2a78d6", "#1baf7a", "#e34948"
SEQ = "Viridis"                      # window index on the phase portraits: magnitude, one ramp

# The control arm of each study -- gold everywhere. See the module docstring.
CONTROL_ARM = {"baseline": "tuned", "aux": "aux", "clone": "both",
               "backbone": "single", "attention": "full"}

# Fixed by entity. `none` taking red across two studies is deliberate: it is the same idea (nothing
# preserved / nothing enabled) and a reader who learns it once carries it between experiments.
ARM_COLOUR = {
    "tuned": GOLD, "aux": GOLD, "both": GOLD, "single": GOLD, "full": GOLD,
    "none": RED, "kl_only": BLUE, "mse_only": AQUA,
    "split": BLUE, "decoupled": AQUA,
    "self_cls": BLUE, "self": AQUA,
}


def colour(arm: str) -> str:
    if arm not in ARM_COLOUR:
        raise KeyError(f"no colour for arm {arm!r} -- add it to ARM_COLOUR rather than cycling a "
                       f"palette, or the same arm changes colour between figures")
    return ARM_COLOUR[arm]


def rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


# ── figure furniture ──────────────────────────────────────────────────

BOTTOM = 176            # px reserved below the axes for x-title + legend + a two-line caption
LEGEND_PX = 62          # px below the plot area
CAPTION_PX = 104


def _paper(height: int, px: int) -> float:
    """A pixel offset below the plot area, as the paper fraction plotly wants.

    Offsets must be constant in PIXELS: a fixed paper fraction is a fraction of the PLOT, so the
    same `y=-0.30` that clears a 440px figure falls outside a 700px one's bottom margin and the
    caption is silently clipped. Which is exactly what happened to the two-row drift panel.
    """
    plot = max(height - (86 + BOTTOM), 60)
    return -px / plot


def style(fig: go.Figure, *, height: int = 420, title: str = "", showlegend: bool = True):
    """House layout: recessive axes, light surface, text in ink tokens rather than series colour."""
    fig._house_height = height
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=12, color=INK),
        title=dict(text=title, font=dict(size=15, color=INK), x=0.0, xanchor="left"),
        margin=dict(l=68, r=124, t=86 if title else 44, b=BOTTOM),
        showlegend=showlegend,
        # Legend BELOW the plot. Every series is also direct-labelled, so the legend is the
        # accessibility guarantee rather than the primary key -- and above the axes it collides with
        # the subplot titles and the boundary rule's annotation.
        legend=dict(orientation="h", yanchor="top", y=_paper(height, LEGEND_PX), x=0,
                    font=dict(color=INK_2), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor=GRID, ticks="outside", tickcolor=GRID,
                     tickfont=dict(color=INK_2), title_font=dict(color=INK_2))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor=GRID, ticks="outside", tickcolor=GRID,
                     tickfont=dict(color=INK_2), title_font=dict(color=INK_2))
    return fig


CAPTION_COLS = 150      # characters per line at 11px; conservative for the narrowest figure we emit


def caption(fig: go.Figure, text: str):
    """The floor this panel draws, named. Required: there is no shared floor vocabulary, so a figure
    that does not say which floor it shows is showing an unlabelled one.

    Wrapped here rather than by hand: an annotation does not wrap itself, so a long line silently
    runs off the right edge of the image and the half of the sentence that names the floor is the
    half that disappears.
    """
    import textwrap
    lines = [w for seg in text.split("<br>") for w in (textwrap.wrap(seg, CAPTION_COLS) or [""])]
    text = "<br>".join(lines)
    # Wrapping can turn a two-line caption into five; grow the bottom margin to match, or the tail
    # of it falls outside the image exactly as an unwrapped line ran off the right of it.
    extra = max(0, len(lines) - 2) * 17
    if extra:
        fig.update_layout(margin_b=BOTTOM + extra)
    y = _paper(getattr(fig, "_house_height", 420), CAPTION_PX)
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0, y=y, xanchor="left",
                       yanchor="top", showarrow=False, align="left",
                       font=dict(size=11, color=INK_MUTED))
    return fig


def synthetic_badge(fig: go.Figure, keys, rollup=None) -> go.Figure:
    """Mark a panel drawn from fabricated data, on its face, naming the arrays.

    Two independent kinds of fake and both get said out loud: SYNTHETIC arrays (no producer has run)
    and a FIXTURE rollup (real numbers, but re-labelled past runs standing in for this study's arms).
    A study can be one, both or neither, and now that real scrapes and fixtures share a directory it
    is no longer safe to infer either from context.
    """
    marks = []
    if rollup is not None and getattr(rollup, "is_fixture", False):
        marks.append("◆ FIXTURE ROLLUP — arms are re-labelled past runs, not this study")
    keys = sorted(keys)
    if keys:
        marks.append(f"◆ SYNTHETIC — {', '.join(keys)} have no producer yet")
    for i, text in enumerate(marks):
        fig.add_annotation(text=text, xref="paper", yref="paper", x=1.0,
                           y=1.13 + 0.045 * (len(marks) - 1 - i), xanchor="right",
                           showarrow=False, font=dict(size=11, color=RED))
    return fig


def _log_y(y):
    """Data-space y -> the log10 an annotation on a `type="log"` axis is positioned in.

    Plotly positions annotations in AXIS coordinates, which on a log axis are the exponent -- so a
    label handed y=0.15 lands at 10^0.15 and one handed y=1867 lands at 10^1867, dragging the axis
    with it. Both happened: the FK panel's `losses/fk` label sat two decades above its curve, and the
    breadth panel's put the axis at 1e52. Non-positive y has no place on a log axis; NaN drops the
    label rather than crashing the figure.
    """
    y = np.asarray(y, float)
    return np.where(y > 0, np.log10(np.where(y > 0, y, 1.0)), np.nan)


def direct_label(fig, x, y, text, col, *, row=None, column=None, dx=8, log=False):
    """End-of-series label in INK, beside the coloured line that carries identity.

    Mandatory, not decorative: the palette validates only with secondary encoding (the red/aqua CVD
    pair) and only under the relief rule (gold and aqua below 3:1 on this surface). Text wears a
    text token rather than the series colour, per the data-viz rule.

    `y` is always in DATA units; pass `log=True` when the target axis is `type="log"` and the
    conversion happens here, so no caller has to know plotly's annotation convention.
    """
    kw = {} if row is None else dict(row=row, col=column)
    if log:
        y = float(_log_y(y))
        if not np.isfinite(y):
            return fig
    fig.add_annotation(x=x, y=y, text=text, xanchor="left", yanchor="middle", xshift=dx,
                       showarrow=False, font=dict(size=11, color=INK), **kw)
    return fig


def add_ci(fig, x, m, lo, hi, *, arm, row=None, col=None, label=True, dash=None,
           legendgroup=None, showlegend=True, width=2, alpha=0.16):
    """Mean line + ribbon for one arm, taking BOUNDS rather than a half-width.

    Bounds, because `stats.nested_bands` returns an outer band that is not symmetric about the mean
    in general, and a half-width API silently symmetrises it.
    """
    c = colour(arm)
    kw = {} if row is None else dict(row=row, col=col)
    x = np.asarray(x, float)
    m = np.asarray(m, float)
    ok = np.isfinite(m)
    if not ok.any():
        return fig
    if lo is not None and hi is not None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        band = np.isfinite(m) & np.isfinite(lo) & np.isfinite(hi)
        if band.any():
            # `mode="lines"` is NOT optional: plotly defaults a trace to lines+markers below ~20
            # points, so a short axis (the 9-level ladder) drew DEFAULT-palette markers on every
            # CI polygon -- colours outside the validated palette, invisible on the 48-window panels.
            fig.add_trace(go.Scatter(
                x=np.concatenate([x[band], x[band][::-1]]),
                y=np.concatenate([hi[band], lo[band][::-1]]),
                mode="lines", fill="toself", fillcolor=rgba(c, alpha), line=dict(width=0),
                hoverinfo="skip", showlegend=False, legendgroup=legendgroup or arm), **kw)
    fig.add_trace(go.Scatter(
        x=x[ok], y=m[ok], mode="lines", name=arm, legendgroup=legendgroup or arm,
        showlegend=showlegend, line=dict(color=c, width=width, dash=dash),
        hovertemplate=f"{arm}: %{{y:.3g}}<extra></extra>"), **kw)
    if label:
        i = np.flatnonzero(ok)[-1]
        direct_label(fig, x[i], m[i], arm, c, row=row, column=col)
    return fig


def rule(fig, x, text=None, *, row=None, col=None, colr=INK_MUTED, dash="dot"):
    """A vertical reference line -- the pretrain->RL boundary, the ladder's T=1 mark, a threshold."""
    kw = {} if row is None else dict(row=row, col=col)
    fig.add_vline(x=x, line=dict(color=colr, width=1, dash=dash), **kw)
    if text:
        fig.add_vline(x=x, line=dict(color=colr, width=1, dash=dash),
                      annotation_text=text, annotation_position="top",
                      annotation=dict(font=dict(size=10, color=INK_MUTED)), **kw)
    return fig


def hline(fig, y, text=None, *, row=None, col=None, colr=INK_MUTED, dash="dot"):
    kw = {} if row is None else dict(row=row, col=col)
    fig.add_hline(y=y, line=dict(color=colr, width=1, dash=dash),
                  annotation_text=text, annotation_position="right",
                  annotation=dict(font=dict(size=10, color=INK_MUTED)), **kw)
    return fig


def label_series(fig, x, ys, arms, *, span, row=None, col=None, min_frac=0.062, rng=None, log=False):
    """End labels for several series at one x, pushed apart so near-equal arms stay readable.

    Direct labels are not optional here -- the palette validates only WITH secondary encoding -- so
    when four arms finish within a hair of each other the answer has to be to space the labels, not
    to drop them. `span` is the panel's own data range, because the natural alternative (the spread
    of these four values) collapses to zero in exactly the case that needs the gap most.

    On a `log=True` axis the whole push-apart is done in log space and `span` is ignored: a gap of
    6.2% of a range spanning three decades is most of the panel down at the floor and invisible at
    the top. Everything in and out of here is still data units.
    """
    ys = np.asarray(ys, float)
    if log:
        ys = _log_y(ys)
        rng = None if rng is None else tuple(float(_log_y(v)) for v in rng)
        span = (rng[1] - rng[0]) if rng is not None and np.isfinite(rng).all() \
            else np.nanmax(ys) - np.nanmin(ys)
    ok = np.flatnonzero(np.isfinite(ys))
    gap = min_frac * (span if np.isfinite(span) and span > 0 else 1.0)
    prev, placed = None, {}
    for i in ok[np.argsort(-ys[ok])]:                 # top down, each pushed below the last
        y = ys[i] if prev is None else min(ys[i], prev - gap)
        placed[i] = y
        prev = y
    # The downward pass can push the bottom label off the axis -- `both` vanished under the critic
    # panel's zero floor. Lift the whole stack back inside rather than dropping the label, since a
    # missing direct label is the one thing the palette's validation does not allow.
    if rng is not None and placed:
        lo, hi = rng
        low = min(placed.values())
        if low < lo:
            placed = {i: y + (lo - low) for i, y in placed.items()}
        high = max(placed.values())
        if high > hi:
            placed = {i: y - (high - hi) for i, y in placed.items()}
    for i, y in placed.items():
        # Back to data units for the one place that owns the axis convention.
        direct_label(fig, x, 10.0 ** y if log else y, arms[i], colour(arms[i]),
                     row=row, column=col, log=log)
    return fig


def arm_lines(fig, x, m, lo, hi, arms, *, row=None, col=None, showlegend=True, rng=None,
              label=True, dash=None, alpha=0.16, log=False):
    """The default multi-arm panel: one CI ribbon per arm, then ONE spaced label pass.

    `add_ci`'s own label is per-series and therefore blind to the others, which stacks four labels on
    top of each other whenever the arms finish close together -- which is the normal case in an
    ablation. Every multi-arm panel goes through here so the spacing is decided once, with all the
    arms in hand.
    """
    m = np.asarray(m, float)
    lo = None if lo is None else np.asarray(lo, float)
    hi = None if hi is None else np.asarray(hi, float)
    for i, arm in enumerate(arms):
        add_ci(fig, x, m[i], None if lo is None else lo[i], None if hi is None else hi[i],
               arm=arm, row=row, col=col, showlegend=showlegend, label=False, dash=dash,
               alpha=alpha)
    if not label:
        return fig
    ends = np.full(len(arms), np.nan)
    xend = np.nan
    for i in range(len(arms)):
        ok = np.flatnonzero(np.isfinite(m[i]))
        if ok.size:
            ends[i] = m[i][ok[-1]]
            xend = np.nanmax([xend, np.asarray(x, float)[ok[-1]]])
    r0, r1 = (rng if rng is not None
              else (np.nanmin(m if lo is None else lo), np.nanmax(m if hi is None else hi)))
    return label_series(fig, xend, ends, list(arms), span=r1 - r0, rng=(r0, r1), row=row, col=col,
                        log=log)
