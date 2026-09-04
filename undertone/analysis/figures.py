"""Paper figures.

matplotlib only -- no seaborn, no style sheet. Every figure is built from the
same row filter the tables use (`analysis.usable`), so a figure can never show a
truncated or unverified cell that the corresponding table excluded.

    fig2_signature   Category x Language per model, with a duration-only panel
                     alongside -- carries F1 (type dominates duration) and F3'
    fig3_fingerprint Stacked distractor choice per model per category -- carries
                     F2 (the salience prior is the mechanism) and F5 (it is
                     specific: C1 must not share the signature)
    fig4_ladder      L1-L4 per category -- carries F4 (perception intact,
                     retrieval is not)
    fig5_repetition  Salience-trap rate against how often the loud competitor was
                     repeated -- F2's dose-response
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..items import CATEGORIES, CATEGORY_LABEL, LANGS
from ..ladder import CONDITIONS
from ..scoring import accuracy, salience_trap
from .tables import usable

CHANCE = 0.25          # four options
ROLE_COLORS = {
    "correct": "#4c956c",
    "salience": "#c1121f",     # the headline failure mode gets the loud colour
    "recency": "#e09f3e",
    "absent": "#8d99ae",
}


def _fig(nrows=1, ncols=1, **kw):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(nrows, ncols, **kw)


def _save(fig, out_dir: str | Path, name: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def fig2_signature(rows: Sequence[dict], out_dir: str | Path,
                   condition: str = "L3") -> list[Path]:
    """Category x Language accuracy per model, beside a duration-only panel.

    The two panels are the F1 test read off one figure: if the spread across
    categories at a fixed band exceeds the spread across bands at a fixed
    category, type dominates duration and length was the weaker axis all along.
    """
    import numpy as np

    scoped = [r for r in usable(rows) if r["condition"] == condition]
    if not scoped:
        return []
    models = sorted({r["model_key"] for r in scoped})
    bands = sorted({r["duration_band"] for r in scoped})

    fig, axes = _fig(len(models), 2, figsize=(11, 2.4 * len(models)),
                     squeeze=False, gridspec_kw={"width_ratios": [3, 1]})

    for row_index, model in enumerate(models):
        cells = [r for r in scoped if r["model_key"] == model]

        grid = np.full((len(LANGS), len(CATEGORIES)), np.nan)
        for i, lang in enumerate(LANGS):
            for j, category in enumerate(CATEGORIES):
                subset = [r for r in cells if r["lang"] == lang and r["category"] == category]
                if subset:
                    grid[i, j] = accuracy(subset)

        ax = axes[row_index][0]
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels([f"{c}\n{CATEGORY_LABEL[c]}" for c in CATEGORIES], fontsize=8)
        ax.set_yticks(range(len(LANGS)))
        ax.set_yticklabels(LANGS)
        ax.set_title(f"{model} - accuracy by category x language ({condition})", fontsize=9)
        for i in range(len(LANGS)):
            for j in range(len(CATEGORIES)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03)

        # The duration-only panel: the axis every other long-audio benchmark
        # varies, shown at the same scale so the comparison is honest.
        ax = axes[row_index][1]
        values = [accuracy([r for r in cells if r["duration_band"] == b]) for b in bands]
        ax.plot([b / 60 for b in bands], values, marker="o", color="#264653")
        ax.axhline(CHANCE, ls=":", c="grey", lw=1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("band (min)", fontsize=8)
        ax.set_title("duration only", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    return _save(fig, out_dir, "fig2_signature")


def fig3_fingerprint(rows: Sequence[dict], out_dir: str | Path,
                     condition: str = "L3") -> list[Path]:
    """Which wrong answer each model reached for, per category.

    This is the figure the whole design exists to produce. Accuracy says a model
    failed; this says it failed by selecting the loud competing mention. F5 reads
    off the C1 column: if C1 shares the P1-P4 signature, the mechanism is not
    specific and the paper's claim weakens to "hard things are hard".
    """
    import numpy as np

    scoped = [r for r in usable(rows) if r["condition"] == condition and not r["is_null"]]
    if not scoped:
        return []
    models = sorted({r["model_key"] for r in scoped})

    fig, axes = _fig(1, len(models), figsize=(3.1 * len(models), 3.6), squeeze=False)
    order = ["correct", "salience", "recency", "absent"]

    for index, model in enumerate(models):
        ax = axes[0][index]
        bottom = np.zeros(len(CATEGORIES))
        for role in order:
            heights = []
            for category in CATEGORIES:
                cell = [r for r in scoped
                        if r["model_key"] == model and r["category"] == category]
                heights.append(
                    sum(1 for r in cell if r["role_chosen"] == role) / len(cell)
                    if cell else 0.0)
            heights = np.asarray(heights)
            ax.bar(CATEGORIES, heights, bottom=bottom, label=role,
                   color=ROLE_COLORS[role], edgecolor="white", linewidth=0.5)
            bottom += heights
        ax.axhline(CHANCE, ls=":", c="black", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(model, fontsize=9)
        ax.tick_params(labelsize=8)
        if index == 0:
            ax.set_ylabel("choice rate")
        if index == len(models) - 1:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.suptitle("Distractor-choice fingerprints - the dotted line is chance", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_fingerprint")


def fig4_ladder(rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    """L1 -> L4 per category, per model.

    A category that starts high at L1 and collapses at L3 was perceived and then
    lost -- that is RetrievalCost, and it is a different finding from a category
    that was never perceived at all. Truncated conditions are simply absent from
    the line rather than plotted as zero.
    """
    scoped = usable(rows)
    if not scoped:
        return []
    models = sorted({r["model_key"] for r in scoped})

    fig, axes = _fig(1, len(models), figsize=(3.1 * len(models), 3.4),
                     squeeze=False, sharey=True)
    cmap = {"P1": "#1d3557", "P2": "#457b9d", "P3": "#c1121f",
            "P4": "#4c956c", "C1": "#e09f3e"}

    for index, model in enumerate(models):
        ax = axes[0][index]
        for category in CATEGORIES:
            xs, ys = [], []
            for position, condition in enumerate(CONDITIONS):
                cell = [r for r in scoped if r["model_key"] == model
                        and r["category"] == category and r["condition"] == condition]
                if cell:
                    xs.append(position)
                    ys.append(accuracy(cell))
            if xs:
                ax.plot(xs, ys, marker="o", label=category, color=cmap[category])
        ax.axhline(CHANCE, ls=":", c="grey", lw=1)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS)
        ax.set_ylim(0, 1)
        ax.set_title(model, fontsize=9)
        ax.tick_params(labelsize=8)
        if index == 0:
            ax.set_ylabel("accuracy")
        if index == len(models) - 1:
            ax.legend(fontsize=7)

    fig.suptitle("Ladder: missing points are truncated conditions, not zeros", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_ladder")


def fig5_repetition(rows: Sequence[dict], out_dir: str | Path,
                    condition: str = "L3") -> list[Path]:
    """Salience-trap rate against how often the loud competitor was repeated.

    F2's dose-response: if the trap is a salience prior rather than noise, the
    rate should rise with repetition count. Needs ``repetition_count`` carried
    through from item provenance; returns nothing if the pack did not record it.
    """
    scoped = [r for r in usable(rows)
              if r["condition"] == condition and not r["is_null"]
              and r.get("repetition_count") is not None]
    if not scoped:
        return []

    fig, ax = _fig(figsize=(5.5, 3.6))
    for lang in LANGS:
        counts = sorted({r["repetition_count"] for r in scoped if r["lang"] == lang})
        if not counts:
            continue
        rates = [salience_trap([r for r in scoped if r["lang"] == lang
                                and r["repetition_count"] == c]) for c in counts]
        ax.plot(counts, rates, marker="o", label=lang)
    ax.axhline(CHANCE, ls=":", c="grey", lw=1)
    ax.set_xlabel("times the competing mention was repeated")
    ax.set_ylabel("salience-trap rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, "fig5_repetition")


def all_figures(rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    paths: list[Path] = []
    for builder in (fig2_signature, fig3_fingerprint, fig4_ladder, fig5_repetition):
        paths.extend(builder(rows, out_dir))
    return paths
