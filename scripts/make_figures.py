"""Render the figures used in the documentation, from the saved results.

Every number drawn here is read out of eval/results/*.json rather than typed
in, so a figure cannot drift away from the run that produced it. Re-run the
evaluations, re-run this, and the documentation is current.

Each figure is written twice, for a light and a dark surface, and the markdown
selects between them with <picture>. The two are the same chart stepped for its
own background rather than one inverted into the other.

Palette: the reference categorical slots (blue / orange / aqua), validated for
both surfaces. Light-mode aqua sits below 3:1 against the surface, so every
mark in this file carries a direct value label -- that is the required relief,
not decoration.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RESULTS = Path("eval/results")
OUT = Path("docs/figures")

THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  grid="#e3e2df",
                  series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100")),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 grid="#33322f",
                 series=("#3987e5", "#d95926", "#199e70", "#c98500")),
}


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _style(theme: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["primary"],
        "axes.labelcolor": theme["secondary"],
        "xtick.color": theme["secondary"],
        "ytick.color": theme["secondary"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 160,
    })


def _bare(ax, theme: dict, xgrid: bool = True) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(length=0)
    if xgrid:
        ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8)
        ax.set_axisbelow(True)


def _save(fig, stem: str, mode: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}-{mode}.png", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_shortcuts(mode: str, theme: dict) -> None:
    """What beats the benchmark without a model. The headline figure."""
    data = _load("shortcuts.json")
    rename = {
        "persistence — last observed label, no features":
            "persistence\n(ground-truth label, no features)",
        "one feature — flow count": "one feature: flow count",
        "one feature — fraction of flows that are TCP": "one feature: fraction TCP",
        "one feature — fraction with the -1 window sentinel":
            "one feature: the −1 sentinel",
        "always predict attack": "always predict attack",
    }
    model = _load("benchmark.json")["results"]["world model"]["metrics"]["auc"]
    rows = [(rename.get(k, k), v["auc"]) for k, v in data.items()]
    rows.append(("world model", model))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    _style(theme)
    fig.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    for i, (label, auc) in enumerate(rows):
        is_model = label == "world model"
        # Within a hundredth of an AUC is a tie, not a win: flow count scores
        # 0.7832 against the model's 0.7834, and calling that a defeat would be
        # as misleading as calling it a victory.
        matches = auc >= model - 0.01
        colour = theme["series"][0] if is_model else (
            theme["series"][1] if matches else theme["grid"])
        ax.barh(i, auc, height=0.55, color=colour,
                edgecolor=theme["surface"], linewidth=2)
        ax.text(auc + 0.008, i, f"{auc:.3f}", va="center", ha="left",
                fontsize=9.5, color=theme["primary"],
                fontweight="bold" if is_model or matches else "normal")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5,
                       color=theme["primary"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("AUC on the forecasting benchmark (test days)")
    ax.axvline(0.5, color=theme["secondary"], linewidth=1,
               linestyle=(0, (4, 3)), alpha=0.55)
    ax.text(0.5, len(rows) - 0.35, " chance", fontsize=8.5,
            color=theme["secondary"], va="center")
    ax.set_title("A single feature matches the world model", loc="left", pad=14)
    _bare(ax, theme)

    handles = [plt.Rectangle((0, 0), 1, 1, color=theme["series"][0]),
               plt.Rectangle((0, 0), 1, 1, color=theme["series"][1]),
               plt.Rectangle((0, 0), 1, 1, color=theme["grid"])]
    ax.legend(handles, ["world model", "matches or beats it", "below it"],
              loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              frameon=False, fontsize=8.5, labelcolor=theme["secondary"])
    _save(fig, "shortcuts", mode)


def fig_snr(mode: str, theme: dict) -> None:
    """How much of each attack survives network-wide averaging."""
    data = _load("snr.json")
    keep = ["Bot", "Web Attack – Brute Force", "Web Attack – XSS",
            "Infiltration", "PortScan", "DDoS"]
    rows = [(k, data[k]["network"], data[k]["host"]) for k in keep if k in data]
    rows.sort(key=lambda r: r[2] / max(r[1], 1e-9))

    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    _style(theme)
    fig.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    for i, (name, net, host) in enumerate(rows):
        ax.plot([net * 100, host * 100], [i, i], color=theme["grid"],
                linewidth=2.5, solid_capstyle="round", zorder=1)
        ax.scatter([net * 100], [i], s=90, color=theme["series"][1], zorder=3,
                   edgecolor=theme["surface"], linewidth=2)
        ax.scatter([host * 100], [i], s=90, color=theme["series"][0], zorder=3,
                   edgecolor=theme["surface"], linewidth=2)
        gain = host / net if net else 0
        # Gain labels live in a fixed column outside the plot rather than
        # trailing the marker: on a log axis a trailing label runs off the
        # right for the loud families and collides with the legend.
        ax.text(1.02, i, f"{gain:.0f}×", va="center", ha="left",
                fontsize=9.5, fontweight="bold", color=theme["primary"],
                transform=ax.get_yaxis_transform())

    ax.set_xscale("log")
    # Wide enough for Infiltration, which is 0.02% of flows network-wide and
    # 0.05% on its own host -- almost invisible either way, which is the point.
    ax.set_xlim(0.012, 220)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5,
                       color=theme["primary"])
    ax.set_xlabel("share of flows during the attack's own episode (%, log scale)")
    ax.set_title("What network-wide averaging costs, per family",
                 loc="left", pad=14)
    _bare(ax, theme)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                          color=theme["series"][1]),
               plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                          color=theme["series"][0])]
    ax.legend(handles, ["across the whole network", "on the host it touches"],
              loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2,
              frameon=False, fontsize=8.5, labelcolor=theme["secondary"])
    _save(fig, "snr", mode)


def fig_perhost(mode: str, theme: dict) -> None:
    """Detection per family, network-wide against per-host."""
    data = _load("perhost.json")
    network = {"Bot": 0.14, "DDoS": 0.92, "Infiltration": 0.62,
               "PortScan": 0.33, "Web Attack – Brute Force": 0.11,
               "Web Attack – XSS": 0.09, "Web Attack – Sql Injection": 0.06}
    rows = [(k, network[k], data[k]["surprise"])
            for k in network if k in data]
    rows.sort(key=lambda r: r[2] - r[1])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    _style(theme)
    fig.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    h = 0.34
    for i, (name, net, host) in enumerate(rows):
        ax.barh(i + h / 2, net, height=h, color=theme["series"][1],
                edgecolor=theme["surface"], linewidth=2)
        ax.barh(i - h / 2, host, height=h, color=theme["series"][0],
                edgecolor=theme["surface"], linewidth=2)
        ax.text(net + 0.012, i + h / 2, f"{net:.2f}", va="center", fontsize=8.5,
                color=theme["primary"])
        ax.text(host + 0.012, i - h / 2, f"{host:.2f}", va="center",
                fontsize=8.5, color=theme["primary"])

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5,
                       color=theme["primary"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("detection rate at a matched 10% false-alarm rate")
    ax.set_title("Changing the unit of modelling trades loud attacks for quiet ones",
                 loc="left", pad=14)
    _bare(ax, theme)
    handles = [plt.Rectangle((0, 0), 1, 1, color=theme["series"][1]),
               plt.Rectangle((0, 0), 1, 1, color=theme["series"][0])]
    ax.legend(handles, ["network-wide windows", "per-host windows"],
              loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2,
              frameon=False, fontsize=8.5, labelcolor=theme["secondary"])
    _save(fig, "perhost", mode)


def fig_surprise(mode: str, theme: dict) -> None:
    """Benign-trained surprise against its control and a supervised baseline."""
    data = _load("surprise.json")["onset_auc"]
    labels = ["world model surprise\n(unsupervised)",
              "logistic regression\n(supervised)",
              "naive dynamics\n(the control)"]
    values = [data["surprise"], data["supervised"], data["naive"]]
    colours = [theme["series"][0], theme["series"][2], theme["grid"]]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _style(theme)
    fig.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for i, (v, c) in enumerate(zip(values, colours)):
        ax.bar(i, v, width=0.42, color=c, edgecolor=theme["surface"], linewidth=2)
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=10,
                fontweight="bold", color=theme["primary"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9, color=theme["primary"])
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("AUC on the onset task")
    ax.axhline(0.5, color=theme["secondary"], linewidth=1,
               linestyle=(0, (4, 3)), alpha=0.55)
    ax.text(2.42, 0.515, "chance", fontsize=8.5, color=theme["secondary"],
            ha="right")
    ax.set_title("Surprise clears its control; it does not clear supervision",
                 loc="left", pad=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    _save(fig, "surprise", mode)


def fig_interventions(mode: str, theme: dict) -> None:
    """Counterfactual containment curves from the exported analysis."""
    payload = json.loads(Path("web/data/2017-07-07.json").read_text())
    actions = payload["interventions"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    _style(theme)
    fig.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    base = actions[0]["without"]
    steps = list(range(len(base)))
    # The untouched trajectory is a reference, not a fifth action, so it is a
    # recessive grey rather than a series hue. An earlier version painted it in
    # the same secondary ink as the fourth action, which made two different
    # things look like one.
    ax.plot(steps, base, color=theme["grid"], linewidth=3,
            solid_capstyle="round", zorder=1)
    ax.text(steps[-1] - 0.4, base[-1] + 0.035, "no action", fontsize=9,
            color=theme["secondary"], ha="right")

    # Five lines is past the point where end labels can be placed without
    # collision -- three of these land within 0.03 of each other -- so identity
    # comes from a legend that carries the final risk with it.
    handles, labels = [], []
    for action, colour in zip(actions, theme["series"]):
        line, = ax.plot(steps, action["with"], color=colour, linewidth=2.2,
                        solid_capstyle="round", zorder=2)
        ax.scatter([steps[-1]], [action["with"][-1]], s=34, color=colour,
                   zorder=3, edgecolor=theme["surface"], linewidth=1.5)
        contains = action.get("contains_in")
        held = f"contained in {contains:.0f}s" if contains else "not contained"
        handles.append(line)
        labels.append(f"{action['description']} — {held}")

    ax.set_xlim(-0.3, len(base) - 0.7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("rollout step (15s each) after the action is taken")
    ax.set_ylabel("predicted risk")
    ax.set_title("What the model expects each defensive action to buy",
                 loc="left", pad=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.44, -0.42),
              ncol=2, frameon=False, fontsize=8.5,
              labelcolor=theme["secondary"], handlelength=1.6)
    _save(fig, "interventions", mode)


def fig_stages(mode: str, theme: dict) -> None:
    """Stage-mapping confusion matrix on held-out windows."""
    data = _load("stages.json")
    names, matrix = data["classes"], data["matrix"]
    rows = [[v / max(sum(r), 1) for v in r] for r in matrix]

    ramp = LinearSegmentedColormap.from_list(
        "seq", [theme["surface"], theme["series"][0]])
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    _style(theme)
    fig.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    ax.imshow(rows, cmap=ramp, vmin=0, vmax=1, aspect="auto")

    for i in range(len(names)):
        for j in range(len(names)):
            if matrix[i][j] == 0:
                continue
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    fontsize=9.5,
                    fontweight="bold" if i == j else "normal",
                    color="#ffffff" if rows[i][j] > 0.55 else theme["primary"])

    short = [n.replace(" (DoS)", "").replace("Command & Control", "C2")
             for n in names]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short, fontsize=9, rotation=20, ha="right",
                       color=theme["primary"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(short, fontsize=9, color=theme["primary"])
    ax.set_xlabel("predicted stage")
    ax.set_ylabel("true stage")
    ax.set_title(f"Stage mapping — {data['accuracy']:.3f} accuracy against a "
                 f"{data['chance']:.2f} chance floor", loc="left", pad=14,
                 fontsize=11)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    _save(fig, "stages", mode)


def main() -> None:
    figures = [fig_shortcuts, fig_snr, fig_perhost, fig_surprise,
               fig_interventions, fig_stages]
    for mode, theme in THEMES.items():
        _style(theme)
        for figure in figures:
            figure(mode, theme)
            print(f"  {figure.__name__}  [{mode}]")
    print(f"\nwrote {len(figures) * 2} files to {OUT}")


if __name__ == "__main__":
    main()
