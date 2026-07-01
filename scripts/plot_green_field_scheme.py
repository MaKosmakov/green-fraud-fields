from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(ax, xy, text, width=2.15, height=0.7):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.06",
        linewidth=1.2,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)
    return box


def make_scheme(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 2.45))
    ax.set_axis_off()

    y_top = 1.08
    y_commit = 0.06
    width = 1.82
    height = 0.62
    xs = [0.32, 2.56, 4.80, 7.04]

    add_box(ax, (xs[0], y_top), "historical graph\n$G_{t^-}$", width, height)
    add_box(ax, (xs[1], y_top), "released history\n$H, D$", width, height)
    add_box(ax, (xs[2], y_top), "local Green solve\n$(L+D)s=DH$", width, height)
    add_box(ax, (xs[3], y_top), "score $x_t$\nbefore insertion", width, height)
    commit_width = 1.95
    commit_x = xs[3] + (width - commit_width) / 2
    commit_box = FancyBboxPatch(
        (commit_x, y_commit),
        commit_width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.06",
        linewidth=1.2,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(commit_box)
    commit_text_x = xs[3] + width / 2
    ax.text(commit_text_x, y_commit + height * 0.69, "after timestamp block", ha="center", va="center", fontsize=9)
    ax.text(commit_text_x, y_commit + height * 0.24, "commit edges + releases", ha="center", va="center", fontsize=9)

    for i in range(3):
        start_x = xs[i] + width + 0.04
        end_x = xs[i + 1] - 0.04
        ax.annotate(
            "",
            xy=(end_x, y_top + height / 2),
            xytext=(start_x, y_top + height / 2),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.25, color="black", mutation_scale=12),
        )
    ax.annotate(
        "",
        xy=(xs[3] + width / 2, y_commit + height + 0.04),
        xytext=(xs[3] + width / 2, y_top - 0.04),
        arrowprops=dict(arrowstyle="->", linewidth=1.4, color="black"),
    )
    feedback_y = y_commit + height / 2
    past_center = xs[0] + width / 2
    ax.plot([commit_x, past_center], [feedback_y, feedback_y], color="0.35", linestyle="--", linewidth=1.1)
    ax.annotate(
        "",
        xy=(past_center, y_top - 0.04),
        xytext=(past_center, feedback_y),
        arrowprops=dict(arrowstyle="->", linewidth=1.1, color="0.35", linestyle="--"),
    )

    outer = FancyBboxPatch(
        (0.18, 0.92),
        8.88,
        0.88,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="0.35",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(outer)
    ax.text(4.62, 1.95, "frozen information available at $t^-$", ha="center", va="center", fontsize=9)

    ax.set_xlim(0, 9.25)
    ax.set_ylim(0, 2.12)
    fig.tight_layout(pad=0.55)

    png_path = output_dir / "green_field_scheme.png"
    pdf_path = output_dir / "green_field_scheme.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the causal Green-field scoring scheme.")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    png_path, pdf_path = make_scheme(Path(args.output_dir))
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
