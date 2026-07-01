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
    fig, ax = plt.subplots(figsize=(8.2, 2.25))
    ax.set_axis_off()

    y = 0.82
    width = 2.05
    height = 0.68
    xs = [0.15, 2.45, 4.75, 7.05]

    add_box(ax, (xs[0], y), "historical graph\n$G_{t^-}$", width, height)
    add_box(ax, (xs[1], y), "released history\n$H, D$", width, height)
    add_box(ax, (xs[2], y), "local Green solve\n$(L+D)s=DH$", width, height)
    add_box(ax, (xs[3], y), "score $x_t$\nbefore insertion", width, height)
    add_box(ax, (xs[3], 0.02), "after timestamp block\ncommit edges + releases", width, height)

    for i in range(3):
        ax.annotate(
            "",
            xy=(xs[i + 1] - 0.08, y + height / 2),
            xytext=(xs[i] + width + 0.08, y + height / 2),
            arrowprops=dict(arrowstyle="->", linewidth=1.4, color="black"),
        )
    ax.annotate(
        "",
        xy=(xs[3] + width / 2, 0.02 + height + 0.03),
        xytext=(xs[3] + width / 2, y - 0.03),
        arrowprops=dict(arrowstyle="->", linewidth=1.4, color="black"),
    )
    feedback_y = 0.42
    past_center = xs[0] + width / 2
    commit_left = xs[3] - 0.12
    ax.plot([commit_left, past_center], [feedback_y, feedback_y], color="0.35", linestyle="--", linewidth=1.1)
    ax.annotate(
        "",
        xy=(past_center, y - 0.03),
        xytext=(past_center, feedback_y),
        arrowprops=dict(arrowstyle="->", linewidth=1.1, color="0.35", linestyle="--"),
    )

    outer = FancyBboxPatch(
        (0.02, 0.72),
        9.28,
        0.98,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="0.35",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(outer)
    ax.text(4.66, 1.78, "frozen information available at $t^-$", ha="center", va="center", fontsize=9)

    ax.set_xlim(0, 9.45)
    ax.set_ylim(0, 1.95)
    fig.tight_layout(pad=0.4)

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
