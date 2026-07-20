from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


DELAYS = [0, 1, 3, 7, 14]

# Mean average-precision gains over raw released history from the block-causal delay sweep.
STATIC_GREEN = [0.0202, -0.0210, 0.0164, -0.0059, 0.0060]
ADAPTIVE_TWO_STAGE = [0.0889, 0.0430, 0.0641, 0.0596, 0.0671]
CROSS_FIT_LOGISTIC_TAIL = [0.0910, 0.0421, 0.0656, 0.0613, 0.0395]


def make_plot(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
        }
    )

    fig, ax = plt.subplots(figsize=(5.8, 3.4))

    ax.plot(
        DELAYS,
        STATIC_GREEN,
        color="0.65",
        linestyle="--",
        marker="o",
        markersize=5,
        linewidth=1.4,
        label="Standalone Green marginal",
    )
    ax.plot(
        DELAYS,
        ADAPTIVE_TWO_STAGE,
        color="black",
        linestyle="-",
        marker="s",
        markersize=5,
        linewidth=1.9,
        label="Green tail reranker",
    )
    ax.plot(
        DELAYS,
        CROSS_FIT_LOGISTIC_TAIL,
        color="0.30",
        linestyle="-.",
        marker="^",
        markersize=5,
        linewidth=1.7,
        label="Cross-fitted Green tail reranker",
    )

    ax.axhline(0.0, color="0.25", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Simulated label-release delay")
    ax.set_ylabel("Mean AP gain over raw history")
    ax.set_xticks(DELAYS)
    ax.set_xlim(min(DELAYS) - 0.5, max(DELAYS) + 0.5)
    ax.set_ylim(-0.03, 0.10)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    pdf_path = output_dir / "fig_delay_auc_sweep.pdf"
    png_path = output_dir / "fig_delay_auc_sweep.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the IEEE-CIS delay-sweep average-precision figure.")
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory where fig_delay_auc_sweep.pdf and fig_delay_auc_sweep.png are written.",
    )
    args = parser.parse_args()

    pdf_path, png_path = make_plot(Path(args.output_dir))
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
