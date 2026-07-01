from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


DELAYS = [0, 1, 3, 7, 14]

# Mean AUC-PR gains over raw released history from the block-causal delay sweep.
STATIC_GREEN = [0.0261, -0.0056, 0.0004, -0.0097, -0.0060]
ADAPTIVE_TWO_STAGE = [0.1103, 0.0589, 0.0563, 0.0580, 0.0622]
CROSS_FIT_LOGISTIC_TAIL = [0.1042, 0.0631, 0.0571, 0.0575, 0.0355]


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
        color="black",
        linestyle="-",
        marker="o",
        markersize=5,
        linewidth=1.6,
        label="Static Green marginal",
    )
    ax.plot(
        DELAYS,
        ADAPTIVE_TWO_STAGE,
        color="0.35",
        linestyle="--",
        marker="s",
        markersize=5,
        linewidth=1.6,
        label="Adaptive two-stage tail",
    )
    ax.plot(
        DELAYS,
        CROSS_FIT_LOGISTIC_TAIL,
        color="0.60",
        linestyle="-.",
        marker="^",
        markersize=5,
        linewidth=1.6,
        label="Cross-fit logistic tail",
    )

    ax.axhline(0.0, color="0.25", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Simulated label-release delay")
    ax.set_ylabel("Mean AUC-PR gain over raw history")
    ax.set_xticks(DELAYS)
    ax.set_xlim(min(DELAYS) - 0.5, max(DELAYS) + 0.5)
    ax.set_ylim(-0.02, 0.12)
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
    parser = argparse.ArgumentParser(description="Reproduce the IEEE-CIS delay-sweep AUC-PR figure.")
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

