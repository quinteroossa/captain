"""Visualise the disturbance layer and stochastic event patterns.

All plots are masked to the NZ EEZ (58,315 valid cells where mask==1).
Heatmaps and histograms are saved as separate files for easy use in reports.

Run with:
    uv run python experiments/utils/plot_disturbance.py \
        --data-dir /path/to/captain3data
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pyperlin import FractalPerlin2D

import captain as cn

# Consistent style
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/disturbance_plots"))
    parser.add_argument("--coherence", type=int, default=8)
    return parser.parse_args()


def load_data(data_dir: Path):
    mask, _        = cn.data_loader.load_map(data_dir / "env_layers/area_mask.npy")
    risk_map, _    = cn.data_loader.load_map(data_dir / "env_layers/area_swept_disturbance.tif")
    future_risk, _ = cn.data_loader.load_map(data_dir / "env_layers/future_area_swept_disturbance.tif")
    eez_mask = mask == 1          # True for valid NZ EEZ cells only
    return mask, eez_mask, risk_map, future_risk


def masked(arr: np.ndarray, eez_mask: np.ndarray) -> np.ndarray:
    """Return array with non-EEZ cells set to NaN for plotting."""
    out = arr.astype(float).copy()
    out[~eez_mask] = np.nan
    return out


def make_noise_generator(risk_map: np.ndarray, coherence: int) -> FractalPerlin2D:
    padded_h = (risk_map.shape[0] // coherence + 1) * coherence
    padded_w = (risk_map.shape[1] // coherence + 1) * coherence
    return FractalPerlin2D(
        shape=(1, padded_h, padded_w),
        resolutions=[(coherence, coherence), (coherence, coherence)],
        factors=[0.5, 0.5],
    )


def savefig(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# Plot 1a — Risk map spatial heatmaps (EEZ only)
# =============================================================================

def plot_risk_map_spatial(risk_map, future_risk, eez_mask, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("area_swept_disturbance.tif — NZ EEZ cells only", fontsize=13)

    kw = dict(cmap="YlOrRd", origin="upper")

    im0 = axes[0].imshow(masked(risk_map, eez_mask), **kw)
    axes[0].set_title("Present disturbance")
    axes[0].set_xlabel("pixel column")
    axes[0].set_ylabel("pixel row")
    plt.colorbar(im0, ax=axes[0], label="disturbance value")

    im1 = axes[1].imshow(masked(future_risk, eez_mask), **kw)
    axes[1].set_title("Future disturbance")
    axes[1].set_xlabel("pixel column")
    plt.colorbar(im1, ax=axes[1], label="disturbance value")

    plt.tight_layout()
    savefig(fig, out_dir / "01a_risk_map_spatial.png")


# =============================================================================
# Plot 1b — Risk map value histograms (EEZ cells only)
# =============================================================================

def plot_risk_map_histograms(risk_map, future_risk, eez_mask, out_dir):
    present_vals = risk_map[eez_mask]
    future_vals  = future_risk[eez_mask]
    n_eez        = eez_mask.sum()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"area_swept_disturbance.tif — value distribution within NZ EEZ ({n_eez:,} cells)",
        fontsize=13
    )

    # Present — all EEZ cells
    axes[0, 0].hist(present_vals, bins=50, color="coral", edgecolor="k", linewidth=0.3)
    axes[0, 0].set_title(f"Present — all EEZ cells\nmean={present_vals.mean():.4f}, max={present_vals.max():.3f}")
    axes[0, 0].set_xlabel("disturbance value")
    axes[0, 0].set_ylabel("cell count")

    # Present — non-zero only
    nz = present_vals[present_vals > 0]
    axes[0, 1].hist(nz, bins=50, color="coral", edgecolor="k", linewidth=0.3)
    axes[0, 1].set_title(f"Present — non-zero only\n({len(nz):,} cells = {len(nz)/n_eez*100:.1f}% of EEZ)")
    axes[0, 1].set_xlabel("disturbance value")
    axes[0, 1].set_ylabel("cell count")

    # Future — all EEZ cells
    axes[1, 0].hist(future_vals, bins=50, color="steelblue", edgecolor="k", linewidth=0.3)
    axes[1, 0].set_title(f"Future — all EEZ cells\nmean={future_vals.mean():.4f}, max={future_vals.max():.3f}")
    axes[1, 0].set_xlabel("disturbance value")
    axes[1, 0].set_ylabel("cell count")

    # Future — non-zero only
    nz_f = future_vals[future_vals > 0]
    axes[1, 1].hist(nz_f, bins=50, color="steelblue", edgecolor="k", linewidth=0.3)
    axes[1, 1].set_title(f"Future — non-zero only\n({len(nz_f):,} cells = {len(nz_f)/n_eez*100:.1f}% of EEZ)")
    axes[1, 1].set_xlabel("disturbance value")
    axes[1, 1].set_ylabel("cell count")

    plt.tight_layout()
    savefig(fig, out_dir / "01b_risk_map_histograms.png")


# =============================================================================
# Plot 2a — Perlin noise sample heatmap (EEZ only)
# =============================================================================

def plot_noise_spatial(risk_map, eez_mask, noise_gen, out_dir):
    H, W = risk_map.shape
    raw  = noise_gen()[:, :H, :W].squeeze(0).numpy()
    norm = (raw - raw.min()) / (raw.max() - raw.min())   # shifted to [0,1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Perlin noise — raw vs normalised (one sample, EEZ masked)", fontsize=13)

    im0 = axes[0].imshow(masked(raw, eez_mask), cmap="RdYlBu", origin="upper")
    axes[0].set_title(f"Raw output\nrange [{raw.min():.2f}, {raw.max():.2f}]")
    axes[0].set_xlabel("pixel column")
    axes[0].set_ylabel("pixel row")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(masked(norm, eez_mask), cmap="RdYlBu", origin="upper", vmin=0, vmax=1)
    axes[1].set_title("Normalised to [0, 1]\n(after +0.5 shift or min-max)")
    axes[1].set_xlabel("pixel column")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    savefig(fig, out_dir / "02a_perlin_noise_spatial.png")


# =============================================================================
# Plot 2b — Perlin noise value histograms
# =============================================================================

def plot_noise_histograms(risk_map, eez_mask, noise_gen, out_dir):
    H, W  = risk_map.shape
    raw   = noise_gen()[:, :H, :W].squeeze(0).numpy()
    norm  = (raw - raw.min()) / (raw.max() - raw.min())

    raw_eez  = raw[eez_mask]
    norm_eez = norm[eez_mask]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Perlin noise value distribution within NZ EEZ", fontsize=13)

    axes[0].hist(raw_eez, bins=50, color="salmon", edgecolor="k", linewidth=0.3)
    axes[0].axvline(0, color="red", linestyle="--", label="0")
    axes[0].set_title(f"Raw output\nmean={raw_eez.mean():.3f}, std={raw_eez.std():.3f}")
    axes[0].set_xlabel("noise value")
    axes[0].set_ylabel("cell count")
    axes[0].legend()

    axes[1].hist(norm_eez, bins=50, color="steelblue", edgecolor="k", linewidth=0.3)
    axes[1].axvline(0.5, color="red", linestyle="--", label="0.5 (centre)")
    axes[1].set_title(f"Normalised [0, 1]\nmean={norm_eez.mean():.3f}, std={norm_eez.std():.3f}")
    axes[1].set_xlabel("noise value")
    axes[1].set_ylabel("cell count")
    axes[1].legend()

    plt.tight_layout()
    savefig(fig, out_dir / "02b_perlin_noise_histograms.png")


# =============================================================================
# Plot 3a — Event masks at different intensities (heatmaps)
# =============================================================================

def plot_event_masks_spatial(risk_map, eez_mask, noise_gen, out_dir):
    H, W = risk_map.shape
    raw  = noise_gen()[:, :H, :W].squeeze(0).numpy()
    norm = (raw - raw.min()) / (raw.max() - raw.min())

    intensities = [0.05, 0.1, 0.3, 0.5]
    fig, axes = plt.subplots(1, len(intensities), figsize=(16, 5))
    fig.suptitle(
        "Event masks at different intensity values (normalised noise, EEZ only)\n"
        "Red = cell disturbed this timestep",
        fontsize=13
    )

    for ax, intensity in zip(axes, intensities):
        event = (norm < (1.0 - risk_map) * intensity).astype(float)
        pct = event[eez_mask].mean() * 100
        ax.imshow(masked(event, eez_mask), cmap="Reds", origin="upper", vmin=0, vmax=1)
        ax.set_title(f"intensity = {intensity}\n{pct:.1f}% of EEZ hit")
        ax.set_xlabel("pixel column")

    axes[0].set_ylabel("pixel row")
    plt.tight_layout()
    savefig(fig, out_dir / "03a_event_masks_spatial.png")


# =============================================================================
# Plot 3b — % cells hit at each intensity (histogram / bar)
# =============================================================================

def plot_event_masks_histograms(risk_map, eez_mask, noise_gen, out_dir):
    H, W = risk_map.shape
    intensities = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    n_samples = 10

    pct_matrix = np.zeros((n_samples, len(intensities)))
    for s in range(n_samples):
        raw  = noise_gen()[:, :H, :W].squeeze(0).numpy()
        norm = (raw - raw.min()) / (raw.max() - raw.min())
        for j, intensity in enumerate(intensities):
            event = norm[eez_mask] < (1.0 - risk_map[eez_mask]) * intensity
            pct_matrix[s, j] = event.mean() * 100

    means = pct_matrix.mean(axis=0)
    stds  = pct_matrix.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(i) for i in intensities], means, yerr=stds,
           color="coral", edgecolor="k", capsize=5)
    ax.set_xlabel("intensity value")
    ax.set_ylabel("% EEZ cells disturbed")
    ax.set_title(
        f"Fraction of EEZ cells disturbed per intensity\n"
        f"(mean ± std over {n_samples} noise samples, normalised Perlin)"
    )
    ax.set_ylim(0, 60)
    plt.tight_layout()
    savefig(fig, out_dir / "03b_event_fraction_by_intensity.png")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    mask, eez_mask, risk_map, future_risk = load_data(args.data_dir)
    noise_gen = make_noise_generator(risk_map, args.coherence)

    n_eez = eez_mask.sum()
    print(f"\nEEZ valid cells : {n_eez:,} of {mask.size:,} total")
    print(f"Risk map (EEZ)  : mean={risk_map[eez_mask].mean():.4f}, "
          f"max={risk_map[eez_mask].max():.4f}, "
          f"non-zero={( risk_map[eez_mask]>0).sum():,} ({(risk_map[eez_mask]>0).mean()*100:.1f}%)")

    print("\nGenerating plots...")
    plot_risk_map_spatial(risk_map, future_risk, eez_mask, args.out_dir)
    plot_risk_map_histograms(risk_map, future_risk, eez_mask, args.out_dir)
    plot_noise_spatial(risk_map, eez_mask, noise_gen, args.out_dir)
    plot_noise_histograms(risk_map, eez_mask, noise_gen, args.out_dir)
    plot_event_masks_spatial(risk_map, eez_mask, noise_gen, args.out_dir)
    plot_event_masks_histograms(risk_map, eez_mask, noise_gen, args.out_dir)

    print(f"\nAll plots saved to: {args.out_dir}")
    print("\nFiles:")
    print("  01a — risk map spatial heatmaps (present + future)")
    print("  01b — risk map value histograms (EEZ cells only)")
    print("  02a — Perlin noise heatmap (raw vs normalised)")
    print("  02b — Perlin noise value histograms")
    print("  03a — event masks at intensity 0.05 / 0.1 / 0.3 / 0.5")
    print("  03b — % EEZ cells hit per intensity (10 noise samples)")


if __name__ == "__main__":
    main()
