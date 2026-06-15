from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from captain.utils import plots


def summarize_episodes(results: list[tuple[dict, float]]) -> dict[str, Any]:
    """Summarize info from all parallel episodes.

    Args:
        results: List of (info, total_reward) tuples.

    Returns:
        Dictionary with averaged metrics.
    """
    infos = [res[0] for res in results]

    # Average rewards by type
    all_rewards = np.array(
        [
            (
                i["rewards"].numpy()
                if isinstance(i["rewards"], torch.Tensor)
                else np.array(i["rewards"])
            )
            for i in infos
        ]
    )
    avg_rewards = np.mean(all_rewards, axis=0)

    # Average reward history
    all_histories = np.array([i["reward_history"] for i in infos])
    avg_history = np.mean(all_histories, axis=0)

    # Average protected cells
    avg_protected = np.mean([i["protected_cells"] for i in infos])

    avg_protection_matrix = np.mean(
        np.array([i["protection_matrix"] for i in infos]), axis=0
    )

    # Average species counts per extinction-risk category
    ext_risk_data = pd.DataFrame([i["extinction_risk"] for i in infos])

    return {
        "avg_rewards_by_type": avg_rewards,
        "avg_reward_history": avg_history,
        "avg_protected_cells": avg_protected,
        "avg_protection_matrix": avg_protection_matrix,
        "jaccard_indx": get_jaccard_indx(
            avg_protection_matrix, n=len(results), k=int(avg_protected)
        ),
        "extinction_risk": ext_risk_data.mean().to_dict(),
    }


def get_jaccard_indx(mean_mask: np.ndarray, n: int, k: int):
    # Calculates the average Intersection over Union (Jaccard Index) among perturbations
    # Assuming:
    # mean_mask: tensor of shape (cells,) with values between 0 and 1
    # n: number of perturbations
    # k: number of protected cells (budget)

    # 1. Calculate Average Pairwise Intersection
    sum_m2 = np.nansum(mean_mask ** 2)
    avg_inter = (n * sum_m2 - k) / (n - 1)

    # 2. Calculate Average Pairwise IoU
    # (Using the 2K - I rule because K is constant)
    avg_union = 2 * k - avg_inter
    iou = avg_inter / np.maximum(avg_union, 1.0)

    return iou


class TrainingLogger:
    def __init__(
            self,
            trainer,
            episode,
            results_dir: Path | str,
            log_file: Path | str,
            weights_file: Path | str,
            plot_freq: int = 1,
            figsize: tuple[int, int] = (6, 7),
    ):
        self.trainer = trainer
        self.plot_freq = plot_freq
        self.results_dir = results_dir
        self.log_path = results_dir / log_file
        self.weights_path = results_dir / weights_file
        self.plot_dir = self.results_dir / "training_plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize

        # Initialize CSV Header
        header = ["epoch", "running_reward", "avg_reward"]
        header += [f"r_{r._name}" for r in episode.rewards._reward_obj_list]
        header += ["protected_cells", "w_mean", "w_std", "lr", "sigma", "jaccard"]
        header += list(episode.env.species_extinction_risk.keys())

        os.makedirs(self.plot_dir, exist_ok=True)

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)

    def log_epoch(self, epoch, avg_reward, summary, epoch_time):
        # 1. Console Output
        print(
            f"Epoch {epoch:4d} | "
            f"reward: {avg_reward:8.2f} | "
            f"running: {self.trainer.running_reward:8.2f} | "
            f"jaccard: {summary['jaccard_indx']:8.2f} | "
            f"time: {epoch_time:.1f}s"
        )

        # 2. Plotting logic
        if epoch % self.plot_freq == 0:
            plots.plot_grid(
                summary["avg_protection_matrix"][0],
                title=f"Epoch {epoch}",
                outfile=self.plot_dir / f"epoch_{epoch}",
                rescale_figure=1.0,
                dpi=300,
                figsize=self.figsize,
            )

        # 3. CSV Logging
        w = self.trainer.get_weights()
        row = [epoch, self.trainer.running_reward, avg_reward]
        row += [r for r in summary["avg_rewards_by_type"]]
        row += [
            summary["avg_protected_cells"],
            np.mean(w),
            np.std(w),
            self.trainer.scheduler.alpha,
            self.trainer.scheduler.sigma,
            summary["jaccard_indx"],
        ]
        row += list(summary["extinction_risk"].values())

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(row)

        # 4. Save Weights (and potentially scheduler state)
        np.save(self.weights_path, w)
