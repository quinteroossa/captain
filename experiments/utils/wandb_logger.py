"""Weights & Biases logging helper for CAPTAIN experiments.

Usage in training scripts:
    from experiments.utils.wandb_logger import WandbLogger
    wb = WandbLogger(enabled=args.wandb, project="captain", name=args.run_name, config=cfg)
    # inside epoch loop:
    wb.log(epoch, avg_reward, summary, trainer)
    # after training:
    wb.finish()
"""

from __future__ import annotations


class WandbLogger:
    def __init__(self, enabled: bool, project: str, name: str, config: dict, group: str | None = None):
        self.enabled = enabled
        if not enabled:
            return

        import wandb
        wandb.init(project=project, name=name, config=config, group=group)

    def log(self, epoch: int, avg_reward: float, summary: dict, trainer) -> None:
        if not self.enabled:
            return

        import wandb

        ext_risk = {f"extinction_risk/{k}": v for k, v in summary["extinction_risk"].items()}

        wandb.log({
            "epoch": epoch,
            "reward/avg": avg_reward,
            "reward/running": trainer.running_reward,
            "jaccard": summary["jaccard_indx"],
            "protected_cells": summary["avg_protected_cells"],
            "lr": trainer.scheduler.alpha,
            "sigma": trainer.scheduler.sigma,
            **ext_risk,
        })

    def log_raw(self, data: dict) -> None:
        """Log arbitrary key-value pairs directly (for PPO and other scripts)."""
        if not self.enabled:
            return
        import wandb
        wandb.log(data)

    def finish(self) -> None:
        if not self.enabled:
            return
        import wandb
        wandb.finish()
