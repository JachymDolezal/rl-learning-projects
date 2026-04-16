"""
Fetch and analyze wandb metrics for minesweeper-rl runs.
Usage:
    uv run minesweeper/analyze.py                  # latest run
    uv run minesweeper/analyze.py --run <run_id>   # specific run
    uv run minesweeper/analyze.py --all            # all runs summary
"""

import argparse
import numpy as np
import wandb

WANDB_PROJECT = "minesweeper-rl"


def fetch_run(run_id=None):
    api = wandb.Api()
    if run_id:
        return api.run(f"{WANDB_PROJECT}/{run_id}")
    runs = api.runs(WANDB_PROJECT, order="-created_at")
    if not runs:
        print("No runs found.")
        return None
    return runs[0]


def analyze_run(run):
    print(f"\n{'='*60}")
    print(f"Run:      {run.name}  ({run.id})")
    print(f"State:    {run.state}")
    print(f"Steps:    {run.summary.get('global_step', 'N/A'):,}")
    print(f"Config:   {run.config}")
    print(f"{'='*60}\n")

    history = run.history(
        keys=[
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/entropy_loss",
            "train/value_loss",
            "train/policy_loss",
            "train/approx_kl",
        ],
        samples=500,  # downsample to 500 points for analysis
    )

    if history.empty:
        print("No history data found.")
        return

    print_metric_summary(history, "rollout/ep_rew_mean", "Episode Reward")
    print_metric_summary(history, "rollout/ep_len_mean", "Episode Length")
    print_metric_summary(history, "train/value_loss",    "Value Loss")
    print_metric_summary(history, "train/entropy_loss",  "Entropy Loss")
    print_metric_summary(history, "train/approx_kl",     "Approx KL")

    analyze_trend(history, "rollout/ep_rew_mean", "reward")
    analyze_trend(history, "rollout/ep_len_mean", "episode length")

    diagnose(history, run)


def print_metric_summary(history, key, label):
    if key not in history.columns:
        return
    col = history[key].dropna()
    if col.empty:
        return
    early  = col.iloc[:len(col)//5].mean()
    recent = col.iloc[-len(col)//5:].mean()
    print(f"{label}:")
    print(f"  min={col.min():.3f}  max={col.max():.3f}  "
          f"mean={col.mean():.3f}  std={col.std():.3f}")
    print(f"  early avg={early:.3f}  →  recent avg={recent:.3f}  "
          f"(change: {recent - early:+.3f})\n")


def analyze_trend(history, key, label):
    if key not in history.columns:
        return
    col = history[key].dropna()
    if len(col) < 10:
        return
    # fit a linear trend over the whole run
    x = np.arange(len(col))
    slope, _ = np.polyfit(x, col, 1)
    direction = "improving" if slope > 0 else "declining"
    print(f"Trend ({label}): {direction}  (slope={slope:.5f} per sample)")


def diagnose(history, run):
    print(f"\n--- Diagnostics ---")

    rew = history.get("rollout/ep_rew_mean", None)
    if rew is not None:
        rew = rew.dropna()
        recent = rew.iloc[-50:].mean() if len(rew) >= 50 else rew.mean()
        if recent > 50:
            print("  Reward: agent occasionally winning (>50 avg reward)")
        elif recent > 10:
            print("  Reward: agent surviving but rarely winning")
        else:
            print("  Reward: agent struggling, dying early most games")

    ep_len = history.get("rollout/ep_len_mean", None)
    if ep_len is not None:
        ep_len = ep_len.dropna()
        if len(ep_len) >= 2:
            early  = ep_len.iloc[:len(ep_len)//5].mean()
            recent = ep_len.iloc[-len(ep_len)//5:].mean()
            if recent > early * 1.2:
                print("  Episode length: growing — agent surviving longer over time")
            else:
                print("  Episode length: flat — agent not improving survival")

    entropy = history.get("train/entropy_loss", None)
    if entropy is not None:
        entropy = entropy.dropna()
        if not entropy.empty and entropy.iloc[-1] > -0.1:
            print("  Entropy: very low — agent may have stopped exploring")

    kl = history.get("train/approx_kl", None)
    if kl is not None:
        kl = kl.dropna()
        if not kl.empty and kl.max() > 0.1:
            print(f"  KL: spiked to {kl.max():.4f} — policy update was unstable at some point")

    print()


def all_runs_summary():
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT, order="-created_at")
    if not runs:
        print("No runs found.")
        return

    print(f"\n{'='*60}")
    print(f"{'Run':<20} {'State':<10} {'Steps':>12} {'Reward':>10} {'EpLen':>8}")
    print(f"{'='*60}")
    for run in runs:
        steps  = run.summary.get("global_step", 0)
        reward = run.summary.get("rollout/ep_rew_mean", float("nan"))
        ep_len = run.summary.get("rollout/ep_len_mean", float("nan"))
        print(f"{run.name:<20} {run.state:<10} {steps:>12,} {reward:>10.2f} {ep_len:>8.1f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Specific run ID")
    parser.add_argument("--all", action="store_true", help="Summarize all runs")
    args = parser.parse_args()

    if args.all:
        all_runs_summary()
    else:
        run = fetch_run(args.run)
        if run:
            analyze_run(run)
