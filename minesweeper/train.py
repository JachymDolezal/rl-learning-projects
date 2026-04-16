import argparse
import math
import gym_minesweeper
import gymnasium as gym
import torch as th
import torch.nn as nn
import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from wandb.integration.sb3 import WandbCallback

TIMESTEP_RUNS = [10_000_000]

# Official minesweeper mine ratios:
#   Beginner:     9x9,  10 mines (~12%)
#   Intermediate: 16x16, 40 mines (~16%)
#   Expert:       30x16, 99 mines (~21%)
# We use ~15% as a sensible default across all sizes.
MINE_RATIO = 0.15


def board_config(cells):
    """Return (height, width, num_mines) for a square grid using official mine ratio."""
    num_mines = max(1, math.floor(cells * cells * MINE_RATIO))
    return cells, cells, num_mines


class MineCNN(BaseFeaturesExtractor):
    """CNN feature extractor for minesweeper observations."""

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


def make_env(height, width, num_mines):
    env = gym.make("Minesweeper-v0", height=height, width=width, num_mines=num_mines)
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    return env


def train(timesteps, height, width, num_mines):
    run = wandb.init(
        project="minesweeper-rl",
        config={
            "grid": f"{height}x{width}",
            "mines": num_mines,
            "mine_ratio": round(num_mines / (height * width), 2),
            "algorithm": "MaskablePPO",
            "policy": "CnnPolicy",
            "features_dim": 128,
            "timesteps": timesteps,
        },
        sync_tensorboard=True,
        monitor_gym=False,
    )

    env = make_env(height, width, num_mines)
    policy_kwargs = dict(
        features_extractor_class=MineCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = MaskablePPO(
        "CnnPolicy",
        env,
        verbose=0,
        tensorboard_log=f"./logs/{run.id}",
        policy_kwargs=policy_kwargs,
        device="cuda",
    )
    model.learn(
        total_timesteps=timesteps,
        tb_log_name=f"MaskPPO_{height}x{width}_{timesteps // 1000}k",
        progress_bar=True,
        callback=WandbCallback(verbose=0),
    )
    model.save(f"minesweeper_maskppo_{height}x{width}_{timesteps // 1000}k")
    env.close()
    run.finish()
    print(f"Done: {height}x{width} {num_mines} mines — {timesteps:,} timesteps")
    return model


def watch(model_path, height, width, num_mines, step_delay=0.5, loss_pause=1.5):
    import time
    import pygame

    env = make_env(height, width, num_mines)
    model = MaskablePPO.load(model_path)
    inner = env.unwrapped
    wins, losses, timeouts = 0, 0, 0

    running = True
    while running:
        obs, _ = env.reset()
        done = False
        last_action = None
        last_info = {}

        while not done:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            time.sleep(step_delay)
            masks = inner.action_masks()
            action, _ = model.predict(obs, action_masks=masks)
            last_action = action
            obs, _, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated

        outcome = last_info.get("outcome")
        if outcome == "win":
            wins += 1
            label = "WIN!"
        elif outcome == "mine":
            losses += 1
            label = "LOSE - hit a mine"
        else:
            timeouts += 1
            label = "Timeout"
        print(f"{label} | W:{wins} L:{losses} T:{timeouts}")

        if outcome == "mine" and last_action is not None:
            env.render()
            bs = inner.block_size
            row, col = divmod(int(last_action), inner.width)
            rect = pygame.Rect(row * bs, col * bs, bs, bs)
            pygame.draw.rect(inner.screen, (220, 0, 0), rect)
            pygame.display.update()
            time.sleep(loss_pause)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=16,
                        help="Board size (NxN). E.g. --cells 8 gives an 8x8 board.")
    parser.add_argument("--watch", type=str, default=None,
                        help="Path to saved model to watch instead of training.")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total timesteps (default: 10M).")
    args = parser.parse_args()

    height, width, num_mines = board_config(args.cells)
    print(f"Board: {height}x{width}  Mines: {num_mines}  "
          f"({round(num_mines / (height * width) * 100)}% density)")

    if args.watch:
        watch(args.watch, height, width, num_mines)
    else:
        runs = [args.timesteps] if args.timesteps else TIMESTEP_RUNS
        for timesteps in runs:
            train(timesteps, height, width, num_mines)
