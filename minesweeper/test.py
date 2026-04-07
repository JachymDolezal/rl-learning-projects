import gym_minesweeper
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

TIMESTEP_RUNS = [5_000_000]


class MineCNN(BaseFeaturesExtractor):
    """Small CNN suited for 16x16 minesweeper observations."""

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # channel-first (C, H, W)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


def train(timesteps):
    env = gym.make("Minesweeper-v0")
    policy_kwargs = dict(
        features_extractor_class=MineCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        tensorboard_log="./logs/",
        policy_kwargs=policy_kwargs,
        device="cuda",
    )
    model.learn(
        total_timesteps=timesteps,
        tb_log_name=f"PPO_{timesteps // 1000}k",
        progress_bar=True,
    )
    model.save(f"minesweeper_ppo_{timesteps // 1000}k")
    env.close()
    print(f"Done: {timesteps:,} timesteps")
    return model


def watch(model_path, step_delay=0.05, loss_pause=1.5):
    import time

    import pygame

    env = gym.make("Minesweeper-v0")
    model = PPO.load(model_path)
    inner = env.unwrapped
    wins, losses, timeouts = 0, 0, 0

    running = True
    while running:
        obs, _ = env.reset()
        done = False
        last_action = None
        last_reward = None

        while not done:
            env.render()
            # Check for window close
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            time.sleep(step_delay)
            action, _ = model.predict(obs)
            last_action = action
            obs, last_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if last_reward == inner.win_reward:
            wins += 1
            outcome = "WIN!"
        elif last_reward == inner.fail_reward:
            losses += 1
            outcome = "LOSE - hit a mine"
        else:
            timeouts += 1
            outcome = "Timeout"
        print(f"{outcome} | W:{wins} L:{losses} T:{timeouts}")

        # On loss: show the mine in red and pause briefly
        if last_reward == inner.fail_reward and last_action is not None:
            env.render()
            bs = inner.block_size
            rect = pygame.Rect(last_action[0] * bs, last_action[1] * bs, bs, bs)
            pygame.draw.rect(inner.screen, (220, 0, 0), rect)
            pygame.display.update()
            time.sleep(loss_pause)

    env.close()


if __name__ == "__main__":
    # for timesteps in TIMESTEP_RUNS:
    #     train(timesteps)

    # Watch the best one (highest timesteps)
    watch(f"minesweeper_ppo_2000k")
