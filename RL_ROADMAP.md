# RL Learning Roadmap

## Phase 0 — Foundations (read first, code second)

- [ ] Read [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) intro pages
- [ ] Skim Sutton & Barto Chapter 1-3 (free PDF online)
- [ ] Watch David Silver Lecture 1-2 (DeepMind YouTube playlist)
- [ ] Do the [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) Unit 1-2

**Key concepts to understand before writing any code:**
- State, action, reward, episode
- Policy (what the agent does) vs value function (how good a state is)
- Exploration vs exploitation tradeoff
- Why discount factor (gamma) exists

---

## Phase 1 — CartPole (get the pipeline working)

**Goal:** train an agent that balances a pole. Solve it in under an hour.

### Setup
```bash
pip install gymnasium stable-baselines3 tensorboard
```

### Minimal working example
```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=100_000)

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
```

### Tips
- PPO is the right default algorithm — robust, works on most problems
- 100k timesteps should solve CartPole. If it doesn't, something is wrong with your setup
- Launch TensorBoard to watch training: `tensorboard --logdir ./logs/`
- Try LunarLander-v2 next with the same code, ~500k timesteps

### What to observe
- Reward curve should climb steadily then plateau at max
- If reward is flat or oscillating wildly, your learning rate is wrong or env is broken
- Solved = average reward > 475 over 100 episodes

---

## Phase 2 — Flappy Bird

**Goal:** build a custom env wrapper, train an agent on a real game.

### Find the game
```bash
pip install flappy-bird-gymnasium
# https://github.com/markub3327/flappy-bird-gymnasium
# Pre-built gymnasium env — use this to focus on RL, not game integration
```

### Two observation modes to try
1. **Feature-based** — pipe positions, bird velocity (numbers). Easier to train, less interesting.
2. **Pixel-based** — raw screen frames. Much harder, needs CNN policy instead of MLP.

**Start with feature-based.**

### Key differences from CartPole
```python
from stable_baselines3 import PPO
import flappy_bird_gymnasium

env = gym.make("FlappyBird-v0", use_lidar=False)  # feature-based
model = PPO("MlpPolicy", env, verbose=1,
            n_steps=2048,          # more steps before update
            learning_rate=3e-4,
            tensorboard_log="./logs/")
model.learn(total_timesteps=3_000_000)  # needs more timesteps than CartPole
```

### Tips
- Expect 1-3M timesteps before decent play, 5-10M for consistent performance
- Reward is sparse (only when passing a pipe) — if training stalls, add +0.1 per frame survived
- Use `VecEnv` to run multiple envs in parallel — cuts wall time significantly:
  ```python
  from stable_baselines3.common.env_util import make_vec_env
  env = make_vec_env("FlappyBird-v0", n_envs=8)
  ```
- Save checkpoints every 500k steps — training can destabilize late
  ```python
  from stable_baselines3.common.callbacks import CheckpointCallback
  checkpoint = CheckpointCallback(save_freq=500_000, save_path="./checkpoints/")
  model.learn(total_timesteps=5_000_000, callback=checkpoint)
  ```

### Pixel-based (optional stretch goal)
```python
# Needs frame stacking so agent can perceive velocity from frames
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3 import PPO

env = gym.make("FlappyBird-v0", use_lidar=False, render_mode="rgb_array")
# wrap with frame stack, grayscale, resize
model = PPO("CnnPolicy", env, ...)  # CnnPolicy instead of MlpPolicy
```

---

## Phase 3 — Tetris

**Goal:** tackle sparse rewards, large state space, long-horizon planning.

### Find the game
```bash
pip install gymnasium-tetris
# or: https://github.com/Farama-Foundation/Minigrid (simpler grid alternative)
# Search GitHub for "tetris gymnasium" — several options exist
```

### Why Tetris is much harder
- Reward only comes when clearing lines — episodes can run thousands of steps with no reward
- Agent must plan several pieces ahead
- Bad early decisions compound — a messy board from move 10 may not kill you until move 200

### Reward shaping is mandatory
Don't use raw game score alone. Shape the reward:
```python
def compute_reward(self, prev_state, next_state):
    reward = 0
    reward += lines_cleared * 10          # primary goal
    reward -= holes_created * 2           # penalize covered empty cells
    reward -= bumpiness_increase * 0.5    # penalize uneven surface
    reward -= height_increase * 0.1       # penalize stacking high
    return reward
```

### Tips
- Use curriculum: start with only I and O pieces, add more piece types as agent improves
- PPO still works but needs much more tuning — consider also trying DQN
- Expect 10-50M timesteps before meaningful line clears
- Read papers: search "Tetris reinforcement learning reward shaping" — well-studied problem
- A rule-based baseline (classic Tetris heuristic AI) is easy to find and useful for comparison

### Algorithm comparison for Tetris
| Algorithm | Pros | Cons |
|---|---|---|
| PPO | Stable, good default | Slower to converge on sparse rewards |
| DQN | Good with discrete actions | Less stable, needs replay buffer tuning |
| A2C | Fast, simple | Less sample efficient than PPO |

Start with PPO, switch to DQN if PPO plateaus early.

---

## General Tips (apply everywhere)

### Debugging a stuck agent
1. Is the reward signal reaching the agent? Print mean reward every 10k steps
2. Is the env correct? Step through it manually and print observations
3. Is exploration happening? Check entropy in TensorBoard — if it drops to 0 early, agent is stuck
4. Try a lower learning rate (1e-4 instead of 3e-4)
5. Try more parallel envs

### Hyperparameters to tune (in order of impact)
1. `total_timesteps` — almost always too low on first try
2. `n_envs` — more parallel envs = faster wall time
3. `learning_rate` — 3e-4 is a good default, try 1e-4 if unstable
4. `n_steps` — how many steps collected before each policy update
5. `gamma` (discount factor) — lower for short episodes, higher for long ones

### TensorBoard metrics to watch
- `rollout/ep_rew_mean` — average episode reward (the main signal)
- `train/entropy_loss` — should stay negative; if near 0 agent stopped exploring
- `train/policy_loss` — should fluctuate but trend toward 0
- `train/value_loss` — should decrease over time

### Folder structure suggestion
```
rl-project/
  envs/           # custom env wrappers
  models/         # saved model checkpoints
  logs/           # tensorboard logs
  notebooks/      # experiments and analysis
  train.py        # main training script
  evaluate.py     # load checkpoint and watch agent play
```

---

## Rough Timeline

| Phase | Timesteps | Wall Time (CPU) |
|---|---|---|
| CartPole solved | 100k | ~5 min |
| LunarLander solved | 500k | ~20 min |
| Flappy Bird decent | 3-5M | ~2-4 hrs |
| Tetris first line clears | 10-20M | ~8-16 hrs |

Use a GPU or run overnight for Tetris. `stable-baselines3` supports GPU automatically if PyTorch with CUDA is installed.
