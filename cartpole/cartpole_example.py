import gymnasium as gym
from stable_baselines3 import PPO

TIMESTEP_RUNS = [10_000, 50_000, 100_000]


def train(timesteps):
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./logs/")
    model.learn(total_timesteps=timesteps, tb_log_name=f"PPO_{timesteps // 1000}k")
    model.save(f"cartpole_ppo_{timesteps // 1000}k")
    env.close()
    print(f"Done: {timesteps:,} timesteps")
    return model


def watch(model_path):
    env = gym.make("CartPole-v1", render_mode="human")
    model = PPO.load(model_path)
    obs, _ = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    # for timesteps in TIMESTEP_RUNS:
    #     train(timesteps)

    # Watch the best one (highest timesteps)
    watch(f"cartpole_ppo_10k")
