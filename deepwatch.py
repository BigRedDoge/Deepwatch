import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
#from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.ppo import CnnPolicy
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.evaluation import evaluate_policy


from Deepwatch.envs.deepwatch_env import DeepwatchEnv

def main():
    env = DummyVecEnv([DeepwatchEnv])

    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)

    vec_env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
    print(mean_reward)

    model.learn(total_timesteps=10000, log_interval=4, learning_rate=0.0005)
    model.save("ppo_deepwatch")
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
    print(mean_reward)


    lstm_states = None
    obs = vec_env.reset()
    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, done, _, info = vec_env.step(action)
        print(info)
        episode_starts = done


if __name__ == '__main__':
    main()