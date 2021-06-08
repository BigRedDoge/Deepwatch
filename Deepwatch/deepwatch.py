from custom_arch import CustomCNN
import gym
from deepwatch.envs.deepwatch_env import DeepwatchEnv
from deepwatch.envs.bc_deepwatch_env import DeepwatchEnv2
from stable_baselines3 import DQN, A2C, DDPG, PPO, TD3, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env
#from stable_baselines3.td3.policies import CnnPolicy
from stable_baselines3.sac import CnnPolicy
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from model import BehaviorCloneNet, CarModel
from logloader import LogLoader
import time
from torchvision.transforms import Compose, ToTensor, Normalize
from custom_arch import CustomCNN, CustomActorCriticPolicy


env = make_vec_env(DeepwatchEnv2)

policy_kwargs = dict(
        features_extractor_class=CustomCNN
    )
#check_env(env)

n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#model = TD3(CnnPolicy, env, action_noise=action_noise, buffer_size=50000, verbose=1) # optimize_memory_usage=True
#model = SAC(CnnPolicy, env, buffer_size=50000, action_noise=action_noise, learning_rate=0.0005, tensorboard_log='./tensorboard', verbose=1)
#model = SAC.load("deepwatch_evolution_sac_7", env)
model = A2C(MlpPolicy, env, verbose=1, n_steps=5)#, policy_kwargs=policy_kwargs)
model.load("deepwatch_evolution_a2c_2")

for i in range(100):
    model.learn(total_timesteps=1000)
    model.save("deepwatch_evolution_a2c_3")
    print("Saved Checkpoint")

#model.learn(total_timesteps=10000)
#model.save("deepwatch_evolution")
#model.load("deepwatch_evolution")

obs = env.reset()
env.render()
n_steps = 1000
for step in range(n_steps):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break

