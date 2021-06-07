import gym
from deepwatch.envs.deepwatch_env import DeepwatchEnv
from stable_baselines3 import DQN, A2C, DDPG, PPO, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.a2c import CnnPolicy
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


env = make_vec_env(DeepwatchEnv)
#check_env(env)
device = torch.device("cuda:0")
model = BehaviorCloneNet(12)
#model = CarModel()
model.to(device)
ckpt_path = './checkpoints/deepwatch_13.pth'
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#model = DQN('MlpPolicy', env, verbose=1).learn(5000)
#model = A2C(CnnModel, env, verbose=1)
#model_env = A2C('CnnPolicy', env, verbose=1)
#transform = Compose([
#    ToTensor(),
#    Normalize(mean=[0.485, 0.456, 0.406],
#                std=[0.229, 0.224, 0.225])
#])
#data = LogLoader('./logging/', 360, transform)
#loader = torch.utils.data.DataLoader(data, batch_size=32, 
#                                     shuffle=True, num_workers=16, 
##                                     pin_memory=True)
#model_env = model_env.pretrain(loader, n_epochs=100)
#model.load('deepwatch_aim_training')
#model.learn(100000)
#model.save('deepwatch_aim_training')

def scale_detection(detection, frame_size):
    game_height = 1440
    game_width = 2560
    detect = np.empty([2, 6, 2])
    scale_x = frame_size / float(game_width)
    scale_y = frame_size / float(game_height)
    for i, d in enumerate(detection[0][0]):
        det_x = d[0]
        det_y = d[1]
        det_x *= scale_x
        det_y *= scale_y
        det_x /= frame_size
        det_y /= frame_size
        detect[0][i] = [det_x, det_y]
    for i, d in enumerate(detection[0][1]):
        det_x = d[0]
        det_y = d[1]
        det_x *= scale_x
        det_y *= scale_y
        det_x /= frame_size
        det_y /= frame_size
        detect[1][i] = [det_x, det_y]
    return detect

obs = env.reset()
env.render()
n_steps = 1000
for step in range(n_steps):
    #action, _ = model.predict(obs, deterministic=True)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    frame_tensor = obs[0].squeeze(0).astype(np.float32)
    frame_tensor = transform_train(frame_tensor).unsqueeze(0).to(device)
    #frame_tensor = torch.from_numpy(obs[0].squeeze(0).astype(np.float32))
    #frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    detect_scaled = scale_detection(obs[1], 360)
    #detect_scaled = np.full((2,6,2,), 0)
    detect_tensor = torch.from_numpy(detect_scaled.flatten().astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        raw_actions = model(frame_tensor, detect_tensor)
        print("Step {}".format(step + 1))
        #print("Action: ", action)
        obs, reward, done, info = env.step(raw_actions.cpu())
        env.render()

    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break


