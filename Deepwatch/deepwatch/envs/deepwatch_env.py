import sys
import cv2
import gym
import mss
import numpy as np
import pytesseract
import win32api
import win32con
import pyautogui
from gym import error, spaces, utils
from gym.envs.classic_control import rendering
from gym.spaces import Box, Discrete, Tuple
from gym.utils import seeding
from PIL import Image
from pynput.keyboard import Controller, Key
import mouse
#import autoit
import pywinauto
from pywinauto.application import Application

sys.path.insert(1, r'C:\Users\Sean\Desktop\Tradebot\Deepwatch\yolov5')
import math
import random
import time

import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import (apply_classifier, check_img_size,
                           non_max_suppression, plot_one_box, scale_coords,
                           set_logging, strip_optimizer, xyxy2xywh)
from utils.torch_utils import load_classifier, select_device, time_synchronized


class DeepwatchEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DeepwatchEnv, self).__init__() 
        
        self.input_width = 2560
        self.input_height = 1440
        self.screen_width = 256
        self.screen_height = 144
        self.imsize = 256
        self.yolo_size = 256

        self.frame = self.get_screen()

        self.y_screen_pos = 0
        self.step_count = 0
        self.total_timesteps = 0

        self.prev_detections = []
        self.skip_frame = False

        self.device = torch.device("cuda:0")
        weights = r'c:\Users\Sean\Desktop\Tradebot\Deepwatch\yolov5\runs\exp41\weights\best.pt'
        self.model = attempt_load(weights, map_location=self.device)                                          
        self.model = self.model.half()

        self.app = Application().connect(path=r'C:\Program Files (x86)\Overwatch\_retail_\Overwatch.exe')
        self.keyboard = Controller()

        self.start_run = True

        self.shots_hit = 0
        self.detect_mem = []
        
        #n_actions = 8
        #self.action_space = spaces.Discrete(n_actions)
        #self.action_space = spaces.MultiDiscrete([5, 3, 3])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.move_btn_memory = { "w": 0, "a": 0, "s": 0, "d": 0 }
                

    def step(self, action):
        reward = 0
        mouse_x = int(action[0] * (self.input_width / 2))
        mouse_y = int(action[1] * (self.input_height / 2))
        mouse_left = action[2]
        #mouse_right = action[3]
        w = action[3]
        a = action[4]
        s = action[5]
        d = action[6]
        #e = action[8]
        #q = action[9]
        #shift = action[7]
        #space = action[8]
        
        done = False
        
        x = int(self.input_width / 2)  
        y = int(self.input_height / 2)

        if mouse_left >= 0:
            self.app.Overwatch.click()

        #if mouse_right >= 0:
            #self.app.Overwatch.right_click()
            #reward -= 1
        
        if w >= 0:
            pywinauto.keyboard.send_keys('{w down}')
            self.move_btn_memory['w'] = 1
            #reward += 3
        elif self.move_btn_memory['w'] == 1:
            pywinauto.keyboard.send_keys('{w up}')
            self.move_btn_memory['w'] = 0

        if a >= 0:
            pywinauto.keyboard.send_keys('{a down}')
            self.move_btn_memory['a'] = 1
        elif self.move_btn_memory['a'] == 1:
            self.move_btn_memory['a'] = 0
        
        if s >= 0:
            pywinauto.keyboard.send_keys('{s down}')
            self.move_btn_memory['s'] = 1
            reward -= 1
        elif self.move_btn_memory['s'] == 1:
            pywinauto.keyboard.send_keys('{s up}')
            self.move_btn_memory['s'] = 0
            
        if d >= 0:
            pywinauto.keyboard.send_keys('{d down}')
            self.move_btn_memory['d'] = 1
        elif self.move_btn_memory['d'] == 1:
            pywinauto.keyboard.send_keys('{d up}')
            self.move_btn_memory['d'] = 0
        
        #if e >= 0:
        #    pywinauto.keyboard.send_keys('{e}')
        
        #if q == 1:
        #    pywinauto.keyboard.send_keys('q')

        #if shift >= 0:
        #    pywinauto.keyboard.send_keys('{VK_SHIFT}')
        #    reward -= 1

        #if space >= 0:
        #    pywinauto.keyboard.send_keys('{SPACE}')
        #    reward -= 1     

        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouse_x * 0.33), 0, 0, 0) # - mouse_y) # | win32con.MOUSEEVENTF_ABSOLUTE,
        #self.app.Overwatch.move_mouse((x - mouse_x, y - mouse_y))   
        
        self.frame = self.get_screen() 
        screen = self.resize(self.frame)

        classify = self.classification()
        if classify:
            #print("Enemies in LOS")
            if len(self.detect_mem) > 0:
                if self.detect_mem[0] == 0:
                    self.detect_mem = []
            self.detect_mem.append(1)
            repeat_reward = lambda x: x * 2
            reward += repeat_reward(len(self.detect_mem) + 1)
            for det in classify:
                #reward += 25
                tl_x = det[0][0]
                tl_y = det[0][1]
                br_x = det[1][0]
                br_y = det[1][1]
                mid_x = int((tl_x + br_x) / 2)
                mid_y = int(((tl_y + br_y) / 2) + 25)
                
                dist = math.ceil(math.sqrt((mid_x - x)**2 + (mid_y - y)**2))
                dist_reward = lambda x: (-x / 40) + 20
                reward += dist_reward(dist)
                
                player_id = det[2]
                pad = 100
                if player_id == 0: # if enemy
                    if x >= tl_x - pad and x <= br_x + pad and y*2 >= tl_y - pad and y*2 <= br_y + pad:
                        reward += 100 * len(self.detect_mem) * 0.75
                        #print("Enemy Spotted")
                        self.shots_hit += 1
                        if mouse_left >= 0:
                            reward += 150 * len(self.detect_mem) * 0.75
                
        else:
            #reward -= 10
            if len(self.detect_mem) > 0:
                if self.detect_mem[0] == 1:
                    self.detect_mem = []
            self.detect_mem.append(0)
            repeat_penalty = lambda x: x * 2
            reward -= repeat_penalty(len(self.detect_mem) + 1)
            #if mouse_left >= 0:
            #    reward -= 1

        if self.shots_hit >= 100:
            done = True

        self.step_count += 1
        self.total_timesteps += 1

        #if self.step_count > 950:
        #    self.step_count = 0
        #    self.reset()    
        #self.frame = self.get_screen() 
        #screen = self.resize(self.frame)    
        screen = screen.astype(np.uint8)
        print("Step {} | Reward: {}".format(self.total_timesteps, reward))

        return screen, reward, done, {}

    def reset(self): 
        """       
        if not self.start_run:
            pywinauto.keyboard.send_keys('{ESC}')
            self.app.Overwatch.move_mouse_input((1280, 800)) 
            self.app.Overwatch.click()
            time.sleep(0.25)
            self.app.Overwatch.move_mouse_input((1400, 1220))
            self.app.Overwatch.click()
            time.sleep(6)
            pywinauto.keyboard.send_keys('{ESC}') 
            time.sleep(0.5)
            pywinauto.keyboard.send_keys('{ESC}')
            time.sleep(0.5)
            self.app.Overwatch.move_mouse_input((1260, 1230))
            time.sleep(0.25)
            self.app.Overwatch.click()
            time.sleep(0.25)
            self.app.Overwatch.move_mouse_input((1280, 1340)) 
            self.app.Overwatch.click()
            time.sleep(0.25)
            pywinauto.keyboard.send_keys('{f down}' '{f down}')
        """
        self.frame = self.get_screen()
        screen = self.resize(self.frame)
        screen = screen.astype(np.uint8)
        
        self.shots_hit = 0
        if self.start_run:
            self.start_run = False

        return screen

    def render(self, mode='human'):
        #cv2.imshow('frame', self.frame)
        #if cv2.waitKey(1): pass
        pass

    def get_screen(self, scale=100):
        """
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": self.input_width, "height": self.input_height}
            screen = np.array(sct.grab(monitor))
            width = int(screen.shape[1] * scale / 100)
            height = int(screen.shape[0] * scale / 100)
            dim = (width, height)
            screen = cv2.resize(screen, dim, interpolation = cv2.INTER_AREA)
            screen = screen[:,:,:3]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            screen = np.array(screen) 
        """
        frame = pyautogui.screenshot()
        frame = np.array(frame)
        screen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return screen

    def resize(self, screen, size=(256, 144)):
        screen = cv2.resize(screen, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        return screen
    
    def classification(self):
        im = self.resize(self.frame, (self.yolo_size, self.yolo_size))
        cv2.imwrite('frame.png', im)
        dataset = LoadImages('frame.png', img_size=self.yolo_size)
        #img = torch.zeros((1, 3, self.imsize, self.imsize), device=self.device)
        #=_ = self.model(img.half(), augment=False)
        detections = []
        for path, img, im0s, _ in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half()
            img /= 255.0                                                                                                            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img, augment=False)[0]
            conf_thresh = 0.4
            pred = non_max_suppression(pred, conf_thresh, 0.5)
            for i, det in enumerate(pred):
                p, im0 = path, im0s
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                            # bounding box of each detection, c1 top left c2 bottom right
                            c1, c2 = (int(xyxy[0] * (self.input_width / self.yolo_size)), int(xyxy[1] * (self.input_width / self.yolo_size))), \
                                (int(xyxy[2] * (self.input_width / self.yolo_size)), int(xyxy[3] * (self.input_width / self.yolo_size)))
                            detections.append([c1, c2, int(cls)])

        return detections


