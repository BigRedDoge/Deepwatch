import sys
import cv2
import gym
import mss
import numpy as np
import pytesseract
import win32api
import win32con
from gym import error, spaces, utils
from gym.envs.classic_control import rendering
from gym.spaces import Box, Discrete, Tuple
from gym.utils import seeding
from PIL import Image
from pynput.keyboard import Controller, Key
import mouse
#import autoit
from pywinauto.application import Application
import pywinauto
import pyautogui

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


class DeepwatchEnv2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DeepwatchEnv2, self).__init__() 
        
        self.input_width = 2560
        self.input_height = 1440
        self.screen_width = 256
        self.screen_height = 144
        self.imsize = 360
        self.yolo_size = 256

        self.frame = self.get_screen()

        self.y_screen_pos = 0
        self.step_count = 0
        
        self.prev_detections = []
        self.skip_frame = False

        self.consec_shots = 0
        self.consec_miss = 0

        self.device = torch.device("cuda:0")
        weights = r'c:\Users\Sean\Desktop\Tradebot\Deepwatch\yolov5\runs\exp41\weights\best.pt'
        self.model = attempt_load(weights, map_location=self.device)                                          
        self.model = self.model.half()

        self.app = Application().connect(path=r'C:\Program Files (x86)\Overwatch\_retail_\Overwatch.exe')
        
        #n_actions = 8
        #self.action_space = spaces.Discrete(n_actions)
        #self.action_space = spaces.Box(np.array((0, 2560)), np.array((0, 1440)))
        #self.action_space = spaces.Box(np.array([0, 0, 0]), np.array([3, 2560, 1440]), dtype=np.float32)
        #self.action_space = spaces.MultiDiscrete([4, 2560, 1440])
        #self.action_space = spaces.MultiDiscrete([5, 3, 3])
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype="float32")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        #self.action_space = spaces.MultiDiscrete([200, 200, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        #self.observation_space = Tuple((spaces.Box(low=0, high=1, shape=(self.imsize, self.imsize, 3), dtype=np.uint8), spaces.Box(low=0, high=1, shape=(                                                                                                                      2, 6, 2), dtype=np.float32)))
        #self.observation_space = spaces.Tuple([spaces.Box(low=0, high=255, shape=(self.imsize, self.imsize, 3), dtype=np.uint8), \
        #    spaces.Box(low=0, high=1, shape=(2,6,2,), dtype=np.uint8)])
        #self.observation_space = spaces.Box(low=-self.screen_width / 2, high=self.screen_width / 2, shape=(2, 6, 2), dtype=np.uint8)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2, 6, 4), dtype=np.float32)
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
        
        classify = self.classification()
        obs = np.full((2,6,4), 0)
        if classify:
            #print("Enemies in LOS")
            enemy = 0
            team = 0
            #closest = None
            #closest_dist = None
            for det in classify:
                #print(det)
                reward += 1
                tl_x = det[0][0] / 0.01
                tl_y = det[0][1] / 0.01
                br_x = det[1][0] / 0.01
                br_y = det[1][1] / 0.01
                #cv2.rectangle(screen, (tl_x, tl_y), (br_x, br_y), (255, 0, 0), -1)
                mid_x = int((tl_x + br_x) / 2)
                mid_y = int(((tl_y + br_y) / 2) + 25)
                #try:
                if det[2] == 0:
                    obs[0][enemy] = [tl_x, tl_y, br_x, br_y] #[mid_x, mid_y]
                    obs[0][enemy] = [(p - (self.screen_width / 2)) / self.screen_width for p in obs[0][enemy]]
                    #print(obs)
                    enemy += 1
                elif det[2] == 1:
                    #obs[team] = [mid_x, mid_y]
                    obs[1][team] = [tl_x, tl_y, br_x, br_y] #[mid_x, mid_y]
                    obs[1][team] = [(p - (self.screen_width / 2)) / self.screen_width for p in obs[1][team]]
                    team += 1
                #except:
                #    print("error?")
                #print("x:", x)
                #print("tl_x:", tl_x)
                #print("br_x:", br_x)
                if x >= tl_x and x <= br_x:# and y >= tl_y and y <= br_y:
                    self.consec_shots += 1
                    self.consec_miss = 0
                    reward += 10 * self.consec_shots
                    #print("Enemy Spotted")
                    if mouse_left:
                        reward += 50 * self.consec_shots
                        
    
                """
                center_x = int(self.screen_width / 2)
                center_y = int(self.screen_height / 2)
                dist_x = center_x - mid_x
                dist_y = center_y - mid_y
                obs[i] = [dist_x, dist_y]
                dist = int(math.sqrt(((center_x - mid_x) ** 2) + ((center_y - mid_y) ** 2)))
                if not closest or dist < closest_dist:
                    closest = [mid_x, mid_y]
                    closest_dist = dist
                """
                

                """
                if closest:
                    print("Target Aquired: ({}, {})".format(closest[0], closest[1]))
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, closest[0] - x_focus, closest[1] - y_focus)
                    #time.sleep(0.1)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
                    #time.sleep(0.1)
                    #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, x-closest[0], y-closest[1])
                """
                """
                    if x + x_focus >= tl_x and x + x_focus <= br_x:
                        if y + y_focus >= tl_y and y + y_focus <= br_y:
                            reward += 100
                            #done = True
                            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
                            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
                            print("Enemy Spotted")

                    if x >= tl_x and x <= br_x:
                        print("enemy spotted")
                        if y >= tl_y and y <= br_y:
                            reward += 500
                            print("enemy spotted")
                """
                
        else:
            self.consec_miss = 1
            reward -= 3 * self.consec_miss
            self.consec_shots = 0
            if mouse_left:
                reward -= 2
        
        #cv2.imshow('frame', screen)
        #if cv2.waitKey(1): pass
        #print(obs)
        self.step_count += 1
        #print(reward)
        #screen = screen.astype(np.float32) 
        return obs, reward, done, {}
        #return obs, reward, done, {}

    def reset(self):
        #keyboard = Controller()
        #keyboard.press('f')
        #keyboard.release('f')
        #screen = self.get_screen()
        #screen = screen.astype(np.float32) 
        screen = self.resize(self.frame)
        #screen /= 255.0
        obs = np.full((2,6,4), 0)
        #obs = np.full((12,2), 0)
        #return (screen, obs)
        return obs
        #return self.observation()

    def render(self, mode='human'):
        #frame = self.get_screen()
        cv2.imshow('frame', self.frame)
        if cv2.waitKey(1): pass

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
        return screen
        """
        frame = pyautogui.screenshot()
        frame = np.array(frame)
        screen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return screen

    def resize(self, screen, size=(256, 144)):
        screen = cv2.resize(screen, size, interpolation=cv2.INTER_AREA)
        return screen
    
    def classification(self):
        self.frame = self.get_screen()
        im = self.resize(self.frame)
        cv2.imwrite('frame.png', im)
        dataset = LoadImages('frame.png', img_size=self.yolo_size)
        #img = torch.zeros((1, 3, self.imsize, self.imsize), device=self.device)
        #_ = self.model(img.half(), augment=False)
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
                            c1, c2 = (int(xyxy[0] * (self.yolo_size / self.input_width)), int(xyxy[1] * (self.yolo_size / self.input_height))), \
                                (int(xyxy[2] * (self.yolo_size / self.input_width)), int(xyxy[3] * (self.yolo_size / self.input_height)))
                            detections.append([c1, c2, int(cls)])
        #print(detections)
        return detections


    def observation(self):
        classify = self.classification()
        obs = np.full((6,2), -1)                                              
        if classify:
            i = 0
            for det in classify:
                tl_x = det[0][0]
                tl_y = det[0][1]
                br_x = det[1][0]
                br_y = det[1][1]
                mid_x = int((tl_x + br_x) / 2)          
                mid_y = int(((tl_y + br_y) / 2) + 25)
                obs[i] = [mid_x, mid_y]
                i += 1      
        return obs
