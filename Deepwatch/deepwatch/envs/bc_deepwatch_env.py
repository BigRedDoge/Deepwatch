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
import autoit
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
        self.imsize = 360
        self.yolo_size = 256

        self.frame = self.get_screen()

        self.y_screen_pos = 0
        self.step_count = 0
        
        self.prev_detections = []
        self.skip_frame = False

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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        #self.action_space = spaces.MultiDiscrete([200, 200, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        #self.observation_space = Tuple((spaces.Box(low=0, high=1, shape=(self.imsize, self.imsize, 3), dtype=np.uint8), spaces.Box(low=0, high=1, shape=(                                                                                                                      2, 6, 2), dtype=np.float32)))
        self.observation_space = spaces.Tuple([spaces.Box(low=0, high=255, shape=(self.imsize, self.imsize, 3), dtype=np.uint8), \
            spaces.Box(low=0, high=1, shape=(2,6,2,), dtype=np.uint8)])
        #self.observation_space = spaces.Box(low=-self.screen_width / 2, high=self.screen_width / 2, shape=(6, 2), dtype=np.uint8)
                

    def step(self, action):
        #list_action = [int(x[1]) for x in action]
        #print(action)
        #print(action.shape)
        self.frame = self.get_screen() 
        screen = self.resize(self.frame)
        
        action = action.numpy()
        #action[0] /= 100
        #action[1] /= 100
        action = np.round(action)
        print(action)
        mouse_x = int(action[0])
        mouse_y = int(action[1])
        mouse_left = action[2]
        mouse_right = action[3]
        w = action[4]
        a = action[5]
        s = action[6]
        d = action[7]
        e = action[8]
        q = action[9]
        shift = action[10]
        space = action[11]
        reward = 0
        done = False
        keyboard = Controller()
        x = int(self.input_width / 2)  
        y = int(self.input_height / 2)

        #if action[2] == 1:
            #print("fire")
        #    fire = True
        #x_focus = (action[1] - 50) * 5
        #y_focus = (action[2] - 50) * 5
        x_focus = 0
        y_focus = 0
        """
        if action[1] == 1:
            x_focus = 100
        elif action[1] == 2:
            x_focus = -100
        if action[2] == 1:
            y_focus = 100
        elif action[2] == 2:
            y_focus = -100
        """
        if action[0] == 1:
            x_focus = 50
        elif action[0] == 2:
            x_focus = -50

        if action[1] == 1:
            y_focus = 50
        elif action[1] == 2:
            y_focus = -50

        #x_focus = int(action[0] * 100)
        #y_focus = int(action[2] * 100)

        
        if self.y_screen_pos <= 2000 and self.y_screen_pos >= -2000:
            self.y_screen_pos += y_focus
            if self.y_screen_pos > 2000:
                self.y_screen_pos = 2000
            elif self.y_screen_pos < -2000:
                self.y_screen_pos = -2000

        if self.y_screen_pos > 1000 or self.y_screen_pos < -1000:
            reward -= 10
            pass
        if self.y_screen_pos > 500 or self.y_screen_pos < -500:
            reward -= 5
            pass
        
        
        
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
                
        if w == 1:
            keyboard.press('w')
            #reward += 10
            
        if a == 1:
            keyboard.press('a')     
            #reward += 5
        
        if s == 1:
            keyboard.press('s')
            #reward += 5
            
        if d == 1:
            keyboard.press('d')
            #reward -= 5
        
        if e == 1:
            keyboard.press('e')
            keyboard.release('e')
            #reward += 5
        
        if q == 1:
            keyboard.press('q')
            keyboard.release('q')

        if shift == 1:
            #keyboard.press('shift')
            #keyboard.release('shift')
            pass

        if space == 1:
            #keyboard.press('space')
            #keyboard.release('space')
            pass
        
        if mouse_left == 1:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

        if mouse_right == 1:
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, x, y, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, x, y, 0, 0)


        #x_diff = x - x_focus
        #y_diff = y - y_focus

        """
        if x_focus < 1280:
            reward += 100
        else:
            reward -= 10
        if y_focus < 840 and y_focus > 600:
            reward += 100
        else:
            reward -= 10
        """
        #if y_focus < 500 or y_focus > 1000:
        #    reward -= 50
        #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, x - mouse_x, y - mouse_y)
        #mouse.move(100, 100)
        #mouse_pos = autoit.mouse_get_pos()
        #print(mouse_pos)
        #autoit.mouse_move(1, 0)
        self.app.Overwatch.move_mouse((x-mouse_x, y-mouse_y))
       # print(app.Overwatch.print_control_identifiers())
        
        """
        if not self.skip_frame:
            classify = self.classification()
            self.prev_detections = classify
            self.skip_frame = True
        else:
            classify = self.prev_detections
            self.skip_frame = False
        """
        
        classify = self.classification()
        obs = np.full((2,6,2), 0)
        if classify:
            #print("Enemies in LOS")
            enemy = 0
            team = 0
            #closest = None
            #closest_dist = None
            for det in classify:
                reward += 5
                tl_x = det[0][0]
                tl_y = det[0][1]
                br_x = det[1][0]
                br_y = det[1][1]
                #cv2.rectangle(screen, (tl_x, tl_y), (br_x, br_y), (255, 0, 0), -1)
                mid_x = int((tl_x + br_x) / 2)
                mid_y = int(((tl_y + br_y) / 2) + 25)
                try:
                    if det[2] == 0:
                        obs[enemy] = [mid_x, mid_y]
                        enemy += 1
                    elif det[2] == 1:
                        obs[team] = [mid_x, mid_y]
                        team += 1
                except:
                    continue
                if x >= tl_x and x <= br_x and y >= tl_y and y <= br_y:
                    reward += 50
                    #done = True
                    print("Enemy Spotted")
                    if mouse_left:
                        reward += 150
                        
    
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
            reward -= 10
            if mouse_left:
                reward -= 5
        
        #cv2.imshow('frame', screen)
        #if cv2.waitKey(1): pass

        self.step_count += 1

        #obs = np.full((2, 6, 2,), 0)

        #if self.step_count % 500 == 0:
        #    done = True
        #a = np.concatenate((self.get_screen(), obs), axis=None)

        #self.render()
        #print("Reward: {}".format   
        # MAKE ONLY 1 SCREENSHOT PER FRAME!
             
        #screen = screen.astype(np.float32) 
        return (screen, obs), reward, done, {}
        #return obs, reward, done, {}

    def reset(self):
        #keyboard = Controller()
        #keyboard.press('f')
        #keyboard.release('f')
        #screen = self.get_screen()
        #screen = screen.astype(np.float32) 
        screen = self.resize(self.frame)
        #screen /= 255.0
        obs = np.full((2,6,2), 0)
        #obs = np.full((12,2), 0)
        return (screen, obs)
        #return self.observation()

    def render(self, mode='human'):
        #frame = self.get_screen()
        cv2.imshow('frame', self.frame)
        if cv2.waitKey(1): pass

    def get_screen(self, scale=100):
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

    def resize(self, screen, size=360):
        screen = cv2.resize(screen, (size, size), interpolation=cv2.INTER_AREA)
        return screen
    
    def classification(self):
        im = self.resize(self.frame, self.yolo_size)
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
