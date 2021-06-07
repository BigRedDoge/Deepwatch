import win32api as win
import time
import keyboard
import os
import mss
import cv2
import numpy as np
import csv
import threading
import math
import random
import sys
import torch
import torch.backends.cudnn as cudnn
sys.path.insert(1, r'C:\Users\Sean\Desktop\Tradebot\Deepwatch\yolov5')
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import (apply_classifier, check_img_size,
                           non_max_suppression, plot_one_box, scale_coords,
                           set_logging, strip_optimizer, xyxy2xywh)
from utils.torch_utils import load_classifier, select_device, time_synchronized
import pandas as pd
import queue


save_dir = './logging/'
num_threads = 10

weights = r'c:\Users\Sean\Desktop\Tradebot\Deepwatch\yolov5\runs\exp41\weights\best.pt'
device = torch.device("cuda:0")
model = attempt_load(weights, map_location=device)
model.half()

csv_lock = threading.Lock()
thread_q = queue.Queue() #maxsize=num_threads)

def main():
    # create save directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    num_saved = len(os.listdir(save_dir))
    print(num_saved)
    log_dir_name = "deepwatch_log_{}".format(num_saved)
    path = os.path.join(save_dir, log_dir_name)
    if not os.path.exists(path):
        os.mkdir(path)
    print("Logs being saved to {} directory".format(path))

    with open(path + '/log.csv', mode='a') as f:
        fieldnames = ['mouse_x', 'mouse_y', 'mouse_left', 'mouse_right', 'key_w', 'key_a', 'key_s', 'key_d', 'key_e', 'key_q', 'key_shift', 'key_space', 'frame_path', 'index'] 
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    with open(path + '/detections.csv', mode='w') as f:
        fieldnames = ['index', 'pos_x', 'pos_y'] 
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    x = 1280
    y = 720
    left_mouse_btn = 0
    tick_left = 0
    right_mouse_btn = 0
    tick_right = 0
    # start worker thread for queue
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    i = 0
    running = True
    prev_x = 0
    prev_y = 0
    while running:
        time_start = time.time()
        # get mouse movement0
        pos = win.GetCursorPos()
        pos_x = pos[0]
        pos_y = pos[1]
        mouse_x = x - pos_x 
        mouse_y =  y - pos_y
        
        mouse_move = [mouse_x, mouse_y]

        left = win.GetKeyState(0x01)
        right = win.GetKeyState(0x02)

        if left == -127 or left == -128:
            left_mouse_btn = 1
        else:
            if left_mouse_btn == 1 and tick_left < 2:
                left_mouse_btn = 1
                tick_left += 1
            else:
                left_mouse_btn = 0
                tick_left = 0
        if right == -127 or right == -128:
            right_mouse_btn = 1
        else:
            if right_mouse_btn == 1 and tick_right < 2:
                right_mouse_btn = 1
                tick_right += 1
            else:
                right_mouse_btn = 0
                tick_right = 0
        mouse_btns = [left_mouse_btn, right_mouse_btn]
        
        # record keyboard presses
        boolean = lambda x: 1 if x else 0
        w = boolean(keyboard.is_pressed('w'))
        a = boolean(keyboard.is_pressed('a'))
        s = boolean(keyboard.is_pressed('s'))
        d = boolean(keyboard.is_pressed('d'))
        e = boolean(keyboard.is_pressed('e'))
        q = boolean(keyboard.is_pressed('q'))
        shift = boolean(keyboard.is_pressed('shift'))
        space = boolean(keyboard.is_pressed('space'))
        keypresses = [w, a, s, d, e, q, shift, space]

        actions = mouse_move + mouse_btns + keypresses
        print(actions)

        frame_path = path + '/frame_{}.jpg'.format(i)
        frame = screenshot()
        index = i
        save = threading.Thread(target=detect_and_save, args=(actions, frame, frame_path, path, index,))
        thread_q.put(save)

        time_end = time.time()
        time_run = time_end - time_start
        fps = 1/10 # 10 fps
        time_diff = fps - time_run
        if time_diff > 0:
            time.sleep(time_diff)

        if keyboard.is_pressed('0'):
            print("Joining threads")
            thread_q.join()
            running = False
        i += 1
        #time.sleep()

def worker():
    while True:
        thread = thread_q.get()
        thread.start()
        #thread.join()
        thread_q.task_done()
 
def detect_and_save(actions, frame, frame_path, path, index):
    # save frame
    cv2.imwrite(frame_path, frame)
    # classify enemies/teammates
    detections = np.full((2, 6, 2), 0)
    try:
        classify = classification(frame_path)
    except:
        classify = classification(frame_path)
    teammate = 0
    enemy = 0
    for det in classify:
        label = det[0]
        if label == 0 and enemy < 6: # 6 is max amount of enemies/teammates
            detections[label][enemy] = [det[1], det[2]]
            enemy += 1
        elif label == 1 and teammate < 6:
            detections[label][teammate] = [det[1], det[2]]
            teammate += 1
    # save log to csv
    with csv_lock:
        with open(path + '/log.csv', mode='a') as f:
            fieldnames = ['mouse_x', 'mouse_y', 'mouse_left', 'mouse_right', 'key_w', 'key_a', 'key_s', 'key_d', 'key_e', 'key_q', 'key_shift', 'key_space', 'frame_path', 'index']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'mouse_x': actions[0], 'mouse_y': actions[1], 'mouse_left': actions[2], 'mouse_right': actions[3], 
                            'key_w': actions[4], 'key_a': actions[5], 'key_s': actions[6], 'key_d': actions[7], 'key_e': actions[8],
                            'key_q': actions[9], 'key_shift': actions[10], 'key_space': actions[11], 'frame_path': frame_path, 'index': index})
        # save detections to csv
        detections = np.reshape(detections, (-1, 2))
        pd.DataFrame(detections).to_csv(path + '/detections.csv', mode='a', header=False)

def screenshot():
    dim = (2560, 1440)
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": dim[0], "height": dim[1]}
        screen = np.array(sct.grab(monitor))
        screen = screen[:,:,:3]
        np.array(screen)
        return screen

def classification(frame_path):
        imgsz = 1280
        scale = 100
        dataset = LoadImages(frame_path, img_size=imgsz)
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)
        _ = model(img.half())
        detections = []
        for path, img, im0s, _ in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half()
            img /= 255.0  
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img, augment='')[0]
            conf_thresh = 0.4
            pred = non_max_suppression(pred, conf_thresh, 0.5)
            for i, det in enumerate(pred):
                p, im0 = path, im0s
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                            # bounding box of each detection, c1 top left c2 bottom right
                            c1, c2 = (int(xyxy[0] * (100 / scale)), int(xyxy[1] * (100 / scale))), \
                                (int(xyxy[2] * (100 / scale)), int(xyxy[3] * (100 / scale)))
                            mid_x = int((c1[0] + c2[0]) / 2)
                            mid_y = int(((c1[1] + c2[1]) / 2) + 25)
                            detections.append([int(cls), mid_x, mid_y])

        detections.sort()
        return detections


if __name__ == '__main__':
    main()