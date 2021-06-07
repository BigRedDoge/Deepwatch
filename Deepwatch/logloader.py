import math
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import cv2
import glob


class LogLoader(data.Dataset):
    def __init__(self, log_dir, frame_size, transform=None, is_val=False, game_size=(2560, 1440)):
        self.is_val = is_val
        self.frame_size = frame_size
        self.transform = transform
        self.game_width = game_size[0]
        self.game_height = game_size[1] #self.get_image_size(self.log.iloc[0]['frame_path'])
        self.log_dir = log_dir
        self.log = self.combine_logs(log_dir)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        log_data = self.log.iloc[idx]

        frame_path = log_data['frame_path']
        frame = self.frame_loader(frame_path)
        if self.transform != None:
            frame_tensor = self.transform(frame)
        else:
            frame_tensor = transforms.ToTensor()(frame)
        
        detection = log_data[['pos_x', 'pos_y']]
        detection = self.scale_detections(detection)

        detect_tensor = torch.from_numpy(detection.flatten())
        #print(detect_tensor.shape)
        
        target = np.ones(12)
        target[0] = log_data['mouse_x']
        target[1] = log_data['mouse_y']
        target[2] = log_data['mouse_left']
        target[3] = log_data['mouse_right']
        target[4] = log_data['key_w']
        target[5] = log_data['key_a']
        target[6] = log_data['key_s']
        target[7] = log_data['key_d']
        target[8] = log_data['key_e']
        target[9] = log_data['key_q']
        target[10] = log_data['key_shift']
        target[11] = log_data['key_space']
        #print(target)
        #target = log_data[self.log.columns.difference(['pos_x', 'pos_y', 'frame_path', 'index'])]
        #target = target.to_numpy()[::-1]
        #print(target)
        #target[0] /= 200
        #target[1] /= 200
        target = torch.from_numpy(target.astype(np.float32))

        return frame_tensor, detect_tensor, target
        
    def combine_logs(self, logging_dir):
        log_dirs = [name for name in os.listdir(logging_dir)]
        combined_log = None

        for log_dir in log_dirs:
            log_path = os.path.join(logging_dir, log_dir, 'log.csv')
            detect_path = os.path.join(logging_dir, log_dir, 'detections.csv')
            game_log = self.read_log(log_path, detect_path)
            if type(combined_log) == None:
                combined_log = game_log
            else:
                game_log = game_log.reset_index(drop=True)
                combined_log = pd.concat([combined_log, game_log], axis=0)

        split = int(len(combined_log) * 0.85)
        if self.is_val:
            return combined_log[split:]
        else:
            return combined_log[:split]
        #return combined_log

    def read_log(self, log_path, detect_path):
        detect = pd.read_csv(detect_path)
        log = pd.read_csv(log_path)

        det = []
        for i in range(0, len(detect), 12): 
            enemies = detect.iloc[i:i+6][['pos_x', 'pos_y']]
            teammates = detect.iloc[i+6:i+12][['pos_x', 'pos_y']]
            det.append([enemies.to_numpy(), teammates.to_numpy()])

        #detect_log = pd.DataFrame(det, columns=[['pos_x', 'pos_y']]) 
        #detect_log = detect.reset_index(drop=True)
        scale_det = []
        for i, detect in enumerate(det): 
            scale_det.append(self.scale_detections(detect))
        #log = pd.concat([action_log, detect_log[['pos_x', 'pos_y']]], axis=1)
        #print(scale_det[0])
        #print(scale_det)
        log = log.sort_values(by='index')

        log['pos_x'] = [x[0] for x in scale_det]
        log['pos_y'] = [x[1] for x in scale_det]

        log = log[log['mouse_x'] <= 200] 
        log = log[log['mouse_x'] >= -200]
        log = log[log['mouse_y'] <= 200]
        log = log[log['mouse_y'] >= -200]
        
        #log['sort'] = log['frame_path'].str.extract('(\d+)', expand=False).astype(int)
        #log.sort_values('sort', inplace=True)
        #sign = lambda n: (100, -100)[n < 0]
        #normalize = lambda x: (x/100.0) + sign(x)*-0.5 if x <= 100 and x >= -100 else sign(x)*0.5
        normalize = lambda x: x / 200.0 #if x <= 100 and x >= -100 else sign(x)
        #normalize_x = lambda x: x / self.game_width
        #normalize_y = lambda x: x / self.game_height
        log['mouse_x'] = log['mouse_x'].apply(normalize)
        log['mouse_y'] = log['mouse_y'].apply(normalize)

        #log[log.columns.difference(['frame_path', 'mouse_x', 'mouse_y', 'pos_x', 'pos_y'])] = \
        #    log[log.columns.difference(['frame_path', 'mouse_x', 'mouse_y', 'pos_x', 'pos_y'])].sub(0.5)
        #print(log)
        return log

    def scale_detections(self, detection):
        detect = np.empty([2, 6, 2])
        scale_x = self.frame_size / float(self.game_width)
        scale_y = self.frame_size / float(self.game_height)
        for i, d in enumerate(detection[0]):
            det_x = d[0]
            det_y = d[1]
            det_x *= scale_x
            det_y *= scale_y
            det_x /= self.frame_size
            det_y /= self.frame_size
            detect[0][i] = [det_x, det_y]
        for i, d in enumerate(detection[1]):
            det_x = d[0]
            det_y = d[1]
            det_x *= scale_x
            det_y *= scale_y
            det_x /= self.frame_size
            det_y /= self.frame_size
            detect[1][i] = [det_x, det_y]
        """
        det_x = detection['pos_x']
        det_y = detection['pos_y']
        print(det_x.shape)
        scale_x = self.frame_size / float(self.game_width)
        scale_y = self.frame_size / float(self.game_height)
        det_x *= scale_x
        det_y *= scale_y
        det_x /= float(self.game_width)
        det_y /= float(self.game_height)
        det = [det_x, det_y]
        """
        #print(detect)
        return detect
        
    def get_image_size(self, img_path):
        img = Image.open(img_path)
        width, height = img.size
        return width, height

    def frame_loader(self, path):
        frame = Image.open(path).convert('RGB')
        frame = frame.resize((self.frame_size, self.frame_size))
        return frame

    def __len__(self):
        return len(self.log)