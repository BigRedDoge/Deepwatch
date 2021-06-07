import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import glob
import os
import sys

obs = np.full((2,6,2), 0)
print(obs)

#sign = lambda n: (1, -1)[n < 0]
#normalize = lambda x: (x/100.0) + sign(x)*-0.5 if x <= 100 and x >= -100 else sign(x)*0.5

#print(normalize(-105))



"""
def frame_loader(path):
    frame = Image.open(path).convert('RGB')
    frame = frame.resize((512, 512))
    return frame

df = pd.read_csv('./logging/deepwatch_log_0/detections.csv')
#print(df)
detect = pd.read_csv('./logging/deepwatch_log_0/detections.csv')
det = []
for i in range(0, len(df), 12):
    enemies = detect.iloc[i:i+6][['pos_x', 'pos_y']]
    teammates = detect.iloc[i+6:i+12][['pos_x', 'pos_y']]
    det.append([enemies.to_numpy(), teammates.to_numpy()])
detect = pd.DataFrame(det, columns=['pos_x', 'pos_y'])
detect = detect.reset_index(drop=True)

log = pd.read_csv('./logging/deepwatch_log_0/log.csv')

log = pd.concat([log, detect], axis=1)
print(log.iloc[0][log.columns.difference(['pos_x', 'pos_y', 'frame_path'])].to_numpy()[::-1])
"""
#frame_path = log.iloc[0]['frame_path']
#frame = frame_loader(frame_path)
#implot = plt.imshow(np.real(frame))
#frame = np.asarray(frame) / 255.0
#frame_tensor = transforms.ToTensor()(frame).unsqueeze_(0)
#print(frame)
#plt.imshow(frame)
#plt.show()
"""
detections = df['detections'].iloc[48]
det = ' '.join(detections.split())
det = ' '.join(detections.split())
"""
#detections = detections.replace('\r\n', ', ')
#det = eval(detections)
#print(det)