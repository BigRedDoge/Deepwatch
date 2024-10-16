from gymnasium import Env, spaces
import numpy as np
from multiprocessing import Process
from threading import Thread
import asyncio
import websockets
import json
from queue import Queue
import multiprocessing
from multiprocessing import shared_memory
import time
import math
import cv2

from movement import Movement
from detections import Detections
from screenshot import Screenshot


class DeepwatchEnv(Env):
    def __init__(self):
        #self.action_space = spaces.Discrete(2)
        #self.observation_space = spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.int)

        self.setup_movement_server()
        self.setup_screenshot_process()
        self.setup_detection_process()
        #self.test()
        #self.close()
        #TODO: create process for environment client

        self.screen_height = 1440 // 10
        self.screen_width = 2560 // 10

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.screen_height, self.screen_width, 1), dtype=np.uint8)

        self.already_shot = False
        self.shots_hit = 0

    def setup_movement_server(self):
        """
        Starts the movement server in a separate thread, environment is the client
        It listens for messages sent from the environment through the movement method
        """
        self.move_queue = Queue()
        self.movement_thread = Thread(target=Movement, args=(self.move_queue,))
        self.movement_thread.start()

    def setup_screenshot_process(self):
        """
        Starts the screenshot process
        Shares the screenshot through shared memory
        """
        # Shared memory empty example to provide the size
        screen = np.array(Screenshot.screenshot())
        #print(screen.shape, screen.dtype, screen.nbytes)
        self.screen_shared_mem = shared_memory.SharedMemory(name='screenshot', create=True, size=screen.nbytes)
        # Queue to send a message to the screenshot process to close
        self.screenshot_end_queue = multiprocessing.Queue()
        screenshot_ready = multiprocessing.Event()

        screenshot = multiprocessing.Process(target=Screenshot, args=(self.screen_shared_mem, self.screenshot_end_queue, screenshot_ready, screen.shape, screen.dtype))
        screenshot.start()

        screenshot_ready.wait()

        existing_memory = shared_memory.SharedMemory(name='screenshot')
        # Create a numpy array from the shared memory

        self.screen_dtype = screen.dtype
        self.screen_shape = screen.shape
        self.screenshot = np.ndarray(screen.shape, dtype=screen.dtype, buffer=existing_memory.buf)

        #self.test()

    def setup_detection_process(self):
        """
        Starts the detection process
        Shares detection results through shared memory
        """
        # Detection model path
        model_path = 'detection-v1.pt'
        # Shared memory empty example to provide the size
        a = np.zeros((10, 5), dtype=np.float64)
        self.detect_shared_mem = shared_memory.SharedMemory(name='detections', create=True, size=a.nbytes)
        # Queue to send a message to the detection process to close
        self.detection_end_queue = multiprocessing.Queue()
        detection_ready = multiprocessing.Event()

        detections = multiprocessing.Process(target=Detections, args=(model_path, self.detect_shared_mem, self.detection_end_queue, detection_ready,))
        detections.start()

        detection_ready.wait()

        existing_memory = shared_memory.SharedMemory(name='detections')
        # Create a numpy array from the shared memory
        self.detections = np.ndarray((10, 5), dtype=np.int64, buffer=existing_memory.buf)

        #self.test()

    def step(self, action):
        reward = 0

        detection_memory = shared_memory.SharedMemory(name='detections')
        self.detections = np.ndarray((10, 5), dtype=np.float64, buffer=detection_memory.buf)
        screenshot_memory = shared_memory.SharedMemory(name='screenshot')
        self.screenshot = np.ndarray(self.screen_shape, dtype=self.screen_dtype, buffer=screenshot_memory.buf)

        frame = self.fill_in_detections()
        frame = cv2.resize(frame, (self.screen_width, self.screen_height))
        non_black_mask = np.any(frame != [0, 0, 0], axis=-1)
        frame[non_black_mask] = [1, 1, 1]
        frame = frame[:, :, 0]

        mouse_x = int(action[0] * 100)
        mouse_y = int(action[1] * 100)
        

        shot = self.detect_shot()
        if shot:
            reward += 10
            self.shots_hit += 1

        if self.detections[0][1] != 0 and self.detections[0][2] != 0 and self.detections[0][3] != 0 and self.detections[0][4] != 0:
            reward += 1
        
        half_size = 50 // 2
        x1, y1 = max(0, 0 - half_size), max(0, 0 - half_size)
        x2, y2 = min(self.screen_width, 0 + half_size), min(self.screen_height, 0 + half_size)
        roi = np.any(frame[y1:y2, x1:x2])
        if roi:
            mouse_left = 1

        detection_memory.close()
        screenshot_memory.close()
        
        info = {'shots_hit': self.shots_hit,
                'hit_shot': shot}
        
        self.movement({'movement': {'mouse_left': mouse_left, 'mouse_right': 0, 'w': 0, 'a': 0, 's': 0, 'd': 0, 'e': 0, 'q': 0, 'shift': 0, 'space': 0}, 'mouse': {'x': mouse_x, 'y': mouse_y}})

        if self.shots_hit >= 25:
            return frame, reward, True, False, info
        else:
            return frame, reward, False, False, info

    def reset(self, seed=None):
        screenshot_memory = shared_memory.SharedMemory(name='screenshot')
        self.screenshot = np.ndarray(self.screen_shape, dtype=self.screen_dtype, buffer=screenshot_memory.buf)
        screenshot_memory.close()

        frame = self.screenshot[:]
        frame = cv2.resize(frame, (self.screen_width, self.screen_height))
        non_black_mask = np.any(frame != [0, 0, 0], axis=-1)
        frame[non_black_mask] = [1, 1, 1]
        frame = frame[:, :, 0]

        return self.screenshot, None
    
    def render(self, mode='human'):
        pass

    def close(self):
        self.movement({'close': True})
        self.screenshot_end_queue.put(True)
        self.detection_end_queue.put(True)

        self.screen_shared_mem.close()
        self.screen_shared_mem.unlink()
        self.detect_shared_mem.close()
        self.detect_shared_mem.unlink()
        

    def movement(self, move):
        """
        Sends the move to the movement server
        """ 
        self.move_queue.put(move)

    def detect_shot(self):
        """
        Detects if the shot hits an enemy
        """
        #existing_memory = shared_memory.SharedMemory(name='screenshot')
        #self.screenshot = np.ndarray(self.screen_shape, dtype=self.screen_dtype, buffer=existing_memory.buf)

        lower_red = np.array([0, 0, 200])
        upper_red = np.array([50, 50, 255])
        
        height, width, _ = self.screenshot.shape
        top_right = self.screenshot[:height//5, width//5:]

        mask = cv2.inRange(top_right, lower_red, upper_red)
        red_fraction = np.sum(mask) / (mask.shape[0] * mask.shape[1])

        if red_fraction > 0.1:
            if not self.already_shot:
                self.already_shot = True
                print("Shot detected")
                return True
        else:
            self.already_shot = False

        #existing_memory.close()


    def fill_in_detections(self):
        frame = self.screenshot[:]
        for detection in self.detections:
            x1, y1 = int(detection[1]), int(detection[2])
            x2, y2 = int(detection[3]), int(detection[4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        return frame

    def test(self):
        try:
            existing_memory1 = shared_memory.SharedMemory(name='detections')
            self.detections = np.ndarray((10, 5), dtype=np.float64, buffer=existing_memory1.buf)
            existing_memory2 = shared_memory.SharedMemory(name='screenshot')
            self.screenshot = np.ndarray(self.screen_shape, dtype=self.screen_dtype, buffer=existing_memory2.buf)
        except Exception as e:
            print("Exception: ", e)
        while True:
            try:
                self.detect_shot()

                #
                #detection = self.detections
                frame = self.fill_in_detections()
                #"""
                if self.detections[0][1] == 0 and self.detections[0][2] == 0 and self.detections[0][3] == 0 and self.detections[0][4] == 0:
                    #print("No detection")
                    pass
                else:
                    #@print(self.detections)
                    closest = (-1, math.inf)
                    for i, detection in enumerate(self.detections):
                        #for clss, x, y, w, h in detection:
                        x1, y1 = int(detection[1]), int(detection[2])
                        x2, y2 = int(detection[3]), int(detection[4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f'{detection[0]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        if detection[0] == 0:
                            #print("Enemy detected")
                            distance = math.sqrt((2560//2 - x2)**2 + (1440//2 - y2)**2)
                            if distance < closest[1]:
                                closest = (i, distance)

                    if closest[0] != -1:
                        detection = self.detections[closest[0]]
                        x1, y1 = int(detection[1]), int(detection[2])
                        x2, y2 = int(detection[3]), int(detection[4])
                        move_x = ((x1 + x2)//2 - 2560//2) // 10
                        move_y = ((y1 + y2)//2 - 1440//2) // 10
                        #print("Move: ", move_x, move_y)
                        if (x1+x2//2) < 2560//2+400 and (x1+x2//2) > 2560//2-400 and (y1+y2//2) < 1440//2+400 and (y1+y2//2) > 1440//2-400:
                            #print("Enemy in the center")
                            #move = {'movement': {'mouse_left': 1, 'mouse_right': 1, 'w': 0, 'a': 0, 's': 0, 'd': 0, 'e': 0, 'q': 0, 'shift': 0, 'space': 0}, 'mouse': {'x': move_x, 'y': move_y}}
                            pass
                        else:
                            pass
                        
                            #move = {'movement': {'mouse_left': 0, 'mouse_right': 0, 'w': 0, 'a': 0, 's': 0, 'd': 0, 'e': 0, 'q': 0, 'shift': 0, 'space': 0}, 'mouse': {'x': move_x, 'y': move_y}}
                        #self.movement(move)
                #"""        
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Closing")
                    existing_memory1.close()
                    existing_memory2.close()
                    break
            except Exception as e:
                print("Exception: ", e)
                
if __name__ == '__main__':
    import cv2
    env = DeepwatchEnv()
    env.test()
    env.close()
    """for i in range(100):
        print(env.detections)
        cv2.imshow('screenshot', env.screenshot)
        env.movement({'movement': {'w': 1, 'a': 0, 's': 0, 'd': 0, 'e': 0, 'q': 0, 'shift': 0, 'space': 0}, 'mouse': {'x': 0, 'y': 0}})

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    """
    #env.close()