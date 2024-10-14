import multiprocessing
from multiprocessing import shared_memory   
import pyautogui
import numpy as np
import cv2


class Screenshot:
    def __init__(self, shared_mem, end_queue, screenshot_ready, size, dtype):
        self.shared_numpy = np.ndarray(size, dtype=dtype, buffer=shared_mem.buf)

        self.end_queue = end_queue
        self.screenshot_ready = screenshot_ready
        self.run()

    @staticmethod
    def screenshot():
        frame = pyautogui.screenshot()
        frame = np.array(frame)
        screen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return screen
        
    def run(self):
        first = True
        while True:
            frame = self.screenshot()
            frame = np.array(frame)
            #print(frame.shape, frame.dtype)
            # copy the detection to the shared memory
            self.shared_numpy[:] = frame[:]
            
            if first:
                self.screenshot_ready.set()
                first = False

            try:
                if self.end_queue.get_nowait():
                    self.close()
                    break
            except:
                pass

    def close(self):
        self.shared_numpy.close()