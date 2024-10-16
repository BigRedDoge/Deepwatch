from ultralytics import YOLO
import pyautogui
import time
import numpy as np
import torch
import cv2
import multiprocessing
from multiprocessing import shared_memory

from screenshot import Screenshot

class Detections:
    def __init__(self, model_path, shared_mem, end_queue, detection_ready):
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')

        self.detection_ready = detection_ready
        self.shared_numpy = np.ndarray((10, 5), dtype=np.float64, buffer=shared_mem.buf)

        self.end_queue = end_queue

        # Create screenshot from shared memory
        screen = np.array(Screenshot.screenshot())
        self.screen_existing_memory = shared_memory.SharedMemory(name='screenshot')
        self.screenshot = np.ndarray(screen.shape, dtype=screen.dtype, buffer=self.screen_existing_memory.buf)
        
        self.run()

    def detect(self, frame):
        #try:
        results = self.model.predict(frame, verbose=False)
        boxes = results[0].boxes.cpu().numpy()
        detections =[(clss, *xy) for clss, xy in zip(boxes.cls, boxes.xyxy)]
        detection = np.array(detections)

        if len(detection) > 0:
            pad_rows = max(0, 10 - detection.shape[0])
            detection = np.pad(detection, ((0, pad_rows), (0, 0)))
        else:
            detection = np.zeros((10, 5))
        #except Exception as e:
        #    print("Exception: ", e)
        #    detection = np.zeros((10, 5))
        #print("type: ", detection.dtype)
        #detection = detection.astype(np.int64)
        return detection

        """def screenshot(self):
            start = time.perf_counter()
            frame = pyautogui.screenshot()
            frame = np.array(frame)
            screen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            end = time.perf_counter()
            print(f"Screenshot took {end-start} seconds")
            return screen
        """

    def run(self):
        first = True
        while True:
            frame = self.screenshot
            detection = self.detect(frame)
            # copy the detection to the shared memory
            self.shared_numpy[:] = detection[:]

            if first:
                self.detection_ready.set()
                first = False
            try:
                if self.end_queue.get_nowait():
                    self.close()
                    break
            except:
                pass

    def close(self):
       self.shared_numpy.close()
       self.screen_existing_memory.close()


if __name__ == '__main__':
    import multiprocessing
    from multiprocessing import shared_memory
    model_path = 'yolov9s.pt'
    a = np.zeros((10, 5), dtype=np.int64)
    shared_mem = shared_memory.SharedMemory(name='detections', create=True, size=a.nbytes)
    end_queue = multiprocessing.Queue()

    detections = multiprocessing.Process(target=Detections, args=(model_path, shared_mem, end_queue,))
    detections.start()

    existing_memory = shared_memory.SharedMemory(name='detections')
    shared_numpy = np.ndarray((10, 5), dtype=np.int64, buffer=existing_memory.buf)
    print("shared memory: ", shared_numpy)
    print(time.perf_counter())
    while True:
        print(shared_numpy)
        time.sleep(1)
    end_queue.put(True)
    detections.join()