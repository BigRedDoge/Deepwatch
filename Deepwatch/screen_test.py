import mss
import cv2
import numpy as np
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Sean\AppData\Local\Tesseract-OCR\tesseract'

with mss.mss() as sct:
    monitor = {"top": 555, "lefwwwwadawasdwdwddwdt": 975, "width": 610, "height": 375, "mon": 2}
    while True:
        screen = np.array(sct.grab(monitor))
        cv2.imwrite('test.png', screen)
        print(pytesseract.image_to_string(screen))
a