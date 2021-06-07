from pynput.mouse import Button, Controller
from pynput.keyboard import Key, Controller
import mouse
from pywinauto.application import Application
import time

"""
app = Application().connect(path=r'C:\Program Files (x86)\Overwatch\_retail_\Overwatch.exe')

time.sleep(5)
app.Overwatch.type_keys('{ESC}') # esc
app.Overwatch.move_mouse_input((1280, 800)) # 1280, 800
app.Overwatch.click()
app.Overwatch.move_mouse_input((1400, 1220)) # 1400, 1220
app.Overwatch.click()
time.sleep(6)
app.Overwatch.type_keys('{ESC}') # esc
app.Overwatch.type_keys('{ESC}') # esc
app.Overwatch.move_mouse_input((1260, 1230)) # 1260, 1230
app.Overwatch.click() # click
app.Overwatch.move_mouse_input((1280, 1340)) # 1280, 1340
app.Overwatch.click()
"""
while True:
    print(mouse.get_position())