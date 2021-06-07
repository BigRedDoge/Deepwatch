import win32api, win32con
import time

x = int(2560 / 2)
y = int(1440 / 2)
print(x,y)
time.sleep(3)

#win32api.SetCursorPos((x, y))
win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, x, y - 150, 0, 0)
#@time.sleep(0.01667)