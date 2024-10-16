import pywinauto
from pywinauto.application import Application
from pynput.mouse import Button, Controller
import asyncio
import websockets
import json 
import time
import win32api
import win32con


class Movement:
    def __init__(self, queue):
        self.move_btn_memory = {'w': 0, 'a': 0, 's': 0, 'd': 0}
        self.mouse = Controller()
        self.queue = queue

        asyncio.run(self.loop())

    async def loop(self):
        while True:
            
            action = self.queue.get()

            try:
                if action['close']:
                    break
            except KeyError:
                pass

            await self.movement(action['movement'])

            # Cancel the previous mouse movement task if it's still running
            if hasattr(self, 'mouse_task') and not self.mouse_task.done():
                self.mouse_task.cancel()

            # Start a new mouse movement task without a timeout
            self.mouse_task = asyncio.create_task(self.look(action['mouse']))
            await self.mouse_task

    async def server(self, websocket, path):
        async for message in websocket:
            action = json.loads(message)
            
            # Check if the action has a close key, if it does, shut down the server
            try:
                if action['close']:
                    self.ws_server.close()
                    asyncio.get_event_loop().run_until_complete(self.server.wait_closed())
                    return
            except KeyError:
                pass

            self.movement(action['movement'])

            """# Cancel the previous mouse movement task if it's still running
            if hasattr(self, 'mouse_task') and not self.mouse_task.done():
                self.mouse_task.cancel()

            # Start a new mouse movement task without a timeout
            self.mouse_task = asyncio.create_task(self.look(action['mouse']))
            await self.mouse_task"""
            self.look(action['mouse'])
        
    async def movement(self, move):
        """
        move is a dictionary with the keys 'mouse_left', 'mouse_right', 'w', 'a', 's', 'd', 'e', 'q', 'shift', 'space'
        their values are either 0 or 1
        the move_btn_memory is a dictionary that keeps track of the state of the movement buttons so that buttons are held down until they are released
        """
        
        if move['mouse_left']:
            self.mouse.press(Button.left)
            self.mouse.release(Button.left)

        if move['mouse_right']:
            self.mouse.press(Button.right)
            self.mouse.release(Button.right)
        
        if move['w'] and not self.move_btn_memory['w']:
            pywinauto.keyboard.send_keys('{w down}')
            self.move_btn_memory['w'] = 1
        elif not move['w'] and self.move_btn_memory['w']:
            pywinauto.keyboard.send_keys('{w up}')
            self.move_btn_memory['w'] = 0

        if move['a'] and not self.move_btn_memory['a']:
            pywinauto.keyboard.send_keys('{a down}')
            self.move_btn_memory['a'] = 1
        elif not move['a'] and self.move_btn_memory['a']:
            pywinauto.keyboard.send_keys('{a up}')
            self.move_btn_memory['a'] = 0
        
        if move['s'] and not self.move_btn_memory['s']:
            pywinauto.keyboard.send_keys('{s down}')
            self.move_btn_memory['s'] = 1
        elif not move['d'] and self.move_btn_memory['s']:
            pywinauto.keyboard.send_keys('{s up}')
            self.move_btn_memory['s'] = 0
            
        if move['d'] and not self.move_btn_memory['d']:
            pywinauto.keyboard.send_keys('{d down}')
            self.move_btn_memory['d'] = 1
        elif not move['d'] and self.move_btn_memory['d']:
            pywinauto.keyboard.send_keys('{d up}')
            self.move_btn_memory['d'] = 0
        
        if move['e']:
            pywinauto.keyboard.send_keys('{e}')
        
        if move['q']:
            pywinauto.keyboard.send_keys('{q}')

        if move['shift']:
            pywinauto.keyboard.send_keys('{VK_SHIFT}')

        if move['space']:
            pywinauto.keyboard.send_keys('{SPACE}') 

    async def look(self, mouse):
        """
        mouse is a dictionary with the keys 'x' and 'y'
        TODO: implement mouse movement
        TODO: make movement smooth
        TODO: make movement relative to the current position of the mouse
        TODO: calculate trajectory of the mouse and move it accordingly
        """
        # absolute mouse movement
        #pywinauto.mouse.move(coords=(mouse['x'], mouse['y']))
        # relative mouse movement
        #self.mouse.move(mouse['x'], mouse['y'])
        #for i in range(10):
        #    #x, y = win32api.GetCursorPos()
        #    #print(x, y)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, mouse['x'], mouse['y'], 0, 0)

if __name__ == '__main__':
    """import threading
    import time
    import queue
    mp_queue = queue.Queue()
    movement_process = multiprocessing.Process(target=Movement, args=(mp_queue,))
    movement_process.start()
    time.sleep(5)
    mp_queue.put({'movement': {'w': 1, 'a': 0, 's': 0, 'd': 0, 'e': 0, 'q': 0, 'shift': 0, 'space': 0}, 'mouse': {'x': 0, 'y': 0}})
    """
    import win32api
    import win32con
    
    for i in range(10):
        x, y = win32api.GetCursorPos()
        print(x, y)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 1, 1, 0, 0)
    