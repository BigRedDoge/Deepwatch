import pywinauto
from pywinauto.application import Application
from pynput.mouse import Button, Controller
import asyncio
import websockets


class Movement:
    def __init__(self, port=8765):
        self.move_btn_memory = {'w': 0, 'a': 0, 's': 0, 'd': 0}
        self.mouse = Controller()

        self.ws_server = websockets.serve(self.server, "localhost", port)

        asyncio.get_event_loop().run_until_complete(self.ws_server)
        asyncio.get_event_loop().run_forever()

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

            await self.movement(action['movement'])

            # Cancel the previous mouse movement task if it's still running
            if hasattr(self, 'mouse_task') and not self.mouse_task.done():
                self.mouse_task.cancel()

            # Start a new mouse movement task without a timeout
            self.mouse_task = asyncio.create_task(self.look(action['mouse']))
            await self.mouse_task
        
    async def movement(self, move):
        """
        move is a dictionary with the keys 'mouse_left', 'mouse_right', 'w', 'a', 's', 'd', 'e', 'q', 'shift', 'space'
        their values are either 0 or 1
        the move_btn_memory is a dictionary that keeps track of the state of the movement buttons so that buttons are held down until they are released
        """
        if move['mouse_left']:
            self.mouse.press(Button.left)
            self.mouse.release(Button.left)

        if mouse_right >= 0:
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
        pywinauto.mouse.move(coords=(mouse['x'], mouse['y']))
        # relative mouse movement
        self.mouse.move(mouse['x'], mouse['y'])