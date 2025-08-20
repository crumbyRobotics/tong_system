from pynput.mouse import Button, Controller
from pynput import mouse, keyboard


class MouseManager:
    def __init__(self):
        self.mouse = Controller()
        self.listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click, on_scroll=self.on_scroll)
        self.listener.start()
        self.x = 0
        self.y = 0
        self.pressed = False
        self.button = None

    def reset(self, init_x, init_y):
        self.x = init_x
        self.y = init_y
        self.mouse.position = self.x, self.y

    def get_key(self):
        return (self.x, self.y, self.pressed, self.button)

    def on_move(self, x, y):
        self.x = x
        self.y = y

    def on_click(self, x, y, button, pressed):
        self.pressed = True if pressed else False
        self.button = button

    def on_scroll(self, x, y, dx, dy):
        pass


class KeyboardManager:
    def __init__(self):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.key = None

    def get_key(self):
        key = self.key
        self.key = None
        return key

    def on_press(self, key):
        # store key which is not char
        try:
            self.key = key.__dict__["_name_"]
        # store char
        except:
            self.key = key.char

    def on_release(self, key):
        if key == keyboard.Key.esc:
            return False
