import Env_Game as env
import numpy as np
from PIL import ImageOps
from PIL import Image
from time import sleep
import threading

RESIZE = 84
np.set_printoptions(threshold=np.nan)

class Game(threading.Thread):
    def __init__(self, render):
        threading.Thread.__init__(self)
        self.render = render

    def run(self):
        e = env.Env(self.render)
        e.reset()
        d = False
        while not d:
            a = np.random.randint(0, 5)
            sleep(0.05)
            o, r, d, _ = e.step(a)
            if d:
                im = ImageOps.mirror(Image.fromarray(o).rotate(270))
                im.show()

g1 = Game(True)
g2 = Game(False)
g3 = Game(False)
g4 = Game(False)

g1.start()
g2.start()
g3.start()
g4.start()

"""
while(True):
    a = np.random.randint(0, 5)
    sleep(0.05)
    o, r, d, _ = Env.step(a)
    im = ImageOps.mirror(Image.fromarray(o).rotate(270)).convert('L').resize((RESIZE, RESIZE))
    im_arr = np.asarray(ImageOps.mirror(Image.fromarray(o).rotate(270)).convert('L').resize((RESIZE, RESIZE)))
    if d:
        im.show()
        sleep(5)
        break
"""