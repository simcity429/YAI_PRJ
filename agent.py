import env_game as env
import numpy as np
from PIL import ImageOps
from PIL import Image
from time import sleep

RESIZE = 84
np.set_printoptions(threshold=np.nan)
Env = env.Env()
Env.reset()
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