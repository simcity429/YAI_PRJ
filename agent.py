import env_game as env
import numpy as np
from PIL import ImageOps
from PIL import Image
from time import sleep

RESIZE = 84
np.set_printoptions(threshold=np.nan)

env.reset()
while(True):
    a = np.random.randint(0, 5)
#    sleep(0.01)
    o, r, d, _ = env.step(a)
    im_arr = np.asarray(ImageOps.mirror(Image.fromarray(o).rotate(270)).convert('L').resize((RESIZE, RESIZE)))
    if d:
        print(im_arr)
        sleep(5)
        break