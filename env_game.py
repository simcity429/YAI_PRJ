import pygame as pg
import sys
import numpy as np
import numpy.random as r

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
SKIP_FRAMES = 2
MAX_SPEED = 20
BULLET_MAX_SPEED = 4
DSPEED = 2

class Bullet:
    def __init__(self):
        global bullet_size
        x = 0
        y = 0
        self.dx = 0
        self.dy = 0
        margin = bullet_size[0]/2
        bullet_loc_select = r.randint(0, 3)
        if bullet_loc_select == 0:
            x = r.randint(0 - margin, SCREEN_WIDTH - margin)
            y = r.randint(0 - margin, 0)
            self.dx = r.randint(-BULLET_MAX_SPEED, BULLET_MAX_SPEED)
            self.dy = r.randint(1, BULLET_MAX_SPEED)
        elif bullet_loc_select == 1:
            x = r.randint(0 - margin, 0)
            y = r.randint(0 - margin, SCREEN_HEIGHT - margin)
            self.dx = r.randint(1, BULLET_MAX_SPEED)
            self.dy = r.randint(-BULLET_MAX_SPEED, BULLET_MAX_SPEED)
        elif bullet_loc_select == 2:
            x = r.randint(0 - margin, SCREEN_WIDTH - margin)
            y = r.randint(SCREEN_HEIGHT, SCREEN_HEIGHT + margin)
            self.dx = r.randint(-BULLET_MAX_SPEED, BULLET_MAX_SPEED)
            self.dy = r.randint(-BULLET_MAX_SPEED, -1)
        else:
            x = r.randint(SCREEN_WIDTH, SCREEN_WIDTH + margin)
            y = r.randint(0 - margin, SCREEN_HEIGHT - margin)
            self.dx = r.randint(-BULLET_MAX_SPEED, -1)
            self.dy = r.randint(-BULLET_MAX_SPEED, BULLET_MAX_SPEED)

        self.rect = pg.Rect(x, y, int(bullet_size[0]*0.3), int(bullet_size[1]*0.3))

    def move_bullet(self):
        self.rect.x += self.dx
        self.rect.y += self.dy

    def draw_bullet(self):
        global bullet_image
        draw_img(bullet_image, self.rect.x, self.rect.y)

class State:
    def __init__(self):
        global flan_size
        flan_loc = [(SCREEN_WIDTH - (flan_size[0] / 2)) / 2, (SCREEN_HEIGHT - (flan_size[1] / 2)) / 2]
        self.flan_rect = pg.Rect(flan_loc, (flan_size[0] * 0.5, flan_size[1] * 0.5))
        self.flan_flag = 0
        self.flan_dx = 0
        self.flan_dy = 0
        self.score = 0

def bullet_create():
    global bullet_list
    bullet = Bullet()
    bullet_list.append(bullet)

def bullet_clean():
    global bullet_list
    global bullet_size
    margin = bullet_size[0]/2
    delete_bullet = []
#    print("bullet list len " ,len(bullet_list))
    for bullet in bullet_list:
        if (bullet.rect.x < -margin or bullet.rect.x > SCREEN_WIDTH + margin) or (bullet.rect.y < -margin or bullet.rect.y > SCREEN_HEIGHT + margin):
            delete_bullet.append(bullet)
    bullet_list = [bullet for bullet in bullet_list if not (bullet in delete_bullet)]
    for bullet in delete_bullet:
        del bullet


def draw_img(img, x, y):
    global gamepad
    gamepad.blit(img, (x, y))

def init_game():
    global gamepad, clock
    global flan_image
    global bullet_image
    global flan_size, bullet_size
    global bullet_list
    pg.init()
    gamepad = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    flan_image = []
    flan_stop = pg.image.load("./image/flan_stop.jpg")
    flan_left1 = pg.image.load("./image/flan_left_1.jpg")
    flan_left2 = pg.image.load("./image/flan_left_2.jpg")
    flan_right1 = pg.image.load("./image/flan_right_1.jpg")
    flan_right2 = pg.image.load("./image/flan_right_2.jpg")
    bullet_image = pg.image.load("./image/danmaku.png")
    flan_image.append(flan_stop)
    flan_image.append(flan_left1)
    flan_image.append(flan_left2)
    flan_image.append(flan_right1)
    flan_image.append(flan_right2)
    flan_size = flan_stop.get_rect().size
    bullet_size = bullet_image.get_rect().size
    bullet_list = []
    clock = pg.time.Clock()


def reset():
    init_game()
    global state
    state = State()

def step(action):
    global state
    global flan_size
    for _ in range(SKIP_FRAMES):
        #bullet create
        r_num = r.randint(0, 10)
        if (r_num == 1):
            bullet_create()
       #action = {0 : stop, 1: right, 2: left, 3: up, 4: down}
        if action == 0:
            state.flan_flag = 0
            state.flan_dx = 0
            state.flan_dy = 0
        elif action == 1:
            state.flan_dx += DSPEED
            if state.flan_dx < MAX_SPEED:
                state.flan_flag = 3
            else:
                state.flan_flag = 4
            if state.flan_dx > MAX_SPEED:
                state.flan_dx -= DSPEED
        elif action == 2:
            state.flan_dx -= DSPEED
            if state.flan_dx > -MAX_SPEED:
                state.flan_flag = 1
            else:
                state.flan_flag = 2
            if state.flan_dx < -MAX_SPEED:
                state.flan_dx += DSPEED
        elif action == 3:
            state.flan_dy -= DSPEED
            if state.flan_dy < -MAX_SPEED:
                state.flan_dy += DSPEED
        elif action == 4:
            state.flan_dy += DSPEED
            if state.flan_dy > MAX_SPEED:
                state.flan_dy -= DSPEED

        state.flan_rect.x += state.flan_dx
        state.flan_rect.y += state.flan_dy

        if state.flan_rect.x < 0:
            state.flan_rect.x = 0
        elif state.flan_rect.x + flan_size[0] > SCREEN_WIDTH:
            state.flan_rect.x = SCREEN_WIDTH - flan_size[0]

        if state.flan_rect.y < 0:
            state.flan_rect.y = 0
        elif state.flan_rect.y + flan_size[1] > SCREEN_HEIGHT:
            state.flan_rect.y = SCREEN_HEIGHT - flan_size[1]

        for bullet in bullet_list:
            bullet.move_bullet()

        bullet_clean()

        gamepad.fill((0, 0, 0))
        draw_img(flan_image[state.flan_flag], state.flan_rect.x, state.flan_rect.y)
        for bullet in bullet_list:
            bullet.draw_bullet()
        ret_state = pg.surfarray.array3d(gamepad)
        pg.display.flip()

        for bullet in bullet_list:
            if state.flan_rect.colliderect(bullet.rect):
                print("score: ", state.score)
                return ret_state, 0, True, _
        state.score += 1
    return ret_state, 1, False, _




def run():
    global gamepad, clock
    global bullet_list
    global flan_image
    global flan_size

    score = 0

    flan_loc = [(SCREEN_WIDTH-(flan_size[0]/2))/2, (SCREEN_HEIGHT-(flan_size[1]/2))/2]
    flan_rect = pg.Rect(flan_loc, (flan_size[0] * 0.5, flan_size[1] * 0.5))
    flan_flag = 0
    flan_dx = 0
    flan_dy = 0
    running = True
    while running:
        score += 1/15
        clock.tick(30)
        pressed = pg.key.get_pressed()
        r_num = r.randint(0, 10)
        if (r_num == 1):
            bullet_create()

        if (pressed[pg.K_RIGHT]):
            flan_dx += DSPEED
            if flan_dx < MAX_SPEED:
                flan_flag = 3
            else :
                flan_flag = 4
            if flan_dx > MAX_SPEED:
                flan_dx -= DSPEED
        elif (pressed[pg.K_LEFT]):
            flan_dx -= DSPEED
            if flan_dx > -MAX_SPEED:
                flan_flag = 1
            else :
                flan_flag = 2
            if flan_dx < -MAX_SPEED:
                flan_dx += DSPEED
        elif (pressed[pg.K_UP]):
            flan_dy -= DSPEED
            if flan_dy < -MAX_SPEED:
                flan_dy += DSPEED
        elif (pressed[pg.K_DOWN]):
            flan_dy += DSPEED
            if flan_dy > MAX_SPEED:
                flan_dy -= DSPEED


        for event in pg.event.get():
            if event.type in [pg.QUIT]:
                pg.quit()
                sys.exit()
            elif event.type in [pg.KEYDOWN]:
                if event.key == pg.K_RIGHT:
                    flan_flag = 3
                    flan_dx += DSPEED
                elif event.key == pg.K_LEFT:
                    flan_flag = 1
                    flan_dx -= DSPEED
                elif event.key == pg.K_UP:
                    flan_dy -= DSPEED
                elif event.key == pg.K_DOWN:
                    flan_dy += DSPEED
            elif event.type in [pg.KEYUP]:
                flan_flag = 0
                flan_dx = 0
                flan_dy = 0

        flan_rect.x += flan_dx
        flan_rect.y += flan_dy

        if flan_rect.x < 0:
            flan_rect.x = 0
        elif flan_rect.x + flan_size[0] > SCREEN_WIDTH:
            flan_rect.x = SCREEN_WIDTH - flan_size[0]

        if flan_rect.y < 0:
            flan_rect.y = 0
        elif flan_rect.y + flan_size[1] > SCREEN_HEIGHT:
            flan_rect.y = SCREEN_HEIGHT - flan_size[1]

        for bullet in bullet_list:
            bullet.move_bullet()

        bullet_clean()

        for bullet in bullet_list:
            if flan_rect.colliderect(bullet.rect):
                print("score: ", score)
                running = False

        gamepad.fill((0,0,0))
        draw_img(flan_image[flan_flag], flan_rect.x, flan_rect.y)
        for bullet in bullet_list:
            bullet.draw_bullet()
        tmp = pg.surfarray.array2d(gamepad)
        pg.display.flip()

if __name__ == "__main__":
    mode = "play"
    if mode == "play":
        for i in range(3):
            init_game()
            run()



