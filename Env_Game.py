import pygame as pg
import sys
import numpy.random as r
from matplotlib import pylab as plt

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
SKIP_FRAMES = 2
MAX_SPEED = 20
BULLET_MAX_SPEED = 12
DSPEED = 2
PROBABILITY = 4
ACTION_SIZE = 5

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
clock = pg.time.Clock()

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

    def draw_bullet(self, gamepad):
        global bullet_image
        draw_img(gamepad, bullet_image, self.rect.x, self.rect.y)

class State:
    def __init__(self, gamepad):
        global flan_size
        flan_loc = [(SCREEN_WIDTH - (flan_size[0] / 2)) / 2, (SCREEN_HEIGHT - (flan_size[1] / 2)) / 2]
        self.gamepad = gamepad
        self.flan_rect = pg.Rect(flan_loc, (flan_size[0] * 0.5, flan_size[1] * 0.5))
        self.flan_flag = 0
        self.flan_dx = 0
        self.flan_dy = 0
        self.score = 0
        self.bullet_list = []

def bullet_create(bullet_list):
    bullet = Bullet()
    bullet_list.append(bullet)

def bullet_clean(bullet_list):
    margin = bullet_size[0]/2
    delete_bullet = []
    for bullet in bullet_list:
        if (bullet.rect.x < -margin or bullet.rect.x > SCREEN_WIDTH + margin) or (bullet.rect.y < -margin or bullet.rect.y > SCREEN_HEIGHT + margin):
            delete_bullet.append(bullet)
    bullet_list = [bullet for bullet in bullet_list if not (bullet in delete_bullet)]
    for bullet in delete_bullet:
        del bullet
    return bullet_list


def draw_img(gamepad, img, x, y):
    gamepad.blit(img, (x, y))

def init_game(render):
    pg.init()
    if render:
        gamepad = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    else:
        gamepad = pg.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    return gamepad

def run(gamepad):
    global clock
    global flan_image
    global flan_size
    bullet_list = []
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
        r_num = r.randint(0, PROBABILITY)
        if (r_num == 1):
            bullet_create(bullet_list)

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

        bullet_list = bullet_clean(bullet_list)

        for bullet in bullet_list:
            if flan_rect.colliderect(bullet.rect):
                print("score: ", score)
                running = False

        gamepad.fill((0,0,0))
        draw_img(gamepad, flan_image[flan_flag], flan_rect.x, flan_rect.y)
        for bullet in bullet_list:
            bullet.draw_bullet(gamepad)
        pg.display.flip()

class Env:
    def __init__(self, render):
        self.action_size = 5

        self.render = render

    def reset(self):
        gamepad = init_game(self.render)
        self.state = State(gamepad)
        self.state.gamepad.fill((0, 0, 0))
        draw_img(self.state.gamepad, flan_image[self.state.flan_flag], self.state.flan_rect.x, self.state.flan_rect.y)
        ret_state = pg.surfarray.array3d(self.state.gamepad)
        if self.render:
            pg.display.flip()
        return ret_state, 0, False, None

    def step(self, action):
        pg.event.pump()
        state = self.state
        global flan_size, flan_image
        for _ in range(SKIP_FRAMES):
            # bullet create
            r_num = r.randint(0, PROBABILITY)
            if (r_num == 1):
                bullet_create(state.bullet_list)
            # action = {0 : stop, 1: right, 2: left, 3: up, 4: down}
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
            flag = False
            if state.flan_rect.x < 0:
                state.flan_rect.x = 0
                flag = True
            elif state.flan_rect.x + flan_size[0] > SCREEN_WIDTH:
                state.flan_rect.x = SCREEN_WIDTH - flan_size[0]
                flag = True

            if state.flan_rect.y < 0:
                state.flan_rect.y = 0
                flag = True
            elif state.flan_rect.y + flan_size[1] > SCREEN_HEIGHT:
                state.flan_rect.y = SCREEN_HEIGHT - flan_size[1]
                flag = True

            for bullet in state.bullet_list:
                bullet.move_bullet()

            state.bullet_list = bullet_clean(state.bullet_list)

            state.gamepad.fill((0, 0, 0))
            draw_img(state.gamepad, flan_image[state.flan_flag], state.flan_rect.x, state.flan_rect.y)

            for bullet in state.bullet_list:
                bullet.draw_bullet(state.gamepad)
            ret_state = pg.surfarray.array3d(state.gamepad)
            if self.render:
                pg.display.flip()

            for bullet in state.bullet_list:
                if state.flan_rect.colliderect(bullet.rect):
#                    print("score: ", state.score)
                    return ret_state, -0.1, True, state.score

            if flag:
                return ret_state, -0.1, True, state.score
            state.score += 0.01
        return ret_state, 0.01, False, state.score

if __name__ == "__main__":
    mode = "random"
    iter = 100
    if mode == "play":
        iter = 10
        gamepad = init_game(True)
        for _ in range(iter):
            run(gamepad)
    else : #random auto play
        score_list = []
        episode_list = []
        e = Env(render=True)
        for episode in range(iter):
            e.reset()
            while True:
                action = r.randint(0, 5)
                _, _, done, score = e.step(action)
                if done:
                    episode_list.append(episode)
                    score_list.append(score)
                    break
        fig, axe = plt.subplots()
        axe.plot(episode_list, score_list)
        fig.savefig('./random_agent_statistics')
        print('average score of a random agent: ', sum(score_list)/iter)



