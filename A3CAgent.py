import Env_Game
from Env_Game import Env
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.layers import BatchNormalization, Conv2D, InputLayer, TimeDistributed, Flatten, Dense, Input, Reshape, GRU
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.optimizers import Adam

import numpy as np
from matplotlib import pylab as plt
from PIL import ImageOps
from PIL import Image
from time import sleep
import threading
import csv
RESIZE = 84
THREAD_NUM = 40
SEQUENCE_SIZE = 5
STATE_SIZE = (SEQUENCE_SIZE, RESIZE, RESIZE, 3)
ACTION_SIZE = Env_Game.ACTION_SIZE
EPISODES = 800000
episode = 0
global_p_max = []
global_score = []
global_episode = []
global_actor_loss = []
global_critic_loss = []

def recent_average(l):
    if len(l) < 100:
        return sum(l)/len(l)
    a = l[-100:len(l)]
    return sum(a)/100

def preprocess(arr):
    #returns preprocessed image
    im = Image.fromarray(arr)
    return np.asarray(ImageOps.mirror(im.rotate(270)).resize((RESIZE, RESIZE)))

class A3CAgent:
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, sequence_size=SEQUENCE_SIZE, thread_num=THREAD_NUM, resume=True, play=False):
        self.state_size = state_size
        self.action_size = action_size
        #hyperparameter
        self.discount_factor = 0.95
        self.stop_step = sequence_size
        self.actor_lr = 0.00001
        self.critic_lr = 0.00001

        self.thread_num = thread_num

        self.actor, self.critic = self.build_model()
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        if play:
            self.load_model("./save_best/touhou_a3c")
            print('successfully loaded best weight')

        if resume and not play:
            self.load_model("./save_model/touhou_a3c")
            print('successfully loaded')

    def train(self):
        global global_p_max, global_critic_loss, global_actor_loss, global_score, global_episode
        # creating agents
        tmp_list = [not bool(i) for i in range(self.thread_num)]

        agents = [Agent(self.action_size, self.state_size, [self.actor, self.critic], self.optimizer, self.discount_factor, render)
                  for render in tmp_list]

        # starts threads
        for agent in agents:
            sleep(1)
            agent.start()

        f = open('output.csv', 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow(['index', 'episode', 'score', 'p_max_avg', 'actor_loss', 'critic_loss'])
        f.close()
        cnt = 0
        max_score = 0
        # saving model
        while True:
            try:
                sleep(60)
                print('saving model')
                f = open('output.csv', 'a', encoding='utf-8', newline="")
                self.save_model("./save_model/touhou_a3c")
                wr = csv.writer(f)
                current_episode = global_episode[-1]
                avg_score = recent_average(global_score)
                avg_pmax = recent_average(global_p_max)
                avg_al = recent_average(global_actor_loss)
                avg_cl = recent_average(global_critic_loss)
                if avg_score > max_score:
                    max_score = avg_score
                    print('max_score: ', max_score)
                    self.save_model("./save_best/touhou_a3c")
                newline = [cnt, current_episode, avg_score, avg_pmax, avg_al, avg_cl]
                wr.writerow(newline)
                f.close()
                cnt += 1
                print('successfully saved')
            except Exception as ex:
                print(ex)
                print('saving fail, but continue')

    def play(self):
        agent = Agent(self.action_size, self.state_size, [self.actor, self.critic], self.optimizer, self.discount_factor, True)
        agent.play()

    def build_model(self):
        input = Input(shape=self.state_size)
        conv  = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), activation="elu"))(input)
        conv = TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), activation="elu"))(conv)
        conv = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='elu'))(conv)
        conv = TimeDistributed(Conv2D(8, (1,1), strides=(1, 1), activation='elu'))(conv)
        conv = TimeDistributed(Flatten())(conv)
        conv = BatchNormalization()(conv)
        lstm = GRU(256, activation='tanh')(conv)

        policy = Dense(self.action_size, activation="softmax")(lstm)
        value = Dense(1, activation='linear')(lstm)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()


        actor.summary()
        critic.summary()

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None,])
        #advatages -> *multi-step*

        policy = self.actor.output

        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.mean(cross_entropy)


        # add (-entropy) to loss function, for enthusiastic search
        minus_entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        minus_entropy = K.mean(minus_entropy)

        # optimizing loss minimizes cross_entropy, maximizes entropy
        loss = cross_entropy  #+ 0.01 * minus_entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(loss, self.actor.trainable_weights)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # loss = MSE(discounted_prediction, value)
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(loss, self.critic.trainable_weights)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")


class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, optimizer, discount_factor, render):
        threading.Thread.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.render = render

        self.states, self.rewards, self.actions = [], [], []

        self.local_actor, self.local_critic = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0
        self.score = 0

        self.t_max = 20
        self.t = 0

    def run(self):
        global episode, global_score, global_p_max, global_episode
        global global_actor_loss, global_critic_loss
        env = Env(self.render)
        np.set_printoptions(threshold=np.nan)
        step = 0
        actor_loss, critic_loss = [], []
        while True:

            observe, reward, done, _ = env.reset()
            state = preprocess(observe).reshape((1, RESIZE, RESIZE, 3))
            history = np.copy(state)
            for _ in range(SEQUENCE_SIZE - 1):
                history = np.append(history, state, axis=0)
                state = np.copy(state)
            history = np.reshape([history], (1, SEQUENCE_SIZE, RESIZE, RESIZE, 3))
            #history.shape = (1, SEQUENCE_SIZE, RESIZE, RESIZE, 3)

            while not done:
                step += 1
                self.t += 1

                #choose action, get policy
                action, policy = self.get_action(history)




                # interact
                observe, reward, done, score = env.step(action)

                # preprocessing, history update
                next_state = preprocess(observe)
                next_state = np.reshape([next_state], (1, 1, RESIZE, RESIZE, 3))
                next_history = np.append(history[:, 1:, :, :, :], next_state, axis=1)

                # milestone: avg_p_max
                self.avg_p_max += np.amax(self.actor.predict(np.float32(history / 255.)))


                # store history
                self.append_sample(history, action, reward)


                history = next_history

                # training logic
                if self.t >= self.t_max or done:
                    a, c = self.train_model(next_history, done)
                    actor_loss.append(a[0])
                    critic_loss.append(c[0])
                    self.update_local_model()
                    self.t = 0

                if done:
                    # reporting information
                    episode += 1
                    self.score = score

                    print("episode:", episode, "  score:", self.score, "  step:",step, "avg_p_max: ", self.avg_p_max/float(step), " actor loss: ", sum(actor_loss)/step, " critic loss: ", sum(critic_loss)/step )
                    global_score.append(self.score)
                    global_p_max.append(self.avg_p_max/float(step))
                    global_episode.append(episode)
                    global_actor_loss.append(sum(actor_loss)/step)
                    global_critic_loss.append(sum(critic_loss)/step)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    self.score = 0
                    actor_loss = []
                    critic_loss = []
                    step = 0

    def build_local_model(self):
        input = Input(shape=self.state_size)
        conv = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), activation="elu"))(input)
        conv = TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), activation="elu"))(conv)
        conv = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='elu'))(conv)
        conv = TimeDistributed(Conv2D(8, (1,1), strides=(1, 1), activation='elu'))(conv)
        conv = TimeDistributed(Flatten())(conv)
        conv = BatchNormalization()(conv)
        lstm = GRU(256, activation='tanh')(conv)

        policy = Dense(self.action_size, activation="softmax")(lstm)
        value = Dense(1, activation='linear')(lstm)

        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        return local_actor, local_critic

    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def discounted_prediction(self, next_history, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            running_add = self.critic.predict(np.float32(next_history / 255.))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    def train_model(self, next_history, done):
        discounted_prediction = self.discounted_prediction(next_history, self.rewards, done)
#        print('discounted prediction: ', discounted_prediction)
        states = np.zeros((len(self.states),SEQUENCE_SIZE, RESIZE, RESIZE, 3))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states / 255.)

        values = self.local_critic.predict(states)
        values = np.reshape(values, len(values))

#        print('values: ', values)

        advantages = discounted_prediction - values

#        print('advantages: ', advantages)
#        print('actions: ', self.actions)
        action_loss = self.optimizer[0]([states, self.actions, advantages])
        critic_loss = self.optimizer[1]([states, discounted_prediction])
        self.states, self.actions, self.rewards = [], [], []
        return action_loss, critic_loss

    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.local_actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def play(self):
        env = Env(self.render)

        step = 0
        EPISODES = 50
        episode = 0
        score_list = []
        episode_list = []
        while episode < EPISODES:
            self.score = 0
            observe, reward, done, _ = env.reset()
            state = preprocess(observe).reshape((1, RESIZE, RESIZE, 3))
            history = np.copy(state)
            for _ in range(SEQUENCE_SIZE - 1):
                history = np.append(history, state, axis=0)
                state = np.copy(state)
            history = np.reshape([history], (1, SEQUENCE_SIZE, RESIZE, RESIZE, 3))
            #history.shape = (1, SEQUENCE_SIZE, RESIZE, RESIZE, 3)

            while not done:
                sleep(0.05)
                step += 1

                #choose action, get policy
                action, policy = self.get_action(history)



                # interact
                observe, reward, done, score = env.step(action)
                # preprocessing, history update
                next_state = preprocess(observe)
                next_state = np.reshape([next_state], (1, 1, RESIZE, RESIZE, 3))
                history = np.append(history[:, 1:, :, :, :], next_state, axis=1)

                self.score = score


                if done:
                    # reporting information
                    episode_list.append(episode)
                    score_list.append(self.score)
                    episode += 1
                    print("episode:", episode, "  score:", self.score, "  step:",step)
                    self.score = 0
                    step = 0
        fig, axe = plt.subplots()
        axe.plot(episode_list, score_list)
        fig.savefig("./play_statistics.png")
        print('average score of a agent: ', sum(score_list)/len(score_list))

if __name__ == "__main__":
    mode = "train"
    if mode == "train":
        global_agent = A3CAgent(resume=True)
        global_agent.train()