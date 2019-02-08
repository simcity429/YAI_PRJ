import Env_Game
from Env_Game import Env
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.layers import LSTM, Conv2D, InputLayer, TimeDistributed, Flatten, Dense, Input, Reshape
from tensorflow.contrib.keras.api.keras.optimizers import RMSprop
from tensorflow.contrib.keras.api.keras.models import Model

import numpy as np
from PIL import ImageOps
from PIL import Image
from time import sleep
import threading
RESIZE = 84
THREAD_NUM = 4
SEQUENCE_SIZE = 8
STATE_SIZE = (SEQUENCE_SIZE, RESIZE, RESIZE)
ACTION_SIZE = Env_Game.ACTION_SIZE

def preprocess(arr):
    #returns preprocessed image
    return ImageOps.mirror(Image.fromarray(arr).rotate(270)).convert('L').resize((RESIZE, RESIZE))

class A3CAgent:
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, sequence_size=SEQUENCE_SIZE, thread_num=THREAD_NUM):
        self.state_size = state_size
        self.action_size = action_size
        #hyperparameter
        self.discount_factor = 0.99
        self.stop_step = sequence_size
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4

        self.thread_num = thread_num

        self.actor, self.critic = self.build_model()


    def build_model(self):
        #for reshaping tensor
        shape = list(self.state_size)
        shape.append(1)
        input = Input(shape=self.state_size)
        reshaped = Reshape(shape)(input)
        conv  = TimeDistributed(Conv2D(16, (8, 8), strides=(4, 4), activation="relu"))(reshaped)
        conv = TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), activation="relu"))(conv)
        conv = TimeDistributed(Flatten())(conv)
        lstm = LSTM(512, activation='tanh')(conv)

        policy = Dense(self.action_size, activation="softmax")(lstm)
        value = Dense(1, activation='linear')(lstm)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

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
        cross_entropy = -K.sum(cross_entropy)

        # add (-entropy) to loss function, for enthusiastic search
        minus_entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        minus_entropy = K.sum(minus_entropy)

        # optimizing loss minimizes cross_entropy, maximizes entropy
        loss = cross_entropy + 0.01 * minus_entropy

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # loss = MSE(discounted_prediction, value)
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
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
    def __init__(self):
        threading.Thread.__init__(self)




tmp = A3CAgent()
tmp.build_model()


class Agent(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def build_model(self):
        pass



"""
np.set_printoptions(threshold=np.nan)
e = Env(True)
e.reset()
while(True):
    a = np.random.randint(0, 5)
    sleep(0.05)
    o, r, d, _ = e.step(a)
    im = preprocess(o)
    if d:
        print(np.asarray(im).shape)
        im.show()
        sleep(5)
        break
"""