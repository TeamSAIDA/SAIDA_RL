# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

import numpy as np
import os
import math
from datetime import datetime
from core.common.processor import Processor
from core.algorithm.Deep_sarsa import DeepSARSAgent
from core.callbacks import DrawTrainMovingAvgPlotCallback
from saida_gym.starcraft.avoidReavers import AvoidReavers

from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

# Hyper Param
EPISODES = 50000
LEARNING_RATE = 0.0005
EPSILON = 1
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.01
# 5 features of a Dropship  + 6 features of 3 Reavers
STATE_SIZE = 5 + 3 * 6


def scale_velocity(v):
    return v


def scale_coordinate(pos):
    return np.tanh(int(pos / 16) / 20)


def scale_angle(angle):
    return (angle - math.pi) / math.pi


def scale_pos(pos):
    return int(pos / 16)


def scale_pos2(pos):
    return int(pos / 8)


# preprocess for observation
class ReaverProcessor(Processor):
    def __init__(self):
        self.last_action = None

    def process_action(self, action):
        self.last_action = action
        return action

    def process_step(self, observation, reward, done, info):
        state_array = self.process_observation(observation)
        reward = self.reward_reshape(reward)
        return state_array, reward, done, info

    def reward_reshape(self, reward):
        """ Reshape the reward

        # Argument
            reward (float): The observed reward after executing the action

        # Returns
            reshaped reward

        """
        return reward

    def process_observation(self, observation, **kwargs):
        """ Pre-process observation

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            processed observation

        """
        if len(observation.my_unit) > 0:
            s = np.zeros(STATE_SIZE)
            me = observation.my_unit[0]
            # Observation for Dropship
            s[0] = scale_pos(me.pos_x)  # X of coordinates
            s[1] = scale_pos(me.pos_y)  # Y of coordinates
            s[2] = scale_velocity(me.velocity_x)  # X of velocity
            s[3] = scale_velocity(me.velocity_y)  # y of coordinates
            s[4] = scale_angle(me.angle)  # Angle of head of dropship

            # Observation for Reavers
            for ind, ob in enumerate(observation.en_unit):
                s[ind * 6 + 5] = scale_pos(ob.pos_x - me.pos_x)  # X of relative coordinates
                s[ind * 6 + 6] = scale_pos(ob.pos_y - me.pos_y)  # Y of relative coordinates
                s[ind * 6 + 7] = scale_velocity(ob.velocity_x)  # X of velocity
                s[ind * 6 + 8] = scale_velocity(ob.velocity_y)  # Y of velocity
                s[ind * 6 + 9] = scale_angle(ob.angle)  # Angle of head of Reavers
                s[ind * 6 + 10] = scale_angle(1 if ob.accelerating else 0)  # True if Reaver is accelerating

        return s


if __name__ == '__main__':
    TRAINING_MODE = True
    LOAD_MODEL = False
    FILE_NAME = os.path.basename(__file__).split('.')[0] + "-" + datetime.now().strftime("%m%d%H%M%S")

    # Create an Environment
    # todo : need to substitute it with env.make() and also remove other parameters such as protobuf_name & verbose!?

    env = AvoidReavers(action_type=0, move_angle=30, move_dist=2, frames_per_step=24, verbose=0, no_gui=False)

    state_size = STATE_SIZE  # 내 정보 5 + 근처 10 개 옵저버 정보 5
    action_size = env.action_space.n

    # Create a model
    model = Sequential()
    model.add(Dense(50, input_dim=state_size, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.summary()

    # Create an Agent
    agent = DeepSARSAgent(action_size, processor=ReaverProcessor(), model=model,
                          epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN, discount_factor=0.99)

    agent.compile(Adam(lr=LEARNING_RATE))

    # For Graph
    cb_plot = DrawTrainMovingAvgPlotCallback('../../save_graph/' + FILE_NAME + '.png', 10, 5, l_label=['episode_reward'])

    # Run your Agent
    agent.run(env, EPISODES * 10, train_mode=TRAINING_MODE, verbose=2, callbacks=[cb_plot])

    agent.save_weights(FILE_NAME + '.h5f', False)

    env.close()