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
from core.algorithm.REINFORCE import ReinforceAgent
from core.callbacks import DrawTrainMovingAvgPlotCallback
from saida_gym.starcraft.avoidReavers import AvoidReavers
import saida_gym.envs.conn.connection_env as Config

from keras.layers import Dense, Reshape
from keras.optimizers import Adam
from keras.models import Sequential

# Hyper Parameter
EPISODES = 5000
LEARNING_RATE = 0.0004
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
def reward_reshape(reward):
    """Reshape the reward

        Starcraft Env returns the reward according to following conditions.

        1. Invalid action : -0.1
        2. get hit : -1
        3. goal : 1
        4. others : 0

    # Argument
        reward (float): The observed reward after executing the action

    # Returns
        reshaped reward
    """
    if math.fabs(reward + 0.1) < 0.01:
        reward = -5
    elif reward == 0:
        reward = -0.1
    elif reward == -1:
        reward = -3
    elif reward == 1:
        reward = 2

    return reward


class ReaverProcessor(Processor):
    def __init__(self):
        self.last_action = None

    def process_action(self, action):
        self.last_action = action
        return action

    def process_step(self, observation, reward, done, info):
        state_array = self.process_observation(observation)
        reward = reward_reshape(reward)
        return state_array, reward, done, info

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
    FILE_NAME = os.path.basename(__file__).split('.')[0] + "-" + datetime.now().strftime("%m%d%H%M%S")

    # todo : need to substitute it with env.make() and also remove other parameters such as protobuf_name & verbose!?
    # Create an Environment
    env = AvoidReavers( move_angle=30, move_dist=3, frames_per_step=24)
    action_size = env.action_space.n

    # Create your model
    model = Sequential()
    model.add(Dense(70, input_dim=STATE_SIZE, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(action_size, activation='softmax'))
    model.summary()

    # Create your Agent
    agent = ReinforceAgent(STATE_SIZE, action_size, processor=ReaverProcessor(), model=model,
                           discount_factor=0.9)

    agent.compile(Adam(lr=LEARNING_RATE))

    # For the Graph
    cb_plot = DrawTrainMovingAvgPlotCallback('../../save_graph/' + FILE_NAME + '.png', 10, 5, l_label=['episode_reward'])

    # Run your agent
    agent.run(env, EPISODES * 100, train_mode=TRAINING_MODE, verbose=2, callbacks=[cb_plot])

    agent.save_weights(FILE_NAME + '.h5f', False)

    env.close()
