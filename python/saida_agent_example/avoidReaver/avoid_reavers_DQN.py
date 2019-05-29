# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun


from core.policies import EpsGreedyQPolicy, GreedyQPolicy, LinearAnnealedPolicy
from core.algorithm.DQN import DQNAgent
from core.memories import SequentialMemory

from keras.layers import Dense, Reshape
from keras.models import Sequential

import numpy as np
import os
from core.common.processor import Processor
from saida_gym.starcraft.avoidReavers import AvoidReavers
from core.callbacks import DrawTrainMovingAvgPlotCallback

from keras.layers import Input, Dense
from keras.optimizers import Adam
import math


NO_GUI = True
DISCOUNT_FACTOR = 0.99
ENABLE_DOUBLE = False
ENABLE_DUELING = False
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 100000
LEARNING_RATE = 0.0005
TARGET_MODEL_UPDATE_INTERVAL = 100
WINDOW_LENGTH = 1
MAX_STEP_CNT = 3000000
STATE_SIZE = 8 + 3 * 8


def scale_velocity(v):
    return v


def scale_angle(angle):
    return (angle - math.pi) / math.pi


def scale_pos(pos):
    return pos / 16


def scale_pos2(pos):
    return pos / 8


def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


# Reshape the reward in a way you want
def reward_reshape(reward):
    """ Reshape the reward
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
        reward = -1
    elif reward == -1:
        reward = -10
    elif reward == 1:
        reward = 10
    elif reward == 0:
        reward = -0.1

    return reward


class ReaverProcessor(Processor):
    def __init__(self):
        self.last_action = None
        self.success_cnt = 0
        self.cumulate_reward = 0

    def process_action(self, action):
        self.last_action = action
        return action

    def process_step(self, observation, reward, done, info):
        state_array = self.process_observation(observation)
        reward = reward_reshape(reward)
        self.cumulate_reward += reward

        if reward == 10:
            if self.cumulate_reward > 0:
                self.success_cnt += 1

            self.cumulate_reward = 0
            print("success_cnt = ", self.success_cnt)

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
            s[0] = scale_pos2(me.pos_x)  # X of coordinates
            s[1] = scale_pos2(me.pos_y)  # Y of coordinates
            s[2] = scale_pos2(me.pos_x - 320)  # relative X of coordinates from goal
            s[3] = scale_pos2(me.pos_y - 320)  # relative Y of coordinates from goal
            s[4] = scale_velocity(me.velocity_x)  # X of velocity
            s[5] = scale_velocity(me.velocity_y)  # y of coordinates
            s[6] = scale_angle(me.angle)  # Angle of head of dropship
            s[7] = 1 if me.accelerating else 0  # True if Dropship is accelerating

            # Observation for Reavers
            for ind, ob in enumerate(observation.en_unit):
                s[ind * 8 + 8] = scale_pos2(ob.pos_x - me.pos_x)  # X of relative coordinates
                s[ind * 8 + 9] = scale_pos2(ob.pos_y - me.pos_y)  # Y of relative coordinates
                s[ind * 8 + 10] = scale_pos2(ob.pos_x - 320)  # X of relative coordinates
                s[ind * 8 + 11] = scale_pos2(ob.pos_y - 320)  # Y of relative coordinates
                s[ind * 8 + 12] = scale_velocity(ob.velocity_x)  # X of velocity
                s[ind * 8 + 13] = scale_velocity(ob.velocity_y)  # Y of velocity
                s[ind * 8 + 14] = scale_angle(ob.angle)  # Angle of head of Reavers
                s[ind * 8 + 15] = 1 if ob.accelerating else 0  # True if Reaver is accelerating

        return s


if __name__ == '__main__':
    training_mode = True
    load_model = False
    FILE_NAME = os.path.basename(__file__).split('.')[0]
    action_type = 0
    # todo : need to substitute it with env.make() and also remove other parameters such as protobuf_name & verbose!?
    env = AvoidReavers(move_angle=15, move_dist=2, frames_per_step=16, verbose=0, action_type=action_type)

    try:

        state_size = STATE_SIZE
        action_size = env.action_space.n

        model = Sequential()
        model.add(Reshape((state_size*WINDOW_LENGTH,), input_shape=(WINDOW_LENGTH, state_size)))
        model.add(Dense(50, input_dim=state_size*WINDOW_LENGTH, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()

        memory = SequentialMemory(limit=REPLAY_BUFFER_SIZE, window_length=WINDOW_LENGTH, enable_per=False, per_alpha=0.6)

        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.05, nb_steps=400000)
        test_policy = GreedyQPolicy()

        agent = DQNAgent(model, action_size, memory, processor=ReaverProcessor(), policy=policy, test_policy=test_policy
                         , enable_double=ENABLE_DOUBLE, enable_dueling=ENABLE_DUELING, train_interval=80, discount_factor=DISCOUNT_FACTOR
                         , batch_size=BATCH_SIZE, target_model_update=TARGET_MODEL_UPDATE_INTERVAL)

        agent.compile(Adam(lr=LEARNING_RATE))

        callbacks = []

        if training_mode:
            cb_plot = DrawTrainMovingAvgPlotCallback(
                os.path.realpath('../../save_graph/' + FILE_NAME + '.png'), 5, 5, l_label=['episode_reward'])
            callbacks.append(cb_plot)
            # h5f = 'vulture_vs_zealot_v0_DQN_double_False_dueling_True_batch_size_128_repm_size_200000_lr_0.001_tn_u_invl_100_window_length_2'
            # agent.load_weights(os.path.realpath('save_model/' + h5f + '.h5f'))
        else:
            h5f = ''
            agent.load_weights(os.path.realpath('../../save_model/' + h5f + '.h5f'))

        agent.run(env, MAX_STEP_CNT, train_mode=training_mode, verbose=2, callbacks=callbacks)

        if training_mode:
            agent.save_weights(os.path.realpath('../../save_model/' + FILE_NAME + '.h5f'), True)

    finally:
        env.close()
