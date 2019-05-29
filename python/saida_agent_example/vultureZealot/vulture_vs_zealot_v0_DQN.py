# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

import os
import math
from collections import deque

from core.policies import EpsGreedyQPolicy, GreedyQPolicy, LinearAnnealedPolicy, AdvEpsGreedyPolicy
from core.common.processor import Processor
from core.algorithm.DQN import DQNAgent
from core.memories import SequentialMemory
from core.callbacks import DrawTrainMovingAvgPlotCallback, ModelIntervalCheckpoint
from core.common.callback import *
from saida_gym.starcraft.vultureVsZealot import VultureVsZealot

from keras.layers import Dense, Reshape
from keras.optimizers import Adam
from keras.models import Sequential

import importlib
import param

import argparse
from core.common.util import OPS

parser = argparse.ArgumentParser(description='DQN Configuration including setting dqn / double dqn / double dueling dqn')

parser.add_argument(OPS.NO_GUI.value, help='gui', type=bool, default=False)
parser.add_argument(OPS.DOUBLE.value, help='double dqn', default=False, action='store_true')
parser.add_argument(OPS.DUELING.value, help='dueling dqn', default=True, action='store_true')
parser.add_argument(OPS.BATCH_SIZE.value, type=int, default=700, help="batch size")
parser.add_argument(OPS.REPLAY_MEMORY_SIZE.value, type=int, default=1000000, help="replay memory size")
parser.add_argument(OPS.LEARNING_RATE.value, type=float, default=0.0005, help="learning rate")
parser.add_argument(OPS.TARGET_NETWORK_UPDATE.value, type=int, default=2000, help="target_network_update_interval")
parser.add_argument(OPS.WINDOW_LENGTH.value, type=int, default=2, help="window length")
parser.add_argument(OPS.DISCOUNT_FACTOR.value, type=float, default=0.99, help="discount factor")
parser.add_argument('-move-angle', type=int, default=15, help="move angle")

args = parser.parse_args()

dict_args = vars(args)
post_fix = ''
for k in dict_args.keys():
    if k == OPS.NO_GUI():
        continue
    post_fix += '_' + k + '_' + str(dict_args[k])

NO_GUI = dict_args[OPS.NO_GUI()]
DISCOUNT_FACTOR = dict_args[OPS.DISCOUNT_FACTOR()]
ENABLE_DOUBLE = dict_args[OPS.DOUBLE()]
ENABLE_DUELING = dict_args[OPS.DUELING()]
BATCH_SIZE = dict_args[OPS.BATCH_SIZE()]
REPLAY_BUFFER_SIZE = dict_args[OPS.REPLAY_MEMORY_SIZE()]
LEARNING_RATE = dict_args[OPS.LEARNING_RATE()]
TRAIN_INTERVAL=500
TARGET_MODEL_UPDATE_INTERVAL = dict_args[OPS.TARGET_NETWORK_UPDATE()]
WINDOW_LENGTH = dict_args[OPS.WINDOW_LENGTH()]
MAX_STEP_CNT = 10000000
DAMAGED_REWARD = -1
ATTACK_REWARD = 2.5
COOLDOWN_REWARD = 0
KILL_REWARD = 10
DEAD_REWARD = -10
INVALID_ACTION_REWARD = 0
MOVE_ANGLE = dict_args['move_angle']


class MyCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_episode_end(self, episode, logs={}):
        importlib.reload(param)


class ObsProcessor(Processor):
    def __init__(self, env):
        self.accumulated_observation = deque(maxlen=3)
        self.last_action = [-1]
        self.env = env

    def process_action(self, action):
        self.last_action = action
        return action

    def process_step(self, observation, reward, done, info):
        if done:
            if len(observation.my_unit) == 1:
                info['hps'] = observation.my_unit[0].hp
            elif len(observation.en_unit) == 1:
                info['hps'] = -observation.en_unit[0].hp - observation.en_unit[0].shield
            else:
                info['hps'] = 0

        state_array = self.process_observation(observation)

        reward = self.reward_reshape(state_array, reward, done, info)

        if param.verbose == 1 or param.verbose == 4:
            print('action : ', self.last_action[0], 'reward : ', reward, ('X' if info['infoMsg'].was_invalid_action[0] else 'O'))
        if param.verbose == 2 or param.verbose == 4:
            print('observation : ', observation)
        if param.verbose == 3 or param.verbose == 4:
            print('state_array : ', state_array)

        return state_array, reward, done, info

    def reward_reshape(self, observation, reward, done, info):
        if done:
            if reward > 0:
                reward = KILL_REWARD
            # 죽은 경우
            else:
                reward = DEAD_REWARD

        else:
            cooldown_zero = self.scale_cooldown(0)
            pre_cooldown = self.accumulated_observation[-2][2]
            cur_cooldown = observation[2]
            pre_hp = self.accumulated_observation[-2][3]
            cur_hp = observation[3]
            enemy_hp_idx = 10 + env.action_space.n - 1 + 5
            enemy_pre_hp = self.accumulated_observation[-2][enemy_hp_idx]
            enemy_cur_hp = observation[enemy_hp_idx]

            # 잘못된 Command
            # if np.all(self.accumulated_observation[-2] == self.accumulated_observation[-1]):
            #     reward = INVALID_ACTION_REWARD
            # 맞은 경우
            if pre_hp > cur_hp:
                reward = (pre_hp * 10 - cur_hp * 10) * DAMAGED_REWARD
            # 때린 경우
            #elif cur_cooldown > pre_cooldown:
            elif enemy_pre_hp > enemy_cur_hp:
                reward = math.ceil(enemy_pre_hp * 8 - enemy_cur_hp * 8) * ATTACK_REWARD
            # 쿨타임 없는 경우
            # elif cur_cooldown == cooldown_zero:
            #     # 쿨타임이 없는데 p컨을 안하면 마이너스
            #     reward = COOLDOWN_REWARD
            else:
                reward = -0.03

        return reward

    def process_observation(self, observation):
        state_array = np.zeros(10 + env.action_space.n - 1 + 11)

        # 64 x 64
        # unwalkable : -1 walkable : 0 enemy pos : 1
        #

        tmp_idx = 0
        my_x = 0
        my_y = 0
        for idx, me in enumerate(observation.my_unit):
            my_x = me.pos_x
            my_y = me.pos_y
            state_array[tmp_idx + 0] = math.atan2(me.velocity_y, me.velocity_x) / math.pi
            state_array[tmp_idx + 1] = self.scale_velocity(math.sqrt((me.velocity_x) ** 2 + (me.velocity_y) ** 2))
            state_array[tmp_idx + 2] = self.scale_cooldown(me.cooldown)
            state_array[tmp_idx + 3] = self.scale_vul_hp(me.hp)
            state_array[tmp_idx + 4] = self.scale_angle(me.angle)
            state_array[tmp_idx + 5] = self.scale_bool(me.accelerating)
            state_array[tmp_idx + 6] = self.scale_bool(me.braking)
            state_array[tmp_idx + 7] = self.scale_bool(me.attacking)
            state_array[tmp_idx + 8] = self.scale_bool(me.is_attack_frame)
            state_array[tmp_idx + 9] = self.last_action[0] / (env.action_space.n - 1)
            tmp_idx += 10
            for i, terrain in enumerate(me.pos_info):
                state_array[tmp_idx + i] = terrain.nearest_obstacle_dist / 320
            tmp_idx += len(me.pos_info)

        tmp_idx = 10 + env.action_space.n - 1

        for idx, enemy in enumerate(observation.en_unit):
            state_array[tmp_idx + 0] = math.atan2(enemy.pos_y - my_y, enemy.pos_x - my_x) / math.pi
            state_array[tmp_idx + 1] = self.scale_coordinate(math.sqrt((enemy.pos_x - my_x) ** 2 + (enemy.pos_y - my_y) ** 2))
            state_array[tmp_idx + 2] = math.atan2(enemy.velocity_y, enemy.velocity_x) / math.pi
            state_array[tmp_idx + 3] = self.scale_velocity(math.sqrt((enemy.velocity_x) ** 2 + (enemy.velocity_y) ** 2))
            state_array[tmp_idx + 4] = self.scale_cooldown(enemy.cooldown)
            state_array[tmp_idx + 5] = self.scale_zeal_hp(enemy.hp + enemy.shield)
            state_array[tmp_idx + 6] = self.scale_angle(enemy.angle)
            state_array[tmp_idx + 7] = self.scale_bool(enemy.accelerating)
            state_array[tmp_idx + 8] = self.scale_bool(enemy.braking)
            state_array[tmp_idx + 9] = self.scale_bool(enemy.attacking)
            state_array[tmp_idx + 10] = self.scale_bool(enemy.is_attack_frame)
            tmp_idx += 11

        self.accumulated_observation.append(state_array)

        return state_array

    @staticmethod
    def scale_velocity(v):
        return v / 6.4

    @staticmethod
    def scale_coordinate(pos):
        if pos > 0:
            return 1 if pos > 320 else int(pos / 16) / 20
        else:
            return -1 if pos < -320 else int(pos / 16) / 20

    @staticmethod
    def scale_angle(angle):
        return (angle - math.pi) / math.pi

    @staticmethod
    def scale_cooldown(cooldown):
        return (cooldown + 1) / 15

    @staticmethod
    def scale_vul_hp(hp):
        return hp / 80

    @staticmethod
    def scale_zeal_hp(hp):
        return hp / 160

    @staticmethod
    def scale_bool(boolean):
        return 1 if boolean else 0


if __name__ == '__main__':
    training_mode = True

    FILE_NAME = os.path.basename(__file__).split('.')[0]

    # todo : need to substitute it with env.make() and also remove other parameters such as protobuf_name & verbose!?
    env = VultureVsZealot(version=0, frames_per_step=6, action_type=0, move_angle=MOVE_ANGLE, move_dist=4, verbose=0, no_gui=NO_GUI
                          ,auto_kill=False)

    try:
        state_size = 10 + env.action_space.n - 1 + 11
        action_size = env.action_space.n

        print('state_size', state_size, 'action_size', action_size)

        model = Sequential()
        model.add(Reshape((state_size*WINDOW_LENGTH,), input_shape=(WINDOW_LENGTH, state_size)))
        model.add(Dense(50, input_dim=state_size*WINDOW_LENGTH, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()

        memory = SequentialMemory(limit=REPLAY_BUFFER_SIZE, window_length=WINDOW_LENGTH)

        policy = AdvEpsGreedyPolicy(value_max=1., value_min=.01, value_test=.05, nb_steps=2000000, min_score=-160
                                        , max_score=80, score_queue_size=50, score_name='hps', str_eps=0.4)
        test_policy = GreedyQPolicy()

        agent = DQNAgent(model, action_size, memory, processor=ObsProcessor(env=env), policy=policy, test_policy=test_policy
                         , enable_double=ENABLE_DOUBLE, enable_dueling=ENABLE_DUELING, train_interval=TRAIN_INTERVAL, discount_factor=DISCOUNT_FACTOR
                         , batch_size=BATCH_SIZE, target_model_update=TARGET_MODEL_UPDATE_INTERVAL)

        agent.compile(Adam(lr=LEARNING_RATE))

        callbacks = []

        if training_mode:
            cb_plot = DrawTrainMovingAvgPlotCallback(os.path.realpath('../../save_graph/' + FILE_NAME + '_{}' + post_fix + '.png')
                                                 , 100, 50, l_label=['episode_reward', 'hps'], save_raw_data=True)
            callbacks.append(cb_plot)
            my_callback = MyCallback()

            callbacks.append(my_callback)

            checkpoint = ModelIntervalCheckpoint(os.path.realpath('../../save_model/' + FILE_NAME + '{episode}.h5f')
                                    , condition=lambda e, l: l['info']['hps'] == 80, condition_count=10, verbose=1, agent=agent)
            callbacks.append(checkpoint)
        else:
            h5f = ''
            agent.load_weights(os.path.realpath(h5f + '.h5f'))

        agent.run(env, MAX_STEP_CNT, train_mode=training_mode, verbose=2, callbacks=callbacks)

        if training_mode:
            agent.save_weights(os.path.realpath('../../save_model/' + FILE_NAME + post_fix + '.h5f'), True)

    finally:
        env.close()
