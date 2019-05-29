"""
LSTM version with global and local observation

Observation design
1. instead of using spatial map #1 with Observers' location in local area within 12 tiles from scurge,
   two vectors with lengths 64 having one hot encoding value to represent scurge' absoulte path.
   This value will be combined with last action vectors

2. spatial map #2 with Scurge's location in entire map

3. last action

lstm time step 10 based on below
"DRQN is trained using backpropagation through time for the last ten timesteps. Thus both the non-recurrent 10-frame DQN and the recurrent 1-frame DRQN have access to the same history of game screens.3 Thus, when dealing with partial observability, a choice exists between using a non- recurrent deep network with a long history of observations or using a recurrent network trained with a single observa- tion at each timestep. The results in this section show that recurrent networks can integrate information through time and serve as a viable alternative to stacking frames in the input layer of a convoluational network"
"""

from saida_gym.starcraft.avoidObservers import AvoidObservers
import saida_gym.envs.conn.connection_env as Config
from keras.optimizers import Adam
from keras.layers import Dense, Input, Concatenate, Conv2D, Flatten, TimeDistributed, LSTM
from core.common.processor import Processor
from core.callbacks import DrawTrainMovingAvgPlotCallback
from core.algorithm.DDPG import DDPGAgent
from core.memories import SequentialMemory
from core.common.util import *
from core.common.random import *
from core.policies import *
import sys
import argparse
from core.common.util import OPS
# note: add l2 regularizers for critic layers
from keras import regularizers

import os
import numpy as np
from keras.models import Model


"""
====================================================================== 
FOR hyper parameters combination parameter setting from auto launcher
======================================================================   
"""
parser = argparse.ArgumentParser(description='DQN Configuration including setting dqn / double dqn / double dueling dqn')

parser.add_argument(OPS.NO_GUI.value, help='gui', type=bool, default=False)
parser.add_argument(OPS.USE_PARAMETERIZED_NOISE.value, type=bool, default=False, help="Parameterized Action noise")
parser.add_argument(OPS.FRAMES_PER_STEP.value, help='frames', type=int, default=4)
parser.add_argument(OPS.BATCH_SIZE.value, type=int, default=128, help="batch size")
parser.add_argument(OPS.REPLAY_MEMORY_SIZE.value, type=int, default=8000, help="replay memory size")
parser.add_argument(OPS.LEARNING_ACTOR_RATE.value, type=float, default=1e-4, help="actor learning rate")
parser.add_argument(OPS.LEARNING_CRITIC_RATE.value, type=float, default=1e-3, help="critic learning rate")
parser.add_argument(OPS.TARGET_NETWORK_UPDATE.value, type=int, default=60, help="target_network_update_interval")
parser.add_argument(OPS.N_STEPS.value, type=int, default=100000, help="n steps for training")
parser.add_argument(OPS.ACTION_REPETITION.value, type=int, default=1, help="Action repetition")
parser.add_argument(OPS.OU_SIGMA.value, type=float, default=0.1, help="Random noise parameter")
parser.add_argument(OPS.OU_THETA.value, type=float, default=0.15, help="Random noise parameter")
parser.add_argument(OPS.TIME_WINDOW.value, type=int, default=2, help="Temporal Splice Size")
parser.add_argument(OPS.MARGINAL_SPACE.value, type=int, default=1, help="Value for marginal space surrounded by safe zone to give bad panelty if a agent die")

args = parser.parse_args()

dict_args = vars(args)
post_fix = ''
for k in dict_args.keys():
    if k == 'no_gui' or  k == 'nsteps' or k == 'critic_lr' or k == 'actor_lr' or k == 'ou_theta':
        continue
    post_fix += '_' + k + '_' + str(dict_args[k])

print('post_fix : {}'.format(post_fix))

CURRENT_FILE_NAME = os.path.basename(__file__).split('.')[0]
CURRENT_FILE_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])

yyyymmdd = mmdd24hhmmss()

# FILE_NAME_FOR_LOG = os.path.basename(__file__).split('.')[0] + "_" + yyyymmdd
FILE_NAME_FOR_LOG = os.path.basename(__file__).split('.')[0] + "_" + yyyymmdd + post_fix


# For training
TIME_WINDOW = dict_args[OPS.TIME_WINDOW()]
BATCH_SIZE = dict_args[OPS.BATCH_SIZE()]
MOVING_AVERAGE_WINDOW = 100
PLOT_EPISODE_INTERVAL = 200
MARGINAL_SPACE = dict_args[OPS.MARGINAL_SPACE()]  # tile unit

STATE1_SIZE = (TIME_WINDOW, 64, 64, 1)  # If you never set it, then it will be "channels_last".
STATE2_SIZE = (TIME_WINDOW, 20, 20, 1)

ACTION_SIZE = 1
TRAINING_MODE = True

CRITIC_L2_REG = 0.01
REWARD_SCALE = 1
LOCAL_OBSERVABLE_TILE_SIZE = 10


env = AvoidObservers( action_type=2, verbose=0, frames_per_step=dict_args[OPS.FRAMES_PER_STEP()], no_gui=dict_args[OPS.NO_GUI()])

observation_input = [Input(shape=STATE1_SIZE, name='scurge_observation_input'), Input(shape=STATE2_SIZE, name='observer_observation_input')]
action_input = Input(shape=(ACTION_SIZE, ), name='action_input')

"""
inspired by here 
https://www.reddit.com/r/MachineLearning/comments/8etje4/d_actor_critic_algorithm_why_we_can_share/
https://github.com/yanpanlau/DDPG-Keras-Torcs/issues/11

here actor and actor network shared convolution layer.  
"""

# todo : confirm whether or not we can apply l2 regularization to shared layers between critic and actor network
SHARED_CONV2D_1_1 = TimeDistributed(Conv2D(10, kernel_size=5, strides=1, activation='relu', padding='SAME', kernel_regularizer=regularizers.l2(CRITIC_L2_REG)))
SHARED_CONV2D_1_2 = TimeDistributed(Conv2D(5, kernel_size=3, strides=1, activation='relu', padding='SAME', kernel_regularizer=regularizers.l2(CRITIC_L2_REG)))
SHARED_FLATTEN_1 = TimeDistributed(Flatten())

SHARED_CONV2D_2_1 = TimeDistributed(Conv2D(10, kernel_size=4, strides=1, activation='relu', padding='SAME', kernel_regularizer=regularizers.l2(CRITIC_L2_REG)))
SHARED_CONV2D_2_2 = TimeDistributed(Conv2D(5, kernel_size=3, strides=1, activation='relu', padding='SAME', kernel_regularizer=regularizers.l2(CRITIC_L2_REG)))
SHARED_FLATTEN_2 = TimeDistributed(Flatten())

SHARED_CONCATENATED = Concatenate()


def build_critic_model():
    oh1 = SHARED_CONV2D_1_1(observation_input[0])
    oh1 = SHARED_CONV2D_1_2(oh1)
    oh1 = SHARED_FLATTEN_1(oh1)

    oh2 = SHARED_CONV2D_2_1(observation_input[1])
    oh2 = SHARED_CONV2D_2_2(oh2)
    oh2 = SHARED_FLATTEN_2(oh2)

    oh = SHARED_CONCATENATED([oh1, oh2])
    oh = LSTM(512)(oh)

    ah = Dense(30, activation='relu', kernel_regularizer=regularizers.l2(CRITIC_L2_REG))(action_input)
    ah = Dense(20, activation='relu', kernel_regularizer=regularizers.l2(CRITIC_L2_REG))(ah)

    h = Concatenate()([oh, ah])
    h = Dense(30, activation='relu', kernel_regularizer=regularizers.l2(CRITIC_L2_REG))(h)
    h = Dense(20, activation='relu', kernel_regularizer=regularizers.l2(CRITIC_L2_REG))(h)
    output = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(CRITIC_L2_REG))(h)

    model = Model(inputs=[observation_input[0], observation_input[1], action_input], outputs=[output])
    model.summary()

    return model


def build_actor_model():
    oh1 = SHARED_CONV2D_1_1(observation_input[0])
    oh1 = SHARED_CONV2D_1_2(oh1)
    oh1 = SHARED_FLATTEN_1(oh1)

    oh2 = SHARED_CONV2D_2_1(observation_input[1])
    oh2 = SHARED_CONV2D_2_2(oh2)
    oh2 = SHARED_FLATTEN_2(oh2)

    oh = SHARED_CONCATENATED([oh1, oh2])

    h = TimeDistributed(Dense(30, activation='relu'))(oh)
    h = LSTM(512)(h)
    h = Dense(20, activation='relu')(h)

    output = Dense(ACTION_SIZE, activation='sigmoid')(h)
    model = Model(inputs=observation_input, outputs=[output])
    model.summary()

    return model


class ObsProcessor(Processor):

    def __init__(self, **kwargs):
        super(ObsProcessor, self).__init__(**kwargs)

        # self.accumulated_observation = deque(maxlen=3)
        # self.state_size = state_size
        self.highest_height = 1900
        # self.deque_last_actions = deque(maxlen=TIME_WINDOW)
        self.last_action = None

    def process_step(self, observation, reward, done, info):
        state = self.process_observation(observation)
        reward = self.reward_shape(observation, done)
        return state, reward, done, info

    def reward_shape(self, observation, done):
        """
        reward range
        :param observation:
        :param done:
        :return:
        """
        # Goal 에 도달하거나 죽으면
        if done:
            self.highest_height = 1900
            if 0 < observation.my_unit[0].pos_y and observation.my_unit[0].pos_y < 65 + MARGINAL_SPACE:
                return 10 * REWARD_SCALE
            # Safe zone : left-top (896, 1888) right-bottom (1056, 2048) with additional (marginal) space -> more penalty
            elif 896 - 32*MARGINAL_SPACE >= observation.my_unit[0].pos_x and observation.my_unit[0].pos_x <= 1056 + 32*MARGINAL_SPACE and observation.my_unit[0].pos_y >= 1888 - 32*MARGINAL_SPACE:
                return -10 * REWARD_SCALE
            return -5 * REWARD_SCALE

        # give important weight per height rank
        # 0 ~ 1888(59 tiles) / 32 : ratio
        if observation.my_unit[0].pos_y < self.highest_height:
            rank = int(observation.my_unit[0].pos_y / 32)  # 2 ~ 59
            weight = (59 / (rank + sys.float_info.epsilon)) / 59
            self.highest_height = observation.my_unit[0].pos_y
            return weight * 3 * REWARD_SCALE

        # 시간이 지나면
        return -0.02 * REWARD_SCALE


    def process_observation(self, observation, **kwargs):
        # making spatial map with size 10 x 10 and its channels include velocity_x, velocity_y, angle, accelerating_yn
        # we change this shape ignoring velocity_x, velocity_y, angle, accelerating_yn. Instead,
        # we give another spatial map which includes flags for walkable(out of map), safe zone
        map_of_scurge = np.zeros(shape=(64, 64))  # this map is for scurge (64, 64), global observation
        map_of_observer = np.zeros(shape=(LOCAL_OBSERVABLE_TILE_SIZE*2, LOCAL_OBSERVABLE_TILE_SIZE*2))  # observer map, local observation centered by sculge's position
        # map_of_map = np.zeros(shape=(64, 64))  # this map is

        me_x = observation.my_unit[0].pos_x
        me_y = observation.my_unit[0].pos_y

        me_x_t = np.clip(int(observation.my_unit[0].pos_x/32), 0, 64)
        me_y_t = np.clip(int(observation.my_unit[0].pos_y/32), 0, 64)

        # Safe zone : left-top (0, 0) right-bottom (64*32, 2048) with additional (marginal) space
        for x in range(int(896/32), int(1056/32)):
            for y in range(int(1888/32), int(2048/32)):
                # map_of_scurge[x][y] = -1  # masking safe zone
                map_of_scurge[y][x] = -1  # masking safe zone

        # Safe zone : left-top (896, 1888) right-bottom (1056, 2048) with additional (marginal) space
        for x in range(int(0/32), int(2048/32)):
            for y in range(int(0/32), int(64/32)):
                # map_of_scurge[x][y] = -1  # masking safe zone
                map_of_scurge[y][x] = -1  # masking safe zone

        # map_of_scurge[me_x_t][me_y_t] = 1
        map_of_scurge[me_y_t][me_x_t] = 1

        map_of_scurge = np.expand_dims(map_of_scurge, -1)

        for ob in observation.en_unit:
            # if 1880 <= ob.pos_y and 880 <= ob.pos_x <= 1050:
            #     pass
            # else:
            en_x_t = ob.pos_x / 32
            en_y_t = ob.pos_y / 32

            rel_x = int(en_x_t - me_x_t) + LOCAL_OBSERVABLE_TILE_SIZE
            rel_y = int(en_y_t - me_y_t) + LOCAL_OBSERVABLE_TILE_SIZE

            rel_x = np.clip(rel_x, 0, LOCAL_OBSERVABLE_TILE_SIZE*2-1)
            rel_y = np.clip(rel_y, 0, LOCAL_OBSERVABLE_TILE_SIZE*2-1)

            # map_of_observer[rel_x][rel_y] = map_of_observer[rel_x][rel_y] + 1  # if two or more observers are duplicated, we use sum
            map_of_observer[rel_y][rel_x] = map_of_observer[rel_y][rel_x] + 1  # if two or more observers are duplicated, we use sum

        # display out of map where scurge can't go based on current location of scurge
        scurge_out_of_map_left = me_x_t - LOCAL_OBSERVABLE_TILE_SIZE
        scurge_out_of_map_right = me_x_t + LOCAL_OBSERVABLE_TILE_SIZE
        scurge_out_of_map_up = me_y_t - LOCAL_OBSERVABLE_TILE_SIZE
        scurge_out_of_map_down = me_y_t + LOCAL_OBSERVABLE_TILE_SIZE

        if scurge_out_of_map_left < 0:
            map_of_observer[:, 0:-scurge_out_of_map_left] = -1
        if scurge_out_of_map_right > 64:
            map_of_observer[:, -(scurge_out_of_map_right-64):] = -1
        if scurge_out_of_map_up < 0:
            map_of_observer[0:-scurge_out_of_map_up,:] = -1
        if scurge_out_of_map_down > 64:
            map_of_observer[-(scurge_out_of_map_down-64):,:] = -1

        map_of_observer = np.expand_dims(map_of_observer, -1)

        if self.last_action is None:
            self.last_action = [-1]

        return [map_of_scurge, map_of_observer, self.last_action]


    def process_state_batch(self, batch):
            """
            BATCH_SIZE + TIME_WINDOW + OBSERVATION SPACE1 #1 (12,12)
            + OBSERVATION SPACE #2 (64,64) + OBSERVATION SPACE #3 (ACTIONS DURING TIME_WINDOWS,)

            :param batch:
            :return:
            """

            # here, skip assert in case that there could be another case that batch size = 1 for forward..
            # assert batch.shape[0] == BATCH_SIZE
            assert batch.shape[1] == TIME_WINDOW

            # observation needs to be concatenated all previous frame into one stacked frame.
            # but if time window is 1, it just squeeze data instead.
            # batch = np.squeeze(batch, axis=1)

            observation_1 = []
            observation_2 = []
            # observation_3 = []   # batch

            for i, sample in enumerate(batch):
                observation_sample_1 = []
                observation_sample_2 = []
                observation_sample_3 = []   # time window
                for t in range(TIME_WINDOW):
                    observation_sample_1.append(batch[i][t][0])  # sculge observation
                    observation_sample_2.append(batch[i][t][1])  # observers observation
                    # observation_sample_3.append([-1] if batch[i][t][2] is None else batch[i][t][2])  # last actions
                    # observation_sample_3.append( [0]*ACTION_SIZE if batch[i][t][2] is None else np.eye(ACTION_SIZE)[int(batch[i][t][2][0])] )  # last actions

                observation_1.append(observation_sample_1)
                observation_2.append(observation_sample_2)
                # observation_3.append(observation_sample_3)

            observation_1 = np.asarray(observation_1)
            observation_2 = np.asarray(observation_2)
            # observation_3 = np.asarray(observation_3)

            # if len(observation_3.shape) == 2:
            #     assert observation_3.shape == (TIME_WINDOW, 1)
            #     observation_3 = np.expand_dims(observation_3, axis=0)

            # if TIME_WINDOW == 1:
            #     observation_1 = np.squeeze(observation_1, axis=1)
            #     observation_2 = np.squeeze(observation_2, axis=1)
            #     observation_3 = np.squeeze(observation_3, axis=1)

            # return [observation_1, observation_2, observation_3]

            return [observation_1, observation_2]


    def process_action(self, action):
        act = []
        actions = []
        act.append(4)  # radiuqs tile position)
        act.append(action)  # angle between 0 and 1
        act.append(0)   # move(0) attack(1)
        act[1] = np.clip(act[1], 0, 1)
        actions.append(act)

        self.last_action = act[1]

        return actions

memory = SequentialMemory(limit=dict_args[OPS.REPLAY_MEMORY_SIZE()], window_length=TIME_WINDOW)
actor = build_actor_model()
critic = build_critic_model()

policy = NoisePolicy(
    random_process=SimpleOUNoise(size=1, theta=dict_args[OPS.OU_THETA()], mu=0., sigma=dict_args[OPS.OU_SIGMA()]))
test_policy = NoisePolicy(
        random_process=SimpleOUNoise(size=1, theta=.15, mu=0., sigma=.1))

# 0.1 for actor, 0.5 for critic
agent = DDPGAgent(actor, critic, ACTION_SIZE, memory, critic_action_input=action_input,
train_interval=dict_args[OPS.TARGET_NETWORK_UPDATE()],
                  processor=ObsProcessor(), batch_size=BATCH_SIZE, tau_for_actor=1e-3, tau_for_critic=1e-2, policy=policy, test_policy=test_policy)

# [optimizer for critic, optimizer for actor]
agent.compile(optimizer=[
    Adam(lr=dict_args[OPS.LEARNING_CRITIC_RATE()], clipvalue=0.5),
    Adam(lr=dict_args[OPS.LEARNING_ACTOR_RATE()], clipvalue=0.5)]
)


cb_plot = DrawTrainMovingAvgPlotCallback('../../save_graph/' + FILE_NAME_FOR_LOG + '.png', PLOT_EPISODE_INTERVAL, MOVING_AVERAGE_WINDOW, l_label=['episode_reward'])

agent.run(env, dict_args[OPS.N_STEPS()], train_mode=TRAINING_MODE, verbose=2, callbacks=[cb_plot], action_repetition=dict_args[OPS.ACTION_REPETITION()])

agent.save_weights('../../save_model/' + CURRENT_FILE_PATH, CURRENT_FILE_NAME, yyyymmdd, True)

env.close()
