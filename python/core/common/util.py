# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import subprocess
import itertools
import os
import pickle
import logging
import keras.backend as K
from keras.models import model_from_config, Sequential, Model, model_from_config
import keras.optimizers as optimizers
from keras.layers import Layer
from enum import Enum
import tensorflow as tf
import time
import os
from typing import Tuple


class PopArtLayer(Layer):
    """
    Automatic network output scale adjuster, which is supposed to keep
    the output of the network up to date as we keep updating moving
    average and variance of discounted returns.
    Part of the PopArt algorithm described in DeepMind's paper
    "Learning values across many orders of magnitude"
    (https://arxiv.org/abs/1602.07714)
    """
    def __init__(self, beta=1e-4, epsilon=1e-4, stable_rate=0.1,
                 min_steps=1000, **kwargs):
        """
        :param beta: a value in range (0..1) controlling sensitivity to changes
        :param epsilon: a minimal possible value replacing standard deviation
                if the original one is zero.
        :param stable_rate: Pop-part of the algorithm will kick in only when
            the amplitude of changes in standard deviation will drop
            to this value (stabilizes). This protects pop-adjustments from
            being activated too soon, which would lead to weird values
            of `W` and `b` and numerical instability.
        :param min_steps: Minimal number of steps before it even begins
            possible for Pop-part to become activated (an extra precaution
            in addition to the `stable_rate`).
        :param kwargs: any extra Keras layer parameters, like name, etc.
        """
        self.beta = beta
        self.epsilon = epsilon
        self.stable_rate = stable_rate
        self.min_steps = min_steps
        super().__init__(**kwargs)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel', shape=(), dtype='float32',
            initializer='ones', trainable=False)
        self.bias = self.add_weight(
            name='bias', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.mean = self.add_weight(
            name='mean', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.mean_of_square = self.add_weight(
            name='mean_of_square', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.step = self.add_weight(
            name='step', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.pop_is_active = self.add_weight(
            name='pop_is_active', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.kernel * inputs + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

    def de_normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Converts previously normalized data into original values.
        """
        online_mean, online_mean_of_square = K.batch_get_value(
            [self.mean, self.mean_of_square])
        std_dev = np.sqrt(online_mean_of_square - np.square(online_mean))
        return (x * (std_dev if std_dev > 0 else self.epsilon)
                + online_mean)

    def pop_art_update(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Performs ART (Adaptively Rescaling Targets) update,
        adjusting normalization parameters with respect to new targets x.
        Updates running mean, mean of squares and returns
        new mean and standard deviation for later use.
        """
        assert len(x.shape) == 2, "Must be 2D (batch_size, time_steps)"
        beta = self.beta
        (old_kernel, old_bias, old_online_mean,
         old_online_mean_of_square, step, pop_is_active) = K.batch_get_value(
            [self.kernel, self.bias, self.mean,
             self.mean_of_square, self.step, self.pop_is_active])

        def update_rule(old, new):
            """
            Update rule for running estimations,
            dynamically adjusting sensitivity with every time step
            to new data (see Eq. 10 in the paper).
            """
            nonlocal step
            step += 1
            adj_beta = beta / (1 - (1 - beta)**step)
            return (1 - adj_beta) * old + adj_beta * new

        x_means = np.stack([x.mean(axis=0), np.square(x).mean(axis=0)], axis=1)
        # Updating normalization parameters (for ART)
        # import functools ?
        online_mean, online_mean_of_square = reduce(
            update_rule, x_means,
            np.array([old_online_mean, old_online_mean_of_square]))
        old_std_dev = np.sqrt(
            old_online_mean_of_square - np.square(old_online_mean))
        std_dev = np.sqrt(online_mean_of_square - np.square(online_mean))
        old_std_dev = old_std_dev if old_std_dev > 0 else std_dev
        # Performing POP (Preserve the Output Precisely) update
        # but only if we are not in the beginning of the training
        # when both mean and std_dev are close to zero or still
        # stabilizing. Otherwise POP kernel (W) and bias (b) can
        # become very large and cause numerical instability.
        std_is_stable = (
            step > self.min_steps
            and np.abs(1 - old_std_dev / std_dev) < self.stable_rate)
        if (int(pop_is_active) == 1 or
                (std_dev > self.epsilon and std_is_stable)):
            new_kernel = old_std_dev * old_kernel / std_dev
            new_bias = (
                (old_std_dev * old_bias + old_online_mean - online_mean)
                / std_dev)
            pop_is_active = 1
        else:
            new_kernel, new_bias = old_kernel, old_bias
        # Saving updated parameters into graph variables
        var_update = [
            (self.kernel, new_kernel),
            (self.bias, new_bias),
            (self.mean, online_mean),
            (self.mean_of_square, online_mean_of_square),
            (self.step, step),
            (self.pop_is_active, pop_is_active)]
        K.batch_set_value(var_update)
        return online_mean, std_dev

    def update_and_normalize(self, x: np.ndarray) -> Tuple[np.ndarray,
                                                           float, float]:
        """
        Normalizes given tensor `x` and updates parameters associated
        with PopArt: running means (art) and network's output scaling (pop).
        """
        mean, std_dev = self.pop_art_update(x)
        result = ((x - mean) / (std_dev if std_dev > 0 else self.epsilon))
        return result, mean, std_dev



def smoothL1(y_true, y_pred):
    """
    https://stackoverflow.com/questions/44130871/keras-smooth-l1-loss
    """
    HUBER_DELTA = 0.5
    x   = K.abs(y_true - y_pred)
    x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)


def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)


def gumbel_softmax(logits, temperature=1, hard=False):
  """
  referenced by # https://github.com/ericjang/gumbel-softmax

  Sample from the Gumbel-Softmax distribution and optionally discretize.

  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y


def clone_optimizer(optimizer):
    if type(optimizer) is str:
        return optimizers.get(optimizer)
    # Requires Keras 1.0.7 since get_config has breaking changes.
    params = dict([(k, v) for k, v in optimizer.get_config().items()])
    config = {
        'class_name': optimizer.__class__.__name__,
        'config': params,
    }
    if hasattr(optimizers, 'optimizer_from_config'):
        # COMPATIBILITY: Keras < 2.0
        clone = optimizers.optimizer_from_config(config)
    else:
        clone = optimizers.deserialize(config)
    return clone


def gradients(loss, variables, grad_ys):
    """Returns the gradients of `loss` w.r.t. `variables`.
    # Arguments
        loss: Scalar tensor to minimize.
        variables: List of variables.
    # Returns
        A gradients tensor.
    """
    return tf.gradients(loss, variables, grad_ys, colocate_gradients_with_ops=False)


class cLogger():
    global LOGGER

    @staticmethod
    def getLogger(loggerName='not_init', loggerFile=None):
        LOGGER = logging.getLogger('Main')
        if loggerName is 'init':

            fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
            if loggerFile is not None:
                fileHandler = logging.FileHandler('./myLoggerTest.log', 'w', 'utf-8')
                fileHandler.setFormatter(fomatter)
                LOGGER.addHandler(fileHandler)

            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(fomatter)
            LOGGER.addHandler(streamHandler)
            LOGGER.setLevel(logging.DEBUG)
            logging.basicConfig(filename='./test.log',level=logging.DEBUG)
        else :
            pass
        return LOGGER


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, my, epsilon=1e-2, shape=()):

        self._sum = K.constant(value=np.zeros(shape), dtype=tf.float64, shape=shape, name=my+"_runningsum")
        self._sumsq = K.constant(value=np.zeros(shape) + epsilon, dtype=tf.float64, shape=shape, name=my+"_runningsumsq")
        self._count = K.constant(value=np.zeros(()) + epsilon, dtype=tf.float64, shape=(), name=my+"_count")

        self.mean = K.cast_to_floatx(self._sum / self._count)
        self.std = K.sqrt(K.maximum(K.cast_to_floatx(self._sumsq / self._count) - K.square(self.mean), epsilon))

        newsum = Input(shape=shape, dtype=tf.float64, name=my+'_sum')
        newsumsq = K.variable(shape=shape, dtype=tf.float64, name=my+'_var')
        newcount = K.variable(shape=[], dtype=K.float64, name=my+'_count')

        self.incfiltparams = K.function([newsum, newsumsq, newcount], [],
            updates=[tf.assign_add(self._sum, newsum),
                     tf.assign_add(self._sumsq, newsumsq),
                     tf.assign_add(self._count, newcount)])

    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n*2+1, 'float64')
        addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)],dtype='float64')])
        self.incfiltparams(totalvec[0:n].reshape(self.shape),
                           totalvec[n:2*n].reshape(self.shape),
                           totalvec[2*n])


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


class AdditionalUpdatesOptimizer(optimizers.Optimizer):
    def __init__(self, optimizer, additional_updates):
        super(AdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_updates(self, params, loss):
        updates = self.optimizer.get_updates(params=params, loss=loss)
        updates += self.additional_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()


def clipped_masked_error(y_true, y_pred, mask, delta_clip):
    loss = huber_loss(y_true, y_pred, delta_clip)
    loss *= mask  # apply element-wise mask
    return K.sum(loss, axis=-1)


def clone_model(model, custom_objects={}):
    # todo : change simpler logic
    """
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    """
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone


def huber_loss(y_true, y_pred, clip_value):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))


def get_soft_target_model_updates(target, source, tau):
    # todo : debug her why it does add sum of non-trainable_weights to target weights?
    target_weights = target.trainable_weights + sum([l.non_trainable_weights for l in target.layers], [])
    source_weights = source.trainable_weights + sum([l.non_trainable_weights for l in source.layers], [])
    assert len(target_weights) == len(source_weights)

    # Create updates.
    updates = []
    for tw, sw in zip(target_weights, source_weights):
        # todo : why does it add just tuple of weights other than K.update(old, new) shape?
        updates.append((tw, tau * sw + (1. - tau) * tw))
    return updates

def mmdd24hhmmss():
    return datetime.now().strftime('%m%d%H%M%S')


def yyyymmdd24hhmmss():
    return datetime.now().strftime('%Y%m%d%H%M%S')


def display_param_list(params=[]):
    return '_'.join(params)


def display_param_dic(params={}):
    lc = []
    for key, value in zip(params.keys(), params.values()):
        lc.append(str(key) + "_" + str(value))
    return '_'.join(lc)

# def reward_moving_avg_plot(rewards=[], window=100, filepath='plot.png', shadow_color_index=5):
#     plt.figure(figsize=(12, 8))
#     plt.ylim([-100, 500])
#     plt.plot(episodes, scores, 'b', label='score', linestyle=':')
#     plt.plot(episodes, last_y_list, 'g', label='y', linestyle='-')
#     plt.legend(['score', 'y'], loc='upper right')
#     plt.savefig("./save_graph/" + FILE_NAME + ".png")
#     plt.close()


def reward_quantile_plot(y_data_list=[], title = None, label='reward', window=100, filepath='plot.png', shadow_color_index=5):
    """
    :param y_data_list: reward list
    :param title: optional, title
    :param label: y axis label
    :param window: moving average time window
    :param filepath: plot file to be saved
    :param shadow_color_index:
        :param shadow_color_index:
    :return:
    """
    df = pd.DataFrame()
    df['episode'] = pd.Series(np.arange(len(y_data_list)))
    df['reward'] = pd.Series(y_data_list)
    df['reward_mean'] = df['reward'].rolling(window=window).mean()
    df['reward_quantile_1'] = df['reward'].rolling(window=window).quantile(0.25, interpolation='midpoint')
    df['reward_quantile_2'] = df['reward'].rolling(window=window).quantile(0.75, interpolation='midpoint')

    plt.figure(figsize=(12, 8))
    palette = sns.color_palette()

    axes = df.loc[:, 'reward_mean'].plot(label='reward')
    axes.fill_between(df.loc[:, 'episode'], df.loc[:, 'reward_mean'], df.loc[:, 'reward_quantile_2'], alpha=0.1,
                      color=palette[shadow_color_index])
    axes.fill_between(df.loc[:, 'episode'], df.loc[:, 'reward_quantile_1'], df.loc[:, 'reward_mean'], alpha=0.1,
                      color=palette[shadow_color_index])
    # plt.legend(['reward'], loc='upper right')

    if title is not None:
        plt.title(str(title) + ' curve', fontsize=25)
    plt.xlabel('episode', fontsize=20)
    plt.ylabel(label+'(window=%d)'%(window), fontsize=20)
    plt.savefig(filepath)
    plt.close()


def reward_moving_avg_plot(y_data_list=[], title='', label='reward', window=100, filepath='plot.png', shadow_color_index=5):
    """
    save reward plot
    :param rewards: reward list
    :param window: moving average time window
    :param filepath: plot file to be saved
    :param shadow_color_index: color palette index number
    :return: None
    """
    df = pd.DataFrame()
    df['episode'] = pd.Series(np.arange(len(y_data_list)))
    df['reward'] = pd.Series(y_data_list)
    df['reward_mean'] = df['reward'].rolling(window=window).mean()
    df['reward_min'] = df['reward'].rolling(window=window).min()
    df['reward_max'] = df['reward'].rolling(window=window).max()

    plt.figure(figsize=(12, 8))
    palette = sns.color_palette()

    axes = df.loc[:, 'reward_mean'].plot(label='reward')
    axes.fill_between(df.loc[:, 'episode'], df.loc[:, 'reward_mean'], df.loc[:, 'reward_max'], alpha=0.1,
                      color=palette[shadow_color_index])
    axes.fill_between(df.loc[:, 'episode'], df.loc[:, 'reward_min'], df.loc[:, 'reward_mean'], alpha=0.1,
                      color=palette[shadow_color_index])
    # plt.legend(['reward'], loc='upper right')
    plt.xlabel('episode', fontsize=20)
    plt.ylabel(label+'(window=%d)'%(window))
    plt.title(title)

    plt.savefig(filepath)
    plt.close()

def save_ci_graph_from_tuple(y_data_list=[], graph_title=[], window=1000, filepath='2plot.png', y_data_legend=[], y_index=[], figsize=(12, 8)
                             , title_font_size=20):

    dict_data = {}
    data_size = 0

    print_all = True if len(y_index) == 0 else False

    for idx, list_data in enumerate(y_data_list):
        for idx2, tuple_data in enumerate(list_data):
            if print_all or idx2 in y_index:
                if type(tuple_data) == tuple:
                    if dict_data.get(tuple_data[0]) is None:
                        dict_data[tuple_data[0]] = []
                    if data_size == 0:
                        data_size = len(tuple_data[1])
                    dict_data[tuple_data[0]].append(tuple_data[1])

    df = pd.DataFrame()
    episode_window_unit = []
    for i in range(data_size):
        if int(i / window) == int(data_size / window) - 1:
            episode_window_unit.append(int(i / window) * window)
        else:
            episode_window_unit.append(int(i / window) * window + 1)

    df['episode_window'] = pd.Series(episode_window_unit)

    for idx, tuple_str in enumerate(dict_data): #hps, episodes
        sns.set(style="darkgrid")
        plt.figure(figsize=figsize)

        for i, values in enumerate(dict_data[tuple_str]):
            title_ = y_data_legend[i] if i < len(y_data_legend) else i
            df[title_] = pd.Series(values)
            sns.lineplot(x="episode_window", y=title_, ci=95, data=df, label=title_)

        plt.xlabel("episode", fontsize=20)
        plt.ylabel("")
        real_graph_title = tuple_str if len(graph_title) == 0 else graph_title[idx]
        plt.title(real_graph_title, fontdict={'fontsize': title_font_size})
        plt.savefig(filepath+"_"+real_graph_title+"_ci.png")
        plt.close()
        
def save_ci_graph(y_data_list=[], title='Some Graph', xlabel='episode', ylabel='reward', window=100, filepath='plot.png', y_data=[], y_index=[], figsize=(12, 8)
                  , title_font_size=20):
    """
    save reward plot
    :param window: moving average time window
    :param filepath: plot file to be saved
    :return: None
    """
    df = pd.DataFrame()
    data_size = len(y_data_list[0])

    episode_window_unit = []
    for i in range(data_size):
        if int(i / window) == int(data_size / window) - 1:
            episode_window_unit.append(int(i / window) * window)
        else:
            episode_window_unit.append(int(i / window) * window + 1)

    df['episode_window'] = pd.Series(episode_window_unit)
    sns.set(style="darkgrid")
    plt.figure(figsize=figsize)

    for idx, values in enumerate(y_data_list):
        if len(y_index) == 0 or idx in y_index:
            title_ = y_data[idx] if len(y_data) - 1 >= idx else idx
            df[str(title_)] = pd.Series(values)
            sns.lineplot(x="episode_window", y=str(title_), ci=95, data=df, label=str(title_))

    # ax = sns.lineplot(x="episode_window", y='test1', ci=95, data=df, label='test1')
    # ax2= sns.lineplot(x="episode_window", y='test2', ci=95, data=df, label='test2')
    # ax2.legend(loc="upper right")

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel)
    plt.title(title,fontdict={'fontsize': title_font_size})
    plt.savefig(filepath+"_ci.png")
    plt.close()


def pickle_to_plot(pickle_name, png_filename, title='', time_window=100, overwrite=False):
    # pickle to graph
    with open(pickle_name, 'rb') as data:
        if len(title) > 120:
            title = title[:120] + '\n' + title[120:]

        plot_data_list = pickle.load(data)

        for idx, plot_data in enumerate(plot_data_list):
            if type(plot_data) is tuple:
                label = plot_data[0]
                y_data = plot_data[1]
            elif type(plot_data) is list:
                label = str(idx)
                y_data = plot_data

            save_file_name = png_filename.format(label)

            if not overwrite and os.path.isfile(save_file_name):
                print(save_file_name + ' file exists.')
                continue

            reward_moving_avg_plot(title=title, y_data_list=y_data, window=time_window, label=label,
                                   filepath=save_file_name)


def gen_agent_params(params={}, filepath=None):
    if type({}) == dict or filepath is None:
        print('this param isn\'t dict type')
    else:
        f = open(filepath, 'w')
        f.write(params)


def get_kv_from_agent(agent):
    kv = {}
    # members = [attr for attr in dir(agent) if not callable(getattr(agent, attr)) and not attr.startswith("__")]
    kv['state_size'] = agent.state_size
    kv['action_size'] = agent.action_size
    kv['build_model'] = agent.build_model
    kv['discount_factor'] = agent.discount_factor
    kv['learning_rate'] = agent.learning_rate
    kv['epsilon_start'] = agent.epsilon_start
    kv['epsilon_decay'] = agent.epsilon_decay
    kv['exploration_steps'] = agent.exploration_steps
    kv['epsilon_min'] = agent.epsilon_min
    kv['batch_size'] = agent.batch_size
    kv['train_start'] = agent.train_start
    kv['max_queue_size'] = agent.max_queue_size
    kv['max_tmp_queue_size'] = agent.max_tmp_queue_size
    return kv


def save_plot(FILE_NAME):
    with open('../save_graph/' + FILE_NAME +'_data.pickle', 'rb') as f:
        p = pickle.load(f)
        [scores, steps, hps, kill_cnts, cont_win_cnts, max_cont_wincnt] = p

        reward_quantile_plot(y_data_list=steps, window=100, label='steps',
                          filepath="../save_graph/" + FILE_NAME + "_steps_quantile3.png", title=FILE_NAME)
        reward_quantile_plot(y_data_list=scores, window=100, label='rewards',
                          filepath="../save_graph/" + FILE_NAME + "_scores_quantile3.png", title=FILE_NAME)
        reward_quantile_plot(y_data_list=hps,window=100, label="HP",
                             filepath="../save_graph/" + FILE_NAME + "_hps_quantile3.png", title=FILE_NAME)


def get_logger(logger_name, logger_level, use_stream_handler=True, use_file_handler=False):
    mylogger = logging.getLogger(logger_name)
    mylogger.setLevel(logger_level)

    if use_stream_handler:
        stream_hander = logging.StreamHandler()
        mylogger.addHandler(stream_hander)

    if use_file_handler:
        file_handler = logging.FileHandler(logger_name + '.log')
        mylogger.addHandler(file_handler)

    return mylogger


def auto_executor(params, filename):
    '''

    사용법
    1. 사용하고 싶은 옵션을 OPS enum 클래스에 추가한다.
    2. params에 enum 의 값을 배열로 입력한다. ex : param[OPS.추가한옵션.value] = [실행할 변수 list]
    3. 수행할 filename 을 입력한다.

    4. 호출되는 소스에서 arg 처리를 해준다.

    import argparse
    from core.common.util import OPS

    parser = argparse.ArgumentParser(description='DQN Configuration including setting dqn / double dqn / double dueling dqn')

    parser.add_argument(OPS.NO_GUI.value, help='gui', type=bool, default=False)
    parser.add_argument(OPS.DOUBLE.value, help='double dqn', default=False, action='store_true')
    parser.add_argument(OPS.DUELING.value, help='dueling dqn', default=False, action='store_true')
    parser.add_argument(OPS.DRQN.value, help="drqn", default=False, action='store_true')
    parser.add_argument(OPS.BATCH_SIZE.value, type=int, default=128, help="batch size")
    parser.add_argument(OPS.REPLAY_MEMORY_SIZE.value, type=int, default=8000, help="replay memory size")
    parser.add_argument(OPS.LEARNING_RATE.value, type=float, default=0.001, help="learning rate")
    parser.add_argument(OPS.TARGET_NETWORK_UPDATE.value, type=int, default=60, help="target_network_update_interval")
    .
    .
    .
    parser.add_argument("추가한 옵션", type=타입, default=기본값, help="help 에 표시될 도움말" ...)

    args = parser.parse_args()

    dict_args = vars(args)
    post_fix = ''
    for k in dict_args.keys():
        if k == 'no_gui':
            continue
        post_fix += '_' + k + '_' + str(dict_args[k])


    5. args 를 적절히 소스에 사용해준다.
    6. output file naming 에 post_fix 를 추가해주면 좋음.


    '''
    # op = lambda k, l: np.concatenate([[k, v] if v is not None else [k] for v in l])
    op = lambda k, l: [[k, v] if v is not None else [k] for v in l]

    # 값이 배열로 주어지는 파라메터
    list_params = {}
    # T/F 로 주어지는 파라메터
    on_off_params = {}

    TOTAL_COUNT = 1

    while len(params):
        p = params.popitem()

        if p[1] == [None]:
            on_off_params[p[0]] = p[1]
            TOTAL_COUNT *= 2
        else:
            list_params[p[0]] = p[1]
            TOTAL_COUNT *= len(p[1])

    keys = on_off_params.keys()

    # on_off_params 이 모두 선택 안된 경우 실행
    ll = []

    for k in list_params.keys():
        ll.append(op(k, list_params[k]))

    now = 1

    for combination in itertools.product(*ll):
        if len(combination) != 0:
            print(np.concatenate(combination).tolist(), str(now) + '/' + str(TOTAL_COUNT))
            now += 1
            subargs = [sys.executable, filename] + np.concatenate(combination).tolist()
            subprocess.call(subargs, shell=True)

    # on_off_params 가 하나 이상 선택된 경우 실행
    for s in range(len(on_off_params)):
        for k_l in itertools.combinations(keys, s + 1):
            ll = []
            for k in k_l:
                ll.append(op(k, on_off_params[k]))

            for k in list_params.keys():
                ll.append(op(k, list_params[k]))

            for combination in itertools.product(*ll):
                if len(combination) != 0:
                    print(np.concatenate(combination).tolist(), str(now) + '/' + str(TOTAL_COUNT))
                    now += 1
                    subargs = [sys.executable, filename] + np.concatenate(combination).tolist()
                    subprocess.call(subargs, shell=True)


class OPS(Enum):

    # warn : please make value as short as possible. because if those values are added, it could exceed 255 length. (windows file max length : 255)

    # For environment
    NO_GUI = '-no-gui'
    FRAMES_PER_STEP = '-frms-per-step'
    MOVE_ANG = '-move_a'
    MOVE_DIST = '-move_d'

    # Common for agent
    N_STEPS = '-nsteps'
    ACTION_REPETITION = '-act-rept'
    REWARD_SCALE = '-rc'
    REWARD_VERSION = '-rv'

    # FOR DQN and its variants
    DOUBLE = '-double'
    DUELING = '-dueling'
    BATCH_SIZE = '-batch-size'
    TARGET_NETWORK_UPDATE = '-tn-u'

    REPLAY_MEMORY_SIZE = '-repm-size'
    LEARNING_RATE = '-lr'
    DISCOUNT_FACTOR = '-df'

    WINDOW_LENGTH = '-wl'

    # For Actor Critic Architecture
    LEARNING_CRITIC_RATE = '-critic-lr'
    LEARNING_ACTOR_RATE = '-actor-lr'

    # DDPG Exploration
    OU_SIGMA = '-ou-sm'
    OU_THETA = '-ou-tt'
    USE_PARAMETERIZED_NOISE = '-p-noise'
    PURE_ACTION_RATIO = '-ratio-pure-action'

    # LSTM Properties
    TIME_WINDOW = '-t-w'

    # Specific for AvoidObserver scenario
    MARGINAL_SPACE = '-m-s'
    REWARD_HEIGHT_RANK_WEIGHT = '-r-w'
    TIME_PENALTY_WEIGHT = '-p-t'

    # For PPO
    EPOCHS = '-epo'
    GAMMA = '-gamma'
    BUFFER_SIZE = '-buf_size'
    ENTROPY_LOSS = '-ent_loss'

    POLICY = '-p'

    PER = '-PER'
    def __call__(self):
        return self._value_[1:].replace('-', '_')


if __name__ == "__main__":
    FILE_NAME = ['vulture_vs_zealot2_DDDQN_20190222180422', 'vulture_vs_zealot2_DDDQN_stacked_20190222205841',
                 'vulture_vs_zealot2_DRQN_20190225141512','vulture_vs_zealot_DQN_20190222163829']

    for f in FILE_NAME:
        save_plot(f)
