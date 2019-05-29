# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from __future__ import division
from __future__ import print_function
import warnings
import timeit
import json
from tempfile import mkdtemp

import numpy as np

from keras import __version__ as KERAS_VERSION
from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList
from keras.utils.generic_utils import Progbar

class Callback(KerasCallback):
    def __init__(self, agent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = agent

    def _set_env(self, env):
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        """Called at beginning of each episode"""
        pass

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        pass

    def on_step_begin(self, step, logs={}):
        """Called at beginning of each step"""
        pass

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        pass

    def on_action_begin(self, action, logs={}):
        """Called at beginning of each action"""
        pass

    def on_action_end(self, action, logs={}):
        """Called at end of each action"""
        pass


class CallbackList(KerasCallbackList):
    def _set_env(self, env):
        """ Set environment for each callback in callbackList """
        for callback in self.callbacks:
            if callable(getattr(callback, '_set_env', None)):
                callback._set_env(env)

    def on_episode_begin(self, episode, logs={}):
        """ Called at beginning of each episode for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_episode_begin` callback.
            # If not, fall back to `on_epoch_begin` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_episode_begin', None)):
                callback.on_episode_begin(episode, logs=logs)
            else:
                callback.on_epoch_begin(episode, logs=logs)

    def on_episode_end(self, episode, logs={}):
        """ Called at end of each episode for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_episode_end` callback.
            # If not, fall back to `on_epoch_end` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_episode_end', None)):
                callback.on_episode_end(episode, logs=logs)
            else:
                from keras.callbacks import TensorBoard
                if type(callback) is TensorBoard:
                    logs.pop('info')
                    callback.on_epoch_end(episode, logs=logs)
                else:
                    callback.on_epoch_end(episode, logs=logs)

    def on_step_begin(self, step, logs={}):
        """ Called at beginning of each step for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_step_begin` callback.
            # If not, fall back to `on_batch_begin` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_step_begin', None)):
                callback.on_step_begin(step, logs=logs)
            else:
                callback.on_batch_begin(step, logs=logs)

    def on_step_end(self, step, logs={}):
        """ Called at end of each step for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_step_end` callback.
            # If not, fall back to `on_batch_end` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_step_end', None)):
                callback.on_step_end(step, logs=logs)
            else:
                from tensorflow.python.keras.callbacks import TensorBoard
                if type(callback) is TensorBoard:
                    scalar_logs = {}
                    for k, v in logs.items():
                        if type(v) in [np.int32, np.int16, np.int64, np.float32, np.float64]:
                            scalar_logs[k] = v
                    callback.on_batch_end(step, logs=scalar_logs)
                else:
                    callback.on_batch_end(step, logs=logs)

    def on_action_begin(self, action, logs={}):
        """ Called at beginning of each action for each callback in callbackList"""
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_begin', None)):
                callback.on_action_begin(action, logs=logs)

    def on_action_end(self, action, logs={}):
        """ Called at end of each action for each callback in callbackList"""
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_end', None)):
                callback.on_action_end(action, logs=logs)
