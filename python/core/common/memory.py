# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random
import numpy as np

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)

    def __len__(self):
        return self.length()

    def __getitem__(self, idx):
        """Return element of buffer at specific index

        # Argument
            idx (int): Index wanted

        # Returns
            The element of buffer at given index
        """
        if idx < 0 or idx >= self.length():
            raise KeyError()
        return self.data[idx]

    def append(self, v):
        """Append an element to the buffer

        # Argument
            v (object): Element to append
        """
        self.data.append(v)

    def length(self):
        """Return the length of Deque

        # Argument
            None

        # Returns
            The lenght of deque element
        """
        return len(self.data)


class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        """Return list of last observations

        # Argument
            current_observation (object): Last observation

        # Returns
            A list of the last observations
        """
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        """Return configuration (window_length, ignore_episode_boundaries) for Memory

        # Return
            A dict with keys window_length and ignore_episode_boundaries
        """
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config

def zeroed_observation(observation):
    """Return an array of zeros with same shape as given observation

    # Argument
        observation (list): List of observation

    # Return
        A np.ndarray of zeros with observation.shape
    """
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.

def sample_batch_indexes(low, high, size):
    """Return a sample of (size) unique elements between low and high

        # Argument
            low (int): The minimum value for our samples
            high (int): The maximum value for our samples
            size (int): The number of samples to pick

        # Returns
            A list of samples of length size, with values between low and high
        """
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = range(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs
