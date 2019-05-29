# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from core.common.memory import *
import operator
import copy

class SequentialMemory(Memory):
    def __init__(self, limit, enable_per=False, per_alpha=0.6, per_beta=0.4, **kwargs):
        """

        :param limit:
        :param enable_per(booelen): enable Prioritized Experience Replay
        :param per_alpha : how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        :param kwargs:
        """
        super(SequentialMemory, self).__init__(**kwargs)

        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)
        self._next_idx = 0
        self.enable_per = enable_per
        self.per_alpha = per_alpha
        self.per_beta = per_beta

        if self.enable_per:
            if per_alpha < 0 or per_alpha > 1:
                assert False, "per_alpha must be between 0 and 1"
            elif per_beta < 0 or per_beta > 1:
                assert False, "per_beta must be between 0 and 1"

        it_capacity = 1
        while it_capacity < limit:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def _sample_proportional(self, time_window, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self.observations) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            while idx < time_window:
                mass = random.random() * self._it_sum.sum(0, len(self.observations) - 1)
                idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.actions)
            self._it_sum[idx] = priority ** self.per_alpha
            self._it_min[idx] = priority ** self.per_alpha

            self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of experiences

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
            per_beta (float): Prioritized Experience Replay Memory Hyper parameter, To what degree to use importance weights(0 - no corrections, 1 - full correction)

        # Returns
            A list of experiences randomly selected
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            if self.enable_per:
                batch_idxs = self._sample_proportional(self.window_length, batch_size)

                weights = []
                p_min = self._it_min.min() / self._it_sum.sum()  # minimum priority
                max_weight = (p_min * len(self.terminals)) ** (-self.per_beta)

                for idx in batch_idxs:
                    p_sample = self._it_sum[idx] / self._it_sum.sum()
                    weight = (p_sample * len(self.terminals)) ** (-self.per_beta)
                    weights.append(weight / max_weight)
                weights = np.array(weights)
            else:
                # Draw random indexes such that we have enough entries before each index to fill the
                # desired window length.
                batch_idxs = sample_batch_indexes(
                    self.window_length, self.nb_entries - 1, size=batch_size)

        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [copy.deepcopy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size

        if self.enable_per:
            return experiences, weights, batch_idxs
        else:
            return experiences

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            if self.enable_per:
                idx = self._next_idx
                self._it_sum[idx] = self._max_priority ** self.per_alpha
                self._it_min[idx] = self._max_priority ** self.per_alpha

                if self._next_idx >= len(self.observations):
                    self.observations.append(observation)
                    self.actions.append(action)
                    self.rewards.append(reward)
                    self.terminals.append(terminal)
                else:
                    self.observations.data[self._next_idx] = observation
                    self.actions.data[self._next_idx] = action
                    self.rewards.data[self._next_idx] = reward
                    self.terminals.data[self._next_idx] = terminal
                self._next_idx = (self._next_idx + 1) % self.limit
            else :
                self.observations.append(observation)
                self.actions.append(action)
                self.rewards.append(reward)
                self.terminals.append(terminal)

    @property
    def nb_entries(self):
        """Return number of observations

        # Returns
            Number of observations
        """
        return len(self.observations)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""
        return super(MinSegmentTree, self).reduce(start, end)
