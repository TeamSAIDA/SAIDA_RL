# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from core.common.policy import *
import numpy as np
from collections import deque

# todo : add policy with annealing ou process based on "GEP-PG: Decoupling Exploration and Exploitation in \
#  Deep Reinforcement Learning Algorithms" https://arxiv.org/pdf/1802.05054.pdf
class NoisePolicy(Policy):
    """Implement policy based on OrnsteinUhlenbeck Process
    This policy returns action added by noise for exploration in ddpg
    """
    def __init__(self, random_process, ratio_of_pure_action=1.0):
        super(NoisePolicy, self).__init__()
        assert random_process is not None
        # raise ValueError('please set random process in policy declaration')
        self.random_process = random_process
        self.ratio_of_pure_action = ratio_of_pure_action

    def select_action(self, pure_action):
        """Return the selected action

        # Arguments
            random_process : Random process

        # Returns
            Selection action
        """
        noise = self.random_process.sample()
        # action_with_noise = np.clip(pure_action * self.ratio_of_pure_action + noise, 0, 1)
        action_with_noise = pure_action * self.ratio_of_pure_action + noise

        return action_with_noise, pure_action

    def reset_states(self):
        self.random_process.reset_states()


class GreedyQPolicy(Policy):
    """Implement the greedy policy

    Greedy policy returns the current best action according to q_values
    """
    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class EpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class LinearAnnealedPolicy(Policy):
    """Implement the linear annealing policy

    Linear Annealing Policy computes a current threshold value and
    transfers it to an inner policy which chooses the action. The threshold
    value is following a linear function decreasing over time."""
    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps):
        if not hasattr(inner_policy, attr):
            raise ValueError('Policy does not have attribute "{}".'.format(attr))

        super(LinearAnnealedPolicy, self).__init__()

        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps

    def get_current_value(self):
        """Return current annealing value

        # Returns
            Value to use in annealing
        """
        # if self.agent.training:
        # Linear annealed: f(x) = ax + b.
        a = -float(self.value_max - self.value_min) / float(self.nb_steps)
        b = float(self.value_max)
        value = max(self.value_min, a * float(self.agent.step) + b)
        # else:
        #     value = self.value_test
        return value

    def select_action(self, **kwargs):
        """Choose an action to perform

        # Returns
            Action to take (int)
        """
        setattr(self.inner_policy, self.attr, self.get_current_value())
        return self.inner_policy.select_action(**kwargs)

    @property
    def metrics_names(self):
        """Return names of metrics

        # Returns
            List of metric names
        """
        return ['mean_{}'.format(self.attr)]

    @property
    def metrics(self):
        """Return metrics values

        # Returns
            List of metric values
        """

        return [getattr(self.inner_policy, self.attr)]

    def get_config(self):
        """Return configurations of LinearAnnealedPolicy

        # Returns
            Dict of config
        """
        config = super(LinearAnnealedPolicy, self).get_config()
        config['attr'] = self.attr
        config['value_max'] = self.value_max
        config['value_min'] = self.value_min
        config['value_test'] = self.value_test
        config['nb_steps'] = self.nb_steps
        config['inner_policy'] = get_object_config(self.inner_policy)
        return config


class MA_EpsGreedyQPolicy(Policy):
    def __init__(self, eps=.1):
        super(MA_EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        assert q_values.ndim == 3
        q_values = np.squeeze(q_values, axis=0)
        nb_agents = q_values.shape[0]
        nb_actions = q_values.shape[1]

        actions = []
        for agent in range(nb_agents):
            if np.random.uniform() < self.eps:
                action = np.random.random_integers(0, nb_actions-1)
            else:
                action = np.argmax(q_values[agent])
            actions.append(action)
        actions = np.array(actions)
        return actions

    def get_config(self):
        config = super(MA_EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class MA_GreedyQPolicy(Policy):
    def select_action(self, q_values):
        assert q_values.ndim == 3
        q_values = np.squeeze(q_values, axis=0)
        actions = np.argmax(q_values, axis=-1)
        return actions


class MA_BoltzmannQPolicy(Policy):
    def __init__(self, tau=1., clip=(-500., 500.)):
        super(MA_BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        assert q_values.ndim == 3
        q_values = np.squeeze(q_values, axis=0)
        actions = np.apply_along_axis(self.select_action_agent, -1, q_values)
        return actions

    def select_action_agent(self, q_value):
        assert q_value.ndim == 1
        q_value = q_value.astype('float64')
        nb_actions = q_value.shape[0]

        exp_value = np.exp(np.clip(q_value / self.tau, self.clip[0], self.clip[1]))
        probs = exp_value / np.sum(exp_value)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        config = super(MA_BoltzmannQPolicy, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config


class MA_MaxBoltzmannQPolicy(Policy):
    """
    A combination of the eps-greedy and Boltzman q-policy.

    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amserdam, Amsterdam (1999)

    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    """
    def __init__(self, eps=.1, tau=1., clip=(-500., 500.)):
        super(MA_MaxBoltzmannQPolicy, self).__init__()
        self.eps = eps
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        assert q_values.ndim == 3
        q_values = np.squeeze(q_values, axis=0)
        actions = np.apply_along_axis(self.select_action_agent, -1, q_values)
        return actions

    def select_action_agent(self, q_value):
        assert q_value.ndim == 1
        q_value = q_value.astype('float64')
        nb_actions = q_value.shape[0]

        if np.random.uniform() < self.eps:
            exp_value = np.exp(np.clip(q_value / self.tau, self.clip[0], self.clip[1]))
            probs = exp_value / np.sum(exp_value)
            action = np.random.choice(range(nb_actions), p=probs)
        else:
            action = np.argmax(q_value)
        return action

    def get_config(self):
        config = super(MA_MaxBoltzmannQPolicy, self).get_config()
        config['eps'] = self.eps
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config


class AdvEpsGreedyPolicy(LinearAnnealedPolicy):
    """Implement the AdvEpsGreedyPolicy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)

    epsilon is calculated by:
    - max(epsilon greedy value, score based value)
    """

    def __init__(self, max_score, min_score=0, score_queue_size=100, score_name='episode_reward', score_type='mean', str_eps=1
                 , nb_agents=1, **kwargs):
        if nb_agents == 1:
            super(AdvEpsGreedyPolicy, self).__init__(inner_policy=EpsGreedyQPolicy(), attr='eps', **kwargs)
        else:
            super(AdvEpsGreedyPolicy, self).__init__(inner_policy=MA_EpsGreedyQPolicy(), attr='eps', **kwargs)
        self.max_score = max_score - min_score
        self.min_score = min_score
        self.score_queue = deque(maxlen=score_queue_size)
        self.score_name = score_name
        self.score_type = score_type
        self.str_eps = str_eps

    def on_episode_end(self, episode, logs={}):
        logs['eps'] = self.inner_policy.eps

        if logs.get(self.score_name) is not None:
            self.score_queue.append(logs[self.score_name])
        elif logs['info'].get(self.score_name) is not None:
            self.score_queue.append(logs['info'][self.score_name])
        else:
            raise LookupError()

    def get_current_value(self):
        if self.nb_steps > self.agent.step:
            score = 0

            if len(self.score_queue) == 0:
                pass
            elif self.score_type == 'mean':
                score = np.average(self.score_queue)
            elif self.score_type == 'max':
                score = np.max(self.score_queue)
            elif self.score_type == 'min':
                score = np.min(self.score_queue)

            return max(super().get_current_value(), min(self.str_eps, 1 - (score - self.min_score) / self.max_score))

        else:
            return self.value_min



class starcraft_multiagent_eGreedyPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    nb_actions = (64*64, 3)
    """
    def __init__(self, nb_agents, nb_actions, eps=.1):
        super(starcraft_multiagent_eGreedyPolicy, self).__init__()
        self.nb_agents = nb_agents
        self.nb_actions = nb_actions
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (list): [action_xy (np.array), action_type (np.array)]
            [(1, nb_agents, actions), (1, nb_agents, actions)]

        # Returns
            Selection action: [(x,y), nothing/attack/move]
            [(nb_agents, 1), (nb_agents, 1)]
        """
        assert len(q_values) == 2
        preference_xy = np.squeeze(q_values[0])
        preference_type = np.squeeze(q_values[1])

        actions_xy = []
        actions_type = []
        for agent in range(self.nb_agents):
            # select action_xy
            if np.random.uniform() < self.eps:
                action_xy = np.random.random_integers(0, self.nb_actions[0]-1)
            else:
                action_xy = np.argmax(preference_xy[agent])
            actions_xy.append(action_xy)

            # select action_type
            if np.random.uniform() < self.eps:
                action_type = np.random.random_integers(0, self.nb_actions[1]-1)
            else:
                action_type = np.argmax(preference_type[agent])
            actions_type.append(action_type)

        actions_xy = np.array(actions_xy)
        actions_type = np.array(actions_type)

        return [actions_xy, actions_type]

    def get_config(self):
        """Return configurations of EpsGreedyPolicy

        # Returns
            Dict of config
        """
        config = super(starcraft_multiagent_eGreedyPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class BoltzmannQPolicy(Policy):
    """Implement the Boltzmann Q Policy

    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """
    def __init__(self, tau=1., clip=(-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        """Return configurations of BoltzmannQPolicy

        # Returns
            Dict of config
        """
        config = super(BoltzmannQPolicy, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config


class MaxBoltzmannQPolicy(Policy):
    """
    A combination of the eps-greedy and Boltzman q-policy.

    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amsterdam, Amsterdam (1999)

    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    """
    def __init__(self, eps=.1, tau=1., clip=(-500., 500.)):
        super(MaxBoltzmannQPolicy, self).__init__()
        self.eps = eps
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action
        The selected action follows the BoltzmannQPolicy with probability epsilon
        or return the Greedy Policy with probability (1 - epsilon)

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(range(nb_actions), p=probs)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of MaxBoltzmannQPolicy

        # Returns
            Dict of config
        """
        config = super(MaxBoltzmannQPolicy, self).get_config()
        config['eps'] = self.eps
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config


class BoltzmannGumbelQPolicy(Policy):
    """Implements Boltzmann-Gumbel exploration (BGE) adapted for Q learning
    based on the paper Boltzmann Exploration Done Right
    (https://arxiv.org/pdf/1705.10257.pdf).

    BGE is invariant with respect to the mean of the rewards but not their
    variance. The parameter C, which defaults to 1, can be used to correct for
    this, and should be set to the least upper bound on the standard deviation
    of the rewards.

    BGE is only available for training, not testing. For testing purposes, you
    can achieve approximately the same result as BGE after training for N steps
    on K actions with parameter C by using the BoltzmannQPolicy and setting
    tau = C/sqrt(N/K)."""

    def __init__(self, C=1.0):
        assert C > 0, "BoltzmannGumbelQPolicy C parameter must be > 0, not " + repr(C)
        super(BoltzmannGumbelQPolicy, self).__init__()
        self.C = C
        self.action_counts = None

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        # We can't use BGE during testing, since we don't have access to the
        # action_counts at the end of training.
        assert self.agent.training, "BoltzmannGumbelQPolicy should only be used for training, not testing"

        assert q_values.ndim == 1, q_values.ndim
        q_values = q_values.astype('float64')

        # If we are starting training, we should reset the action_counts.
        # Otherwise, action_counts should already be initialized, since we
        # always do so when we begin training.
        if self.agent.step == 0:
            self.action_counts = np.ones(q_values.shape)
        assert self.action_counts is not None, self.agent.step
        assert self.action_counts.shape == q_values.shape, (self.action_counts.shape, q_values.shape)

        beta = self.C/np.sqrt(self.action_counts)
        Z = np.random.gumbel(size=q_values.shape)

        perturbation = beta * Z
        perturbed_q_values = q_values + perturbation
        action = np.argmax(perturbed_q_values)

        self.action_counts[action] += 1
        return action

    def get_config(self):
        """Return configurations of BoltzmannGumbelQPolicy

        # Returns
            Dict of config
        """
        config = super(BoltzmannGumbelQPolicy, self).get_config()
        config['C'] = self.C
        return config