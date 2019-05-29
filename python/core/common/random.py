# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from __future__ import division
import numpy as np
import copy
from math import sqrt


class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1):
        super(GaussianWhiteNoiseProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.size = size

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab \
# https://www.quora.com/Why-do-we-use-the-Ornstein-Uhlenbeck-Process-in-the-exploration-of-DDPG
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = np.random.normal(self.mu,self.current_sigma,self.size)


class SimpleOUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size=1, mu=0, theta=0.05, sigma=0.25):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset_states()

    def reset_states(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


"""
inspired by paper "PARAMETER SPACE NOISE FOR EXPLORATION" https://arxiv.org/pdf/1706.01905.pdf
https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
"""
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)

def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist


# class OrnsteinUhlenbeckProcessWithAnnealing(AnnealedGaussianProcess):
#     """
#     Ornstein-Uhlenbeck Noise (original code by @slowbull)
#     """
#     def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, sigma_min=None, n_steps_annealing=100, size=1):
#         self.theta = theta
#         self.sigma = sigma
#         self.sigma_min = sigma_min
#         self.n_steps_annealing = n_steps_annealing
#         self.sigma_step = - self.sigma / float(self.n_steps_annealing)
#         self.x0 = x0
#         self.mu = mu
#         self.dt = dt
#         self.size = size
#
#     def sample(self, step):
#         sigma = max(sigma_min, self.sigma_step * step + self.sigma)
#         x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
#         self.x0 = x
#         return x
#
#     def reset_states(self):
#         self.x_prev = np.random.normal(self.mu,self.current_sigma,self.size)
