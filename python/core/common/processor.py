# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun


class Processor(object):
    """Abstract base class for implementing processors.

    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.

    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own.
    """

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation, state_size=None):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            observation (object): An observation as obtained by the environment

        # Returns
            Observation obtained by the environment processed
        """
        return observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            reward (float): A reward as obtained by the environment

        # Returns
            Reward obtained by the environment processed
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            info (dict): An info as obtained by the environment

        # Returns
            Info obtained by the environment processed
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.

        # Arguments
            action (int): Action given to the environment

        # Returns
            Processed action given to the environment
        """
        return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.

        # Arguments
            batch (list): List of states

        # Returns
            Processed list of states
        """
        return batch

    @property
    def metrics(self):
        """The metrics of the processor, which will be reported during training.

        # Returns
            List of `lambda y_true, y_pred: metric` functions.
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []
