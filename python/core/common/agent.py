# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

import numpy as np
from copy import deepcopy
import warnings
import timeit
from core.common.processor import Processor

from core.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,

    History
)


class Agent(object):
    """Abstract base class for all implemented agents.

    Each agent interacts with the environment (as defined by the `Env` class) by first observing the
    state of the environment. Based on this observation the agent changes the environment by performing
    an action.

    Do not use this abstract base class directly but instead use one of the concrete agents implemented.
    Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
    interface, you can use them interchangeably.

    To implement your own agent, you have to implement the following methods:

    - `forward`
    - `backward`
    - `compile`
    - `load_weights`
    - `save_weights`

    # Arguments
        processor (`Processor` instance): See [Processor](#processor) for details.
    """
    def __init__(self, processor=None):
        self.processor = processor
        self.train_mode = False
        self.step = 0
        self.episode = 0
        self.compiled = False

        if not issubclass(self.processor.__class__, Processor):
            assert False, "Please set preocessor type as a Processor not {}".format(type(self.processor))


    def run(self, env, nb_steps, shared_cb_params={},train_mode=True, action_repetition=1, callbacks=None, verbose=1, visualize=False,
            nb_max_start_steps=0, random_policy=None, log_interval=10000, nb_max_episode_steps=None, nb_episodes=0):
        """Trains the agent on the given environment.

        # Arguments
            env: Environment instance that the agent interacts with.
            nb_steps (integer): Number of training steps to be performed.
            shared_cb_params (map) : Shared (key, value) parameters where it can be used in callbacks
            action_repetition (integer): Number of times the agent repeats the same action without observing the environment again.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError('Your tried to run your agent but it hasn\'t been compiled yet. Please call `compile()` before `run()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        self.episode_reward = None
        self.episode_step = None
        did_abort = False
        self.nb_steps = nb_steps
        self.visualize = visualize

        callbacks = [] if not callbacks else callbacks[:]

        if train_mode:
            if verbose == 1:
                callbacks += [TrainIntervalLogger(interval=log_interval)]
            elif verbose > 1:
                callbacks += [TrainEpisodeLogger()]
        else:
            callbacks += [TestLogger()]

        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)

        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)

        callbacks._set_env(env)

        shared_cb_params['nb_steps'] = nb_steps

        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(shared_cb_params)
        else:
            callbacks._set_params(shared_cb_params)


        if train_mode:
            self.train_mode = True
            self._on_train_begin()
            callbacks.on_train_begin()
        else:
            self.train_mode = False
            nb_max_start_steps = 0

        while self.step < nb_steps or self.episode < nb_episodes:
            if observation is None:  # start of a new episode
                callbacks.on_episode_begin(self.episode)
                self.episode_step = np.int16(0)
                self.episode_reward = np.float32(0)

                # Obtain the initial observation by resetting the environment.
                self.reset_states()
                observation = deepcopy(env.reset())

                if self.processor is not None:
                    if hasattr(self.processor, 'process_observation'):
                        observation = self.processor.process_observation(observation)
                assert observation is not None

                # Perform random starts at beginning of episode and do not record them into the experience.
                # This slightly changes the start position between games.
                nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                for _ in range(nb_random_start_steps):
                    if self.visualize:
                        env.render()
                    if random_policy is None:
                        action = env.action_space.sample()
                    else:
                        action = random_policy(observation)
                    if self.processor is not None:
                        action = self.processor.process_action(action)
                    callbacks.on_action_begin(action)
                    observation, reward, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, reward, done, info = self.preocessor.process_step(observation, reward, done, info)
                    callbacks.on_action_end(action)
                    if done:
                        warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                        observation = deepcopy(env.reset())
                        self._process_observation()

            # At this point, we expect to be fully initialized.
            assert self.episode_reward is not None
            assert self.episode_step is not None
            assert observation is not None

            callbacks.on_step_begin(self.episode_step)
            action = self.forward(observation)
            if self.processor is not None:
                if hasattr(self.processor, 'process_action'):
                    action = self.processor.process_action(action)
            reward = np.float32(0)
            #accumulated_info = {}
            done = False
            for _ in range(action_repetition):

                if self.visualize:
                    env.render()

                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(observation, r, done, info)
                # for key, value in info.items():
                #     if not np.isreal(value):
                #         continue
                #     if key not in accumulated_info:
                #         accumulated_info[key] = np.zeros_like(value)
                #     accumulated_info[key] += value
                callbacks.on_action_end(action)
                reward += r

                if done:
                    break
            if nb_max_episode_steps and self.episode_step >= nb_max_episode_steps - 1:
                # Force a terminal state.
                done = True

            self.append_replay_memory(reward, terminal=done)

            if train_mode:
                self.backward(reward, terminal=done)
            self.episode_reward += reward

            step_logs = {
                'action': action,
                'observation': observation,
                'reward': reward,
                'episode': self.episode,
                'info': info,
                # 'info': accumulated_info,
            }

            # step_logs.update(info)

            callbacks.on_step_end(self.episode_step, step_logs)
            self._on_step_end(self.episode_step, step_logs)
            self.episode_step += 1
            self.step += 1

            if done:
                # We are in a terminal state but the agent hasn't yet seen it. We therefore
                # perform one more forward-backward call and simply ignore the action before
                # resetting the environment. We need to pass in `terminal=False` here since
                # the *next* state, that is the state of the newly reset environment, is
                # always non-terminal by convention.
                self.forward(observation)

                self.append_replay_memory(reward, terminal=False)

                if train_mode:
                    self.backward(0., terminal=False)

                # This episode is finished, report and reset.
                episode_logs = {
                    'episode_reward': self.episode_reward,
                    'nb_episode_steps': self.episode_step,
                    'nb_steps': self.step,
                    'info': info,
                }

                self._on_episode_end(self.episode, episode_logs)
                callbacks.on_episode_end(self.episode, episode_logs)
                self.episode += 1
                observation = None
                self.episode_step = None
                self.episode_reward = None
        if self.train_mode:
            self._on_train_end()
            callbacks.on_train_end()

    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        pass

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    def append_replay_memory(self, reward, terminal):
        """save sates to replay buffer after having executed the action returned by `forward`.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.

        # Returns
            List of metrics values
        """
        pass

    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.

        # Returns
            List of metrics values
        """
        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.

        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def load_weights(self, filepath, filename):
        """Loads the weights of an agent from an HDF5 file.

        # Arguments
            filepath (str or list): The path to the HDF5 files. In case of algorithms using multiple models, it could be list of path of models.
            filename (str or list): The name to the HDF5 files. In case of algorithms using multiple models, it could be list of name of models.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, filename=None, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.

        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).

        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.

        # Returns
            A list of the model's layers
        """
        raise NotImplementedError()

    # def _on_episode_begin(self, episode):
    #     """Callback that is called before episode begins."
    #     """
    #     pass

    def _on_episode_end(self,episode, episode_logs):
        """Callback that is called before episode end."
        """
        """ Compute and print training statistics of the episode when done """
        pass

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        self.train_start = timeit.default_timer()
        print('Starting training for {} steps ...'.format(self.nb_steps))

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        self.train_end = timeit.default_timer()
        print('Training took {} seconds'.format(self.train_end - self.train_start))

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        self.test_start = timeit.default_timer()
        print('Starting training for {} steps ...'.format(self.nb_steps))

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        self.test_end = timeit.default_timer()
        print('Training took {} seconds'.format(self.test_end - self.test_start))

    def _on_step_end(self, step, logs={}):
        """ Update statistics of episode after each step """
        pass