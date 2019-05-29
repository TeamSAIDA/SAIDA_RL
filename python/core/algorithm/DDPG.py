# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

"""
referenced by
https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
"""
from core.common.agent import Agent
from keras.layers import Input
from core.policies import NoisePolicy
from core.common.random import *
from core.common.random import OrnsteinUhlenbeckProcess
from math import sqrt
from core.common.util import *
import keras.backend as K
from core.common.util import *
import tensorflow as tf
from core.policies import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy


def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1 - actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPGAgent(Agent):

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def __init__(self, actor, critic, action_shape, memory, critic_action_input,
                 policy=None, test_policy=None, discount_factor=0.99, learning_rate=0.001,
                 batch_size=32, train_interval=1, delta_clip=np.inf,
                 nb_warmup_critic_step_cnt=500, nb_warmup_actor_step_cnt=500, random_process=None,
                 tau_for_actor=0.001, tau_for_critic=0.001,
                 **kwargs):

        super(DDPGAgent, self).__init__(**kwargs)

        self.actor = actor
        self.critic = critic
        self.action_shape = action_shape
        self.memory = memory

        if policy is None:
            self.policy = NoisePolicy(random_process=OrnsteinUhlenbeckProcess(size=self.action_shape, theta=.15, mu=0., sigma=.2, n_steps_annealing=100000))
        else:
            self.policy = policy

        if test_policy is None:
            self.test_policy = NoisePolicy(random_process=OrnsteinUhlenbeckProcess(size=self.action_shape, theta=.15, mu=0., sigma=.1, n_steps_annealing=1))
        else:
            self.test_policy = test_policy

        self.critic_action_input = critic_action_input
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.delta_clip = delta_clip
        self.nb_warmup_critic_step_cnt = nb_warmup_critic_step_cnt
        self.nb_warmup_actor_step_cnt = nb_warmup_actor_step_cnt
        self.random_process = random_process
        self.tau_for_actor = tau_for_actor  # rate for updating actor target model softly
        self.tau_for_critic = tau_for_critic  # rate for updating critic target model softly
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)

        self.compiled = False
        self.reset_states()

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()

        self.recent_action = None
        self.recent_observation = None

        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()
            self.policy.reset_states()  # ou reset!!
            self.test_policy.reset_states()  # in test mode,  is it still available?


    def forward(self, observation):
        """
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

        :param observation:
        :return:
        """

        state = self.memory.get_recent_state(observation)

        # in here, there is a place where we can pre-process observation
        batch = self.process_state_batch([state])

        action = self.actor.predict_on_batch(batch).flatten()

        assert action.shape == (self.action_shape,)

        if self.train_mode:
            action = self.policy.select_action(action)[0]
        else:
            if self.test_policy is not None:
                action = self.test_policy.select_action(action)[0]
            else:
                if self.test_policy is not None:
                    action = self.test_policy.select_action(action)[0]
                else:
                    raise NameError('test policy needs to be fixed.')

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward(self, reward, terminal):

        self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                           training=self.train_mode)

        can_train_either = self.step > self.nb_warmup_critic_step_cnt or self.step > self.nb_warmup_actor_step_cnt
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)

            if len(action_batch.shape) == 1:
                action_batch = np.expand_dims(action_batch, axis=-1)

            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.action_shape)

            # Update critic, if warm up is over.
            if self.step > self.nb_warmup_critic_step_cnt:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.action_shape)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.discount_factor * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_warmup_actor_step_cnt:
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:]
                else:
                    inputs = [state0_batch]
                if self.uses_learning_phase:
                    inputs += [self.training]
                action_values = self.actor_train_fn(inputs)[0]
                assert action_values.shape == (self.batch_size, self.action_shape)

        # in original paper, there is no hard update
        # if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
        #     self.update_target_models_hard()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def compile(self, optimizer, metrics=[]):
        def mean_q(y_true, y_pred):
            return K.mean(K.max(y_pred, axis=-1))

        metrics += [mean_q]

        if not ( type(optimizer) in (list, tuple) and len(optimizer) == 2):
            raise ValueError(
                'More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')

        if type(optimizer[0]) is str:
            critic_optimizer = optimizers.get(optimizer[0])
        else:
            critic_optimizer = optimizer[0]

        if type(optimizer[1]) is str:
            actor_optimizer = optimizers.get(optimizer[1])
        else:
            actor_optimizer = optimizer[1]

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_actor = clone_model(self.actor)
        self.target_critic = clone_model(self.critic)

        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # here we don't train this network instead, we use our policy gradient to update this network.
        self.actor.compile(optimizer='adam', loss='mse')
        self.critic.compile(optimizer='adam', loss='mse')

        if self.tau_for_critic < 1.:
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.tau_for_critic)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        combined_inputs = []
        state_inputs = []

        # combine actor and critic
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                state_inputs.append(i)

        combined_inputs[self.critic_action_input_idx] = self.actor(state_inputs)

        combined_output = self.critic(combined_inputs)

        # todo : loss -K.mean(combined_output) is correct?
        updates = actor_optimizer.get_updates(params=self.actor.trainable_weights, loss=-K.mean(combined_output))

        if self.tau_for_actor < 1:
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.tau_for_actor)

        updates += self.actor.updates

        self.actor_train_fn = K.function(state_inputs + [K.learning_phase()], [self.actor(state_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer
        self.compiled = True

    def save_weights(self, filepath, filename, yyyymmdd=None,  overwrite=False):

        algorithm_critic_name = '_critic_'
        algorithm_actor_name = '_actor_'

        if yyyymmdd is None:
            yyyymmdd = yyyymmdd24hhmmss()
        critic_filepath = filepath + os.path.sep + filename + algorithm_critic_name + yyyymmdd + '.' + 'h5f'
        actor_filepath = filepath + os.path.sep + filename + algorithm_actor_name + yyyymmdd + '.' + 'h5f'

        self.critic.save_weights(critic_filepath, overwrite)
        print('{} file is saved as a critic model for evaluation'.format(critic_filepath))

        self.actor.save_weights(actor_filepath, overwrite)
        print('{} file is saved as a actor model for evaluation'.format(actor_filepath))

    def load_weights(self, filepath, filename):

        # algorithm_critic_name = '_critic_'
        # algorithm_actor_name = '_actor_'

        critic_path = filepath + os.path.sep + filename[0] + '.' + 'h5f'
        actor_path = filepath + os.path.sep + filename[1] + '.' + 'h5f'

        self.critic.load_weights(critic_path)
        print('{} file is loaded as a critic model for evaluation'.format(critic_path))

        self.actor.load_weights(actor_path)
        print('{} file is loaded as a actor model for evaluation'.format(actor_path))
