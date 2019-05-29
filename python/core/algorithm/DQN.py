# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from core.common.agent import Agent
from core.common.util import *
from core.policies import *
from keras.layers import Lambda, Input, Dense, Add
from keras.losses import mean_absolute_error

class DQNAgent(Agent):

    def __init__(self, model, nb_actions, memory, discount_factor=0.99, batch_size=32, train_interval=1000,
                 target_model_update=10000, delta_clip=np.inf, warmup_step_cnt=1000, enable_dueling=False,
                 memory_interval=1, enable_double=False, dueling_type='avg', policy=None, test_policy=None,
                 enable_encouraged_action=False, enable_discouraged_action=False, action_affected_observation_space=None,
                 enable_pop_art = False,
                 **kwargs):
        """
        Constructor function for DQN Agent. It can create agent setting hyper parameters and other configuration you want.

        # Arguments
        :param model(Keras Model): DQN network
        :param nb_actions(int): size of action space
        :param memory(int): replay memory size
        :param discount_factor(float): discount reward factor, gamma, default value = 0.99
        :param batch_size(int): size of sample to train in backward, default = 32
        :param train_interval(int): interval period to train
        :param target_model_update(int or float): if it is between 0 and 1, soft target model update. if it is over 1, hard target model update
        :param delta_clip(int): loss clip range (min, max)
        :param warmup_step_cnt(int): step count for agent to warm up before its training
        :param enable_dueling(boolean): enable dueling network, you can use both dueling an double at the same time if you want
        :param enable_double(boolean): enable double network, you can use both dueling an double at the same time if you want
        :param dueling_type(boolean): you can use choose among (avg, max, naive). default value : avg
        :param policy(Policy):  default value : LinearAnnealedPolicy
        :param test_policy(Policy): default value : EpsGreedyQPolicy
        :param kwargs: you can add extra parameters to customize agent.
        :param enable_encouraged_action(boolean) : enable encouraged action inspired by Hindsight Experience Replay, default = False,
        :param enable_discouraged_action(boolean) : enable discouraged action inspired by Hindsight Experience Replay, default = False,
        :param enable_pop_art(boolean) : enable pop-art to scale reward adaptively
        :param action_affected_observation_space(tuple) : observation region to be affected by action, if enable_discouraged_action, it should not to be None, default =None
        #Return
        Agent class
        """

        super(DQNAgent, self).__init__(**kwargs)

        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))

        self.enable_dueling = enable_dueling
        self.enable_double = enable_double
        self.enable_encouraged_action = enable_encouraged_action
        self.enable_discouraged_action = enable_discouraged_action
        self.enable_pop_art = enable_pop_art
        self.action_affected_observation_space = action_affected_observation_space

        if self.enable_discouraged_action:
            if self.action_affected_observation_space is None:
                assert False, "if enable_discouraged_action is True, action_affected_observation_space has tuple value."

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        self.dueling_type = dueling_type

        self.render = False

        # DQN 하이퍼파라미터
        self.nb_actions = nb_actions
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.warmup_step_cnt = warmup_step_cnt
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.memory = memory
        self.enable_PER = hasattr(self.memory, 'enable_per') and getattr(self.memory, 'enable_per')

        if self.enable_dueling:
            # get the second last layer of the model, abandon the last layer
            layer = model.layers[-2]
            nb_action = model.output._keras_shape[-1]
            # layer y has a shape (nb_action+1,)
            # y[:,0] represents V(s;theta)
            # y[:,1:] represents A(s,a;theta)
            y = Dense(nb_action + 1, activation='linear')(layer.output)
            # caculate the Q(s,a;theta)
            # dueling_type == 'avg'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            # dueling_type == 'max'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            # dueling_type == 'naive'
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            if self.dueling_type == 'avg':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"

            model = Model(inputs=model.input, outputs=outputlayer)

        # note : pop-art add layer
        if self.enable_pop_art:
            if self.enable_dueling:
                layer = model.layers[-1]
            else:
                layer = model.layers[-2]
            self.popart = PopArtLayer()
            normalized_output_layer = self.popart(layer.output)
            model = Model(inputs=model.input, outputs=normalized_output_layer)

        # Related objects.
        self.model = model
        if policy is None:
            policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        self.reset_states()


    def reset_states(self):
        """
        you can specify any logic which run whenever episode ends.
        """
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    # 입실론 탐욕 정책으로 행동 선택
    def forward(self, observation):
        """
        choose action
        :param observation: observation which is used for agent to choose action
        :return: action
        """
        state = self.memory.get_recent_state(observation)

        # in here, there is a place where we can pre-process observation
        batch = self.process_state_batch([state])

        q_values = self.model.predict_on_batch(batch).flatten()

        assert q_values.shape == (self.nb_actions,)

        actions = []

        if self.train_mode:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        actions.append(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = actions

        return actions

    def append_replay_memory(self, reward, terminal):
        """

        :param reward:
        :param terminal:
        :return:
        """
        self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                           training=self.train_mode)

    def backward(self, reward, terminal):
        """

        :param reward:
        :param terminal:
        :return:
        """

        # Train the network on a single stochastic batch.
        if self.step > self.warmup_step_cnt and self.step % self.train_interval == 0:
            if self.enable_PER:
                experiences, b_weights, b_idxes = self.memory.sample(self.batch_size)
                sample_weight = {'dummy_targets': b_weights, 'targets': b_weights}

                if self.step > self.memory.limit:
                    self.memory.per_beta += (1.0 - self.memory.per_beta) / (self.nb_steps - self.memory.limit)
            else:
                experiences = self.memory.sample(self.batch_size)
                sample_weight = None

            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            state1_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            terminal1_batch = np.array(terminal1_batch)

            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)  # todo : need to be de-normalized
                if self.enable_pop_art:
                    q_values = self.popart.de_normalize(q_values)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(state1_batch)  # todo : need to be de-normalized
                if self.enable_pop_art:
                    target_q_values = self.popart.de_normalize(target_q_values)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)  # todo : need to be de-normalized
                if self.enable_pop_art:
                    target_q_values = self.popart.de_normalize(target_q_values)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.discount_factor * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch

            if self.enable_PER:
                action_values = self.model.predict_on_batch(state0_batch)  # todo : need to be de-normalized
                if self.enable_pop_art:
                    action_values = self.popart.de_normalize(action_values)
                td_error = np.zeros(self.batch_size)
                for i in range(self.batch_size):
                    td_error[i] = Rs[i] - action_values[i][action_batch[i]]

                # sample_weight: Optional Numpy array of weights for the training samples,
                # used for weighting the loss function (during training only)
                new_priorities = np.abs(td_error) + 0.00001
                self.memory.update_priorities(b_idxes, new_priorities)

            if self.enable_discouraged_action:
                if self.enable_double:
                    q_values_a0 = self.model.predict_on_batch(state0_batch)  # todo : need to be de-normalized
                else:
                    q_values_a0 = self.target_model.predict_on_batch(state0_batch)  # todo : need to be de-normalized
                if self.enable_pop_art:
                    q_values_a0 = self.popart.de_normalize(q_values_a0)

                # note : experimental
                axis0_min = self.action_affected_observation_space[0][0]  # for atari, 76:77, 4:76
                axis0_max = self.action_affected_observation_space[0][1]

                if len(self.action_affected_observation_space) == 2:
                    axis1_min = self.action_affected_observation_space[1][0]
                    axis1_max = self.action_affected_observation_space[1][1]

                    if self.enable_encouraged_action:
                        discourage_state0_batch = state0_batch[0][:, -1, axis0_min:axis0_max, axis1_min:axis1_max]
                        discourage_state1_batch = state1_batch[0][:, -1, axis0_min:axis0_max, axis1_min:axis1_max]
                    else:
                        discourage_state0_batch = state0_batch[:, -1, axis0_min:axis0_max, axis1_min:axis1_max]
                        discourage_state1_batch = state1_batch[:, -1, axis0_min:axis0_max, axis1_min:axis1_max]
                else:
                    if self.enable_encouraged_action:
                        discourage_state0_batch = state0_batch[0][:, -1, axis0_min:axis0_max]
                        discourage_state1_batch = state1_batch[0][:, -1, axis0_min:axis0_max]
                    else:
                        discourage_state0_batch = state0_batch[:, -1, axis0_min:axis0_max]
                        discourage_state1_batch = state1_batch[:, -1, axis0_min:axis0_max]

                self.trainable_model.train_on_batch(
                    ins + [targets, masks, discourage_state0_batch, discourage_state1_batch, q_values_a0], [dummy_targets, targets], sample_weight=sample_weight)
            else:
                self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets], sample_weight=sample_weight)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

    def _on_episode_end(self,episode, episode_logs):
        self.policy.on_episode_end(episode, episode_logs)

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def compile(self, optimizer, metrics=[]):

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        if self.enable_pop_art:
            self.target_model = clone_model(self.model, custom_objects={'PopArtLayer':self.popart})
        else:
            self.target_model = clone_model(self.model, custom_objects={})
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))

        def calculate_loss_model_based_state_difference(args):
            """
            here we can calculate how different between present state and next state to evalute how good action it would be.
            then, if the action selected has good action value compared to other actions, the different become less important

            alpha =  calculate how different between present state and next state to evalute how good action it would be.
            beta = if the action selected has good action value compared to other actions, the different become less important

            alpha * beta == loss

            """
            next_state, pred_next_state = args
            loss = mean_absolute_error(next_state, pred_next_state)  # alpha
            return K.sum(loss, axis=-1)

        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])

        def calculate_loss_normalized(args):
            mask, q_values_a0, loss = args
            return loss * (K.sum(q_values_a0 * mask) / K.sum(q_values_a0))

        # note : experimental
        if self.enable_discouraged_action:
            # for atari, self.action_affected_observation_space would be  [(76,77), (4, 76)]

            axis0_min = self.action_affected_observation_space[0][0]  # for atari, 76:77, 4:76
            axis0_max = self.action_affected_observation_space[0][1]

            nb_observation = (axis0_max-axis0_min, )

            if len(self.action_affected_observation_space) == 2:
                axis1_min = self.action_affected_observation_space[1][0]
                axis1_max = self.action_affected_observation_space[1][1]

                nb_observation = (axis0_max-axis0_min, axis1_max-axis1_min)

            mb_state = Input(name='model_based_next_state', shape=nb_observation)  # for atari, (1, 72)
            mb_next_state = Input(name='model_based_predicted_next_state', shape=nb_observation)
            q_values_a0 = Input(name='state0_action0_value', shape=(self.nb_actions,))
            loss_mb_out = Lambda(calculate_loss_model_based_state_difference, output_shape=(1,), name='mb_loss')(
                [mb_state, mb_next_state])
            loss_final = Lambda(calculate_loss_normalized, output_shape=(1,), name='normalize_mb_loss')(
                [mask, q_values_a0, loss_mb_out])
            loss_out = Add()([loss_final, loss_out])

        ins = [self.model.input] if type(self.model.input) is not list else self.model.input

        # note : experimental
        if self.enable_discouraged_action:
            trainable_model = Model(inputs=ins + [y_true, mask, mb_state, mb_next_state, q_values_a0],
                                    outputs=[loss_out, y_pred])
        else:
            trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])

        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        else:
            if hasattr(self.processor, 'process_state_batch'):
                return self.processor.process_state_batch(batch)
            else:
                return batch


    def save_weights(self, filepath, overwrite=False, force=False):
        """

        :param filepath:
        :param overwrite:
        :return:
        """
        if force:
            self.model.save_weights(filepath, overwrite)
        else:
            path = filepath.split('.')

            if self.enable_dueling and self.enable_double:
                algorithm_name = '_DDDQN_'
            elif self.enable_dueling:
                algorithm_name = '_DuelingDQN_'
            elif self.enable_double:
                algorithm_name = '_DoubleDQN_'
            else:
                algorithm_name = '_DQN_'

            filepath = ''.join(path[0:-1]) + algorithm_name + yyyymmdd24hhmmss() + '.' + path[-1]
            self.model.save_weights(filepath, overwrite)

    def load_weights(self, filepath):
        """

        :param filepath:
        :return:
        """
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)
