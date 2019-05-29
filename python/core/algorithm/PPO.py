# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun


from core.common.agent import Agent
from core.common.util import *
import math


class PPOAgent(Agent):

    def __init__(self, state_size, action_size, continuous, actor, critic,
                 discount_factor=0.99, loss_clipping=0.2, epochs=10, noise=1.0, entropy_loss=1e-3,
                 buffer_size=256, batch_size=64, **kwargs):
        """
            Constructor for PPO with clipped loss
         #Arguments
            state_size(integer): Number of state size
            action_size(integer): Number of action space
            continuous(bool): True if action space is continuous type
            actor(Keras Model): network for actor
            critic(Keras Model): network for critic
            discount_factor(float): discount reward factor
            loss_clipping(float): hyper parameter for loss clipping, in PPO paper, 0.2 is recommended.
            epochs(integer): hyper parameter
            noise(float): hyper parameter
            entropy_loss : hyper parameter
            buffer_size : hyper parameter
            batch_size : hyper parameter
        #Return
            None
        """
        super(PPOAgent, self).__init__(**kwargs)

        self.action_size = action_size
        self.state_szie = state_size
        self.continuous = continuous
        self.critic = critic
        self.actor = actor

        self.episode = 0

        self.discount_factor = discount_factor
        self.loss_clipping = loss_clipping  # Only implemented clipping for the surrogate loss, paper said it was best
        self.epochs = epochs
        self.noise = noise  # Exploration noise
        self.entropy_loss = entropy_loss

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.dummy_action,self.dummy_value = np.zeros((1, action_size)), np.zeros((1, 1))

        self.observation = None
        self.reward = []
        self.reward_over_time = []
        self.gradient_steps = 0

        self.batch = [[], [], [], []]
        self.tmp_batch = [[], [], []]

    def reset_env(self):
        self.reward = []

    def discounted_reward(self):
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * self.discount_factor

    def forward(self, observation):

        self.observation = observation
        self.tmp_batch[0].append(observation)

        if self.continuous:
            p = self.actor.predict([self.observation.reshape(1, self.state_szie), self.dummy_value, self.dummy_action])
            action = action_matrix = p[0] + np.random.normal(loc=0, scale=self.noise, size=p[0].shape)
            self.tmp_batch[1].append(action_matrix)
            self.tmp_batch[2].append(p)

            return action, action_matrix, p
        else:
            state = np.reshape(self.observation, [1, self.state_szie])
            p = self.actor.predict([state, self.dummy_value, self.dummy_action])
            action = np.random.choice(self.action_size, p=np.nan_to_num(p[0]))
            action_matrix = np.zeros(self.action_size)
            action_matrix[action] = 1

            self.tmp_batch[1].append(action_matrix)
            self.tmp_batch[2].append(p)

            return [action]

    def backward(self, reward, terminal):

        if self.train_mode is False:
            return

        self.reward.append(reward)

        if terminal:
            self.discounted_reward()
            for i in range(len(self.tmp_batch[0])):
                obs, action, pred = self.tmp_batch[0][i], self.tmp_batch[1][i], self.tmp_batch[2][i]
                r = self.reward[i]
                self.batch[0].append(obs)
                self.batch[1].append(action)
                self.batch[2].append(pred)
                self.batch[3].append(r)

            self.tmp_batch = [[], [], []]
            self.reset_env()

        if len(self.batch[0]) >= self.buffer_size:

            obs, action, pred, reward = np.array(self.batch[0]), np.array(self.batch[1]), np.array(self.batch[2]), np.reshape(
                np.array(self.batch[3]), (len(self.batch[3]), 1))
            self.batch = [[], [], [], []]

            pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
            obs, action, pred, reward = obs[:self.buffer_size], action[:self.buffer_size], \
                                        pred[:self.buffer_size], reward[:self.buffer_size]

            pred_values = self.critic.predict(obs)
            self.gradient_steps += 1

            # Calculate Loss
            advantage = reward - pred_values
            old_prediction = pred
            advantage = (advantage - advantage.mean()) / advantage.std()
            actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=self.batch_size,
                                        shuffle=True, epochs=self.epochs, verbose=False)
            critic_loss = self.critic.fit([obs], [reward], batch_size=self.batch_size, shuffle=True, epochs=self.epochs,
                                          verbose=False)

    def compile(self, optimizer, metrics=[]):
        """
        # Argument
            optimizer (object) : [0] = actor optimizer, [1] = critic optimizer
            metrics (Tensor) :  [0] = Keras Tensor as an advantage , [1] = Keras Tensor as an old_prediction
        # Return
            None
        """
        # Compile actor model
        advantage = metrics[0]
        old_prediction = metrics[1]
        if self.continuous:
            self.actor.compile(optimizer=optimizer[0],
                               loss=[self.proximal_policy_optimization_loss_continuous(
                                   advantage=advantage, old_prediction=old_prediction)])
        else:
            self.actor.compile(optimizer=optimizer[0],
                               loss=[self.proximal_policy_optimization_loss(
                                   advantage=advantage, old_prediction=old_prediction)])
        self.actor.summary()

        #Compile critic model
        self.critic.compile(optimizer=optimizer[1], loss='mse')

        self.compiled=True

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - self.loss_clipping,
                                                           max_value=1 + self.loss_clipping) * advantage) + self.entropy_loss * (
                                   prob * K.log(prob + 1e-10)))

        return loss

    def proximal_policy_optimization_loss_continuous(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            var = K.square(self.noise)
            pi = math.pi
            denom = K.sqrt(2 * pi * var)
            prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
            old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

            prob = prob_num / denom
            old_prob = old_prob_num / denom
            r = prob / (old_prob + 1e-10)

            return -K.mean(K.minimum(r * advantage,
                                     K.clip(r, min_value=1 - self.loss_clipping,
                                            max_value=1 + self.loss_clipping) * advantage))

        return loss

    def load_weights(self, file_path, filename):
        self.actor.load_weights(file_path[0])
        self.critic.load_weights(file_path[1])

    def save_weights(self, filepath, filename=None, overwrite=False):
        algorithm_critic_name = '_critic_'
        algorithm_actor_name = '_actor_'

        critic_filepath = filepath + os.path.sep + filename + algorithm_critic_name + yyyymmdd24hhmmss() + '.' + 'h5f'
        actor_filepath = filepath + os.path.sep + filename + algorithm_actor_name + yyyymmdd24hhmmss() + '.' + 'h5f'

        self.critic.save_weights(critic_filepath, overwrite)
        print('{} file is saved as a critic model for evaluation'.format(critic_filepath))

        self.actor.save_weights(actor_filepath, overwrite)
        print('{} file is saved as a actor model for evaluation'.format(actor_filepath))
        return [actor_filepath, critic_filepath]
