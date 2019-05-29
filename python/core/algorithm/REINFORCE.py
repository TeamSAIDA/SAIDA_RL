# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from core.common.agent import Agent
from core.common.util import *


class ReinforceAgent(Agent):

    def __init__(self, state_size, action_size, model, discount_factor=0.99, **kwargs):
        """ REINFORCE agent example

        # Argument
            state_size (integer):  Number of state size
            action_size (integer): Number of action space
            model: Network model to train
            discount_factor (float): Discount factor for discounted reward

        # Returns
            No returns
        """
        super(ReinforceAgent, self).__init__(**kwargs)

        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.model = model

        self.train_model = None

        # list to calculate discounted reward
        self.states, self.actions, self.rewards = [], [], []

    def discount_rewards(self, rewards):
        """ calculate discounted rewards

        # Argument
            rewards (float): list of rewards

        # Returns
            List of discounted rewards
        """
        discounted_rewards = np.zeros_like(rewards)
        G_t = 0
        # G_t = r_(t+1) + discount_factor * G_(t+1)
        for t in reversed(range(0, len(rewards))):
            G_t = rewards[t] + G_t * self.discount_factor
            discounted_rewards[t] = G_t
        return discounted_rewards

    def forward(self, observation):
        """ Get action to be taken from observation
             See the description in agent.py
        """
        state = np.reshape(observation, [1, self.state_size])
        policy = self.model.predict(state)[0]
        action = np.random.choice(self.action_size, 1, p=policy)[0]

        # Append sample
        if self.train_mode:
            self.states.append(state[0])

            a = np.zeros(self.action_size)
            np.put(a, action, 1)

            self.actions.append(a)

        return [action]

    def backward(self, reward, terminal):
        """ Updates the agent
            See the description in agent.py
        """
        if not self.train_mode:
            return

        self.rewards.append(reward)

        # REINFORCE algorithm only train model
        # when the episode is end
        if not terminal:
            return

        # get discounted rewards
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)

        # for divide by zero error
        if np.std(discounted_rewards) == 0:
            self.states, self.actions, self.rewards = [], [], []
            return

        # normalize rewards
        discounted_rewards /= np.std(discounted_rewards)

        # train your model
        self.train_model([self.states, self.actions, discounted_rewards])

        # reset memory
        self.states, self.actions, self.rewards = [], [], []
        return

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent
            Define new optimizer instead of input optimizer
            See the description in agent.py
        """
        action = K.placeholder(shape=[None, self.action_size])
        discounted_rewards = K.placeholder(shape=[None, ])

        # define loss function as cross entropy
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        updates = optimizer.get_updates(self.model.trainable_weights, [],
                                        loss)

        train = K.function([self.model.input, action, discounted_rewards],
                           [], updates=updates)

        self.train_model = train
        self.compiled = True
        return

    def load_weights(self, file_path):
        self.model.load_weights(file_path)
        return

    def save_weights(self, file_path, overwrite=False):
        self.model.save_weights(file_path, overwrite)
        return
