# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from core.common.agent import Agent
from collections import deque
import random
import numpy as np


class DeepSARSAgent(Agent):
    def __init__(self, action_size, model, discount_factor=0.99, epsilon=1, epsilon_decay=0.999, epsilon_min=0.01
                 , **kwargs):
        """ Deep SARSA Agent example

        # Arguments
            action_size (integer): Number of action spaces
            model : Network model to train
            discount_factor (float): Discount factor for discounted reward
            epsilon (float): Current epsilon value
            epsilon_decay (float): Decaying value for epsilon
            epsilon_min (float): Min value of epsilon

        # Returns
            No returns
        """
        super(DeepSARSAgent, self).__init__(**kwargs)

        self.action_size = action_size
        self.model = model
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # memory for train (S,A,R,S',A')
        self.observations = deque(maxlen=2)
        self.recent_observation = None
        self.recent_action = None

    def forward(self, observation):
        """Get action to be taken from observation
            See the description in agent.py
        """
        if self.train_mode and np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state = np.float32(observation)
            q_values = self.model.predict(np.expand_dims(state, 0))
            action = np.argmax(q_values[0])

        # set memory for training
        self.recent_observation = observation
        self.recent_action = action

        return [action]

    def backward(self, reward, terminal):
        """ Updates the agent's network
                """
        self.observations.append([self.recent_observation, self.recent_action, reward, terminal])

        if self.step == 0:
            return

        # Decaying the epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Use a memory to train
        experience = self.observations.popleft()
        state = np.float32(experience[0])
        action = experience[1]
        reward = experience[2]
        done = experience[3]

        # Get next action on next state from current model
        next_state = np.float32(self.recent_observation)
        next_action = self.forward(next_state)

        # Compute Q values for target network update
        # Q(S,A) <- Q(S,A) + alpha(R + gammaQ(S',A') - Q(S,A))
        target = self.model.predict(np.expand_dims(state, 0))[0]
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(np.expand_dims(next_state, 0))[0][next_action])

        target = np.reshape(target, [1, self.action_size])

        self.model.fit(np.expand_dims(state, 0), target, epochs=1, verbose=0)
        return

    def compile(self, optimizer, metrics=[]):
        """Compile the model
        """
        self.model.compile(optimizer=optimizer, loss='mse')

        self.compiled=True
        return

    def load_weights(self, filepath):
        """ Load trained weight from an HDF5 file.
        """
        self.model.load_weights(filepath)
        return

    def save_weights(self, filepath, overwrite=False):
        """ Save trained weight from an HDF5 file.
        """
        self.model.save_weights(filepath, overwrite)
        return
