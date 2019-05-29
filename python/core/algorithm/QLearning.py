# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from collections import defaultdict
import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.01, discount_factor=0.9, epsilon=0.9):
        # Action space =  [0,1,2,3] means UP, DOWN, LEFT, RIGHT
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: [0.0 for _ in range(actions.n)])

    # Update your model
    def learn(self, state, action, reward, done, next_state):
        if done:
            self.q_table[state][action] = reward
        else:
            # Q- Learning algorithm
            # Q(s_t, a_t) = Q(s_ t, a_t) + alpha[r_t + gamma * maxQ(s_t+1, a') - Q(s_t, a_t)]
            self.q_table[state][action] = self.q_table[state][action] + \
                                          self.learning_rate * (reward + self.discount_factor * max(self.q_table[next_state]) - self.q_table[state][action])

    # Choose an action
    def choose_action(self, state):
        # epsilon greedy strategy
        # choose random action with probability = epsilon
        if np.random.rand() > self.epsilon:
            action = self.actions.sample()
        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)
