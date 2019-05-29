# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

import os
from core.algorithm.QLearning import QLearningAgent
from saida_gym.starcraft.gridWorld import GridWorld


if __name__ == "__main__":
    env = GridWorld(verbose=0, local_speed=42)

    agent = QLearningAgent(env.action_space, learning_rate=1)

    env.set_agent(agent)

    for episode in range(100):
        state = env.reset()
        done = False

        while not done:
            env.render()

            # Choose ac action at this point
            action = agent.choose_action(state)

            # 1 step progress
            strAction = action;
            next_state, reward, done, _ = env.step([strAction])

            # update Q table
            agent.learn(state, action, reward, done, next_state)
            state = next_state

    env.close()
