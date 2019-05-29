# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from gym import spaces
from saida_gym.envs.SAIDAGym import SAIDAGym

class GridWorld(SAIDAGym):

    """
       Has the following members
       - nS: number of states
       - nA: number of actions
       - P: transitions (*)
       - isd: initial state distribution (**)

       (*) dictionary dict of dicts of lists, where
         P[s][a] == [(probability, nextstate, reward, done), ...]
       (**) list or array of length nS
    '
    """

    def __init__(self, **kwargs):
        super().__init__('GridWorld', protobuf_name='saida_gym.envs.protobuf.gridWorld_pb2', version=0, action_type=0
                         , frames_per_step=-1, move_angle=-1, move_dist=-1, **kwargs)
        self.nS = self.nrow * self.ncol

        self.observation_space = spaces.Discrete(self.nS)
        self.agent = None

    def set_initial_data(self, msg):
        self.nrow = msg.max_row
        self.ncol = msg.max_col
        self.action_space = spaces.Discrete(msg.num_action_space)

    def set_reset_data(self, msg):
        self.state = int(msg.next_state.index)
        return self.state

    def set_step_data(self, msg, **kwargs):
        self.state = int(msg.next_state.index)
        self.reward = msg.reward
        self.done = False if msg.done == 0 else True

        return self.state, self.reward, self.done, self.info

    def render(self):
        if self.agent is not None:
            renderMsg = self.message.RenderReq()

            for i in range(self.observation_space.n):
                for j in range(self.action_space.n):
                    renderMsg.q_table.append(self.agent.q_table[i][j])

            super().render(renderMsg)

    def set_agent(self, agent):
        self.agent = agent
