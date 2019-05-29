# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

import numpy as np
from gym import spaces
from saida_gym.envs.SAIDAGym import SAIDAGym


class MarineScvDefence(SAIDAGym):
    def __init__(self, marine_count=3, zergling_count=4, shm_name='SAIDA_MSD', **kwargs):
        self.marine_count = marine_count
        self.zergling_count = zergling_count

        super().__init__(gym_name='MarineScvDefence', name=shm_name, **kwargs)

    def set_initial_data(self, msg):
        self.state = None
        self.action_space = spaces.Discrete(msg.num_action_space)
        # 아군 정보 7 * (아군수)
        # 적군 정보 7 * (적군수)
        # 지형정보
        # countdown time
        num_state_space = 7 * self.marine_count + 7 * self.zergling_count + 1
        self.observation_space = spaces.Discrete(num_state_space)

    def render(self):
        print("render")
