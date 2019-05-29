# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

import numpy as np
from gym import spaces
from saida_gym.envs.SAIDAGym import SAIDAGym


class MarineVsZealot(SAIDAGym):
    def __init__(self, shm_name='SAIDA_MVZ', **kwargs):
        super().__init__(gym_name='MarineVsZealot', name=shm_name, **kwargs)

    def set_initial_data(self, msg):
        self.action_space = spaces.Discrete(msg.num_action_space)

        self.vultureType = msg.my_unit_type
        self.zealotType = msg.en_unit_type

        # Walkable Map is only For GroundUnits
        self.walkableMap = np.reshape(msg.iswalkable, [512, 512])

        # human define observationSpace Size
        self.observation_space = spaces.Discrete(31)

    def render(self):
        print("render")
