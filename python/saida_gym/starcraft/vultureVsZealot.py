# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

import numpy as np
from gym import spaces
from saida_gym.envs.SAIDAGym import SAIDAGym


class VultureVsZealot(SAIDAGym):
    def __init__(self, shm_name='SAIDA_VZ', **kwargs):
        super().__init__(gym_name='VultureVsZealot', name=shm_name, **kwargs)

    def set_initial_data(self, msg):
        self.action_space = spaces.Discrete(msg.num_action_space)

        for m in msg.unit_type_map:
            setattr(self, m.key, m.value)

        # Walkable Map is only For GroundUnits
        self.walkableMap = np.reshape(msg.iswalkable, [512, 512])

        # human define observationSpace Size
        self.observation_space = spaces.Discrete(31)

    def render(self):
        print("render")
