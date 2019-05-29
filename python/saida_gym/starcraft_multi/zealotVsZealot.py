# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

import numpy as np
from saida_gym.envs.SAIDAGym import SAIDAGym


class ZealotVsZealot(SAIDAGym):
    def __init__(self, shm_name='SAIDA_ZVZ', **kwargs):
        # todo : move this to cpp
        self.nb_agents = 3

        super().__init__(gym_name='ZealotVsZealot', name=shm_name, **kwargs)

    def set_initial_data(self, msg):

        if self.action_type == 0:
            self.action_space = (self.nb_agents, msg.num_action_space)
            self.observation_space = None
        elif self.action_type == 1 or self.action_type == 2:
            self.action_space = (self.nb_agents, msg.num_action_space, 2)
            # human define observationSpace Size
            self.observation_space = None

        for m in msg.unit_type_map:
            setattr(self, m.key, m.value)

        # Walkable Map is only For GroundUnits
        self.walkableMap = np.reshape(msg.iswalkable, [512, 512])

    def render(self):
        print("render")
