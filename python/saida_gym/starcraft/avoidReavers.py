# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from gym import spaces
from saida_gym.envs.SAIDAGym import SAIDAGym


class AvoidReavers(SAIDAGym):
    def __init__(self, shm_name='SAIDA_AR', **kwargs):
        super().__init__(gym_name='AvoidReavers', version=0, name=shm_name, **kwargs)

    def make_initial_msg(self, msg):
        msg.action_type = self.action_type
        msg.frames_per_step = self.frames_per_step
        msg.move_angle = self.move_angle
        msg.move_dist = self.move_dist
        return msg

    def set_initial_data(self, msg):
        self.state = None
        self.action_space = spaces.Discrete(msg.num_action_space)

    def render(self):
        print("render")
