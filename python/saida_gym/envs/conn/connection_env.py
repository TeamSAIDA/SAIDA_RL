# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

#import saida_gym.envs.starcraft.shm

MESSAGE_TYPE = {
    "Create": "Create",
    "Init": "Init",
    "Reset": "Reset",
    "Step": "Step",
    "Close": "Close",
    "Render": "Render"
}

CONNECT_TYPE = {
    0: "SharedMemory",
    1: "TCP"
}


class Connection:
    #
    # def __init__(self, conn_type=CONNECT_TYPE[0]):
    #     self.conn_type = conn_type
    #
    #     self.conn = None
    #     if conn_type is CONNECT_TYPE[0]:
    #         self.conn = shm.SharedMemory(1024 * 1024 * 5, "SAIDA_RL")
    #     else:
    #         self.conn = None
    #
    #     return self.conn

    def write(self,message_type, write_msg):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError



