# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

import sys, os
from time import sleep

from gym import Env

import saida_gym.envs.conn.shm as shm
import saida_gym.envs.conn.connection_env as env


class SAIDAGym(Env):

    @staticmethod
    def get_proto_msg(protobuf_name):
        return __import__('%s' %protobuf_name, fromlist=[protobuf_name])

    def __init__(self, gym_name, version=0, action_type=0, frames_per_step=6, move_angle=30, move_dist=4
                 , verbose=0, protobuf_name='saida_gym.envs.protobuf.common_pb2', connect_type=env.CONNECT_TYPE[0]
                 , bot_runner=r"..\..\..\cpp\Release\SAIDA\SAIDA.exe", **kwargs):
        if bot_runner is not None:
            os.startfile(bot_runner)

        print("Initialize...")
        self.verbose = verbose
        self.action_type = action_type
        self.map_version = version
        self.frames_per_step = frames_per_step
        self.move_angle = move_angle
        self.move_dist = move_dist

        if connect_type == env.CONNECT_TYPE[0]:
            self.conn = shm.SharedMemory(gym_name, version, verbose=self.verbose, **kwargs)

            if self.conn is None:
                print('connection failed.')
                sys.exit(0)

        self.message = self.get_proto_msg(protobuf_name)
        self.common_message = self.get_proto_msg('saida_gym.envs.protobuf.common_pb2')

        if self.message is None:
            print('cannot find protobufName module.', protobuf_name)
            sys.exit(0)

        if 'InitReq' in dir(self.message):
            self.init_req_msg = self.message.InitReq()
        else:
            self.init_req_msg = self.common_message.InitReq()

        if 'InitRes' in dir(self.message):
            self.init_res_msg = self.message.InitRes()
        else:
            self.init_res_msg = self.common_message.InitRes()

        if 'ResetReq' in dir(self.message):
            self.reset_req_msg = self.message.ResetReq()
        else:
            self.reset_req_msg = self.common_message.ResetReq()

        if 'ResetRes' in dir(self.message):
            self.reset_res_msg = self.message.ResetRes()
        else:
            self.reset_res_msg = self.common_message.ResetRes()

        if 'StepReq' in dir(self.message):
            self.step_req_msg = self.message.StepReq()
        else:
            self.step_req_msg = self.common_message.StepReq()

        if 'StepRes' in dir(self.message):
            self.step_res_msg = self.message.StepRes()
        else:
            self.step_res_msg = self.common_message.StepRes()

        if 'CloseReq' in dir(self.message):
            self.close_req_msg = self.message.Close()
        else:
            self.close_req_msg = self.common_message.Close()

        # 서버 측 연결 대기
        create_val = self.conn.read()

        if create_val[0] != "Create":
            print('Initializing failed.. message is not for Create.', create_val[0])
            sys.exit(0)

        self.init_req_msg.Clear()
        self.conn.write(env.MESSAGE_TYPE["Init"], self.make_initial_msg(self.init_req_msg))
        init_val = self.conn.read()

        if init_val[0] != "Init":
            print('Initializing failed.. message is not for Init.', init_val[0])
            sys.exit(0)

        self.init_res_msg.Clear()
        self.init_res_msg.ParseFromString(init_val[1])

        self.set_initial_data(self.init_res_msg)
        self.state = None
        self.reward = 0
        self.done = 0
        self.info = {}

        return

    def make_initial_msg(self, msg):
        msg.action_type = self.action_type
        msg.version = self.map_version
        msg.frames_per_step = self.frames_per_step
        msg.move_angle = self.move_angle
        msg.move_dist = self.move_dist
        return msg

    def set_initial_data(self, msg):
        raise NotImplementedError

    def step(self, action, **kwargs):
        self.step_req_msg.Clear()
        self.conn.write(env.MESSAGE_TYPE["Step"], self.make_step_msg(self.step_req_msg, action, **kwargs))

        step_val = self.conn.read()

        if step_val[0] != "Step":
            print("Step Request Failed!!")
            sys.exit(0)

        self.step_res_msg.Clear()
        self.step_res_msg.ParseFromString(step_val[1])

        return self.set_step_data(self.step_res_msg, action=action, **kwargs)

    def make_step_msg(self, msg, actions, **kwargs):
        for val in actions:
            act = msg.action.add()
            if self.action_type == 0:
                act.action_num = val
            elif self.action_type == 1:
                act.pos_x = val[0]
                act.pos_y = val[1]
                act.action_num = val[2]
            elif self.action_type == 2:
                act.radius = val[0]
                act.angle = val[1]
                act.action_num = val[2]
            else:
                act.action_num = val

        return msg

    def set_step_data(self, msg, **kwargs):
        self.state = msg.next_state
        self.reward = msg.reward
        self.done = False if msg.done == 0 else True
        self.info = {'infoMsg': msg.info}

        if self.verbose > 1:
            print(msg)
        if self.verbose > 0:
            print(msg.next_state)

        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.reset_req_msg.Clear()
        self.conn.write(env.MESSAGE_TYPE["Reset"], self.make_reset_msg(self.reset_req_msg))

        msg = self.conn.read()

        if msg[0] != "Reset":
            print("reset Failed!!")
            sys.exit(0)

        self.reset_res_msg.Clear()
        self.reset_res_msg.ParseFromString(msg[1])

        return self.set_reset_data(self.reset_res_msg)

    def make_reset_msg(self, msg):
        msg.content = "reset content"
        return msg

    def set_reset_data(self, msg):
        self.state = msg.next_state

        if self.verbose > 1:
            print(msg)
        if self.verbose > 0:
            print(msg.next_state)

        return self.state

    def close(self):
        self.close_req_msg.Clear()
        self.conn.close(self.close_req_msg)
        sleep(1)

    def render(self, msg=None):
        if msg is None:
            raise NotImplementedError
        else:
            self.conn.write(env.MESSAGE_TYPE["Render"], msg)

            msg = self.conn.read()

            if msg[0] != "Render":
                print("render Failed!!")
                sys.exit(0)

    def set_log_level(self, level):
        self.verbose = level
