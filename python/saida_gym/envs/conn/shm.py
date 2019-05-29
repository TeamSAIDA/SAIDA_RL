# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

import sys
import os
import mmap
from time import sleep
import saida_gym.envs.protobuf.common_pb2 as cmn
import saida_gym.envs.conn.connection_env as env


class SharedMemory:
    def __init__(self, gym_name, version, size=2000000, name="SAIDA_RL", verbose=0, no_gui=False, **kwargs):
        self.verbose = verbose
        self.memory_size = size
        name = name + str(os.getpid())
        self.shakehands(gym_name, name, version, no_gui, **kwargs)

        self.connect(size, name)

    def connect(self, size, name):
        self.shm = mmap.mmap(0, size, name)
        if self.shm:
            self.shm.seek(0)  # 파일의 현재 위치 설정
            msg = self.shm.read(1).decode('ascii')
            if msg != 'S':
                print("Shared Memory create" + msg)
                self.write(env.MESSAGE_TYPE["Create"], "create")
            else:
                print(name + " Shared memory found.")
        else:
            self.shm = None
            print("Shared memory not found")

    def shakehands(self, gym_name, name, version, no_gui, local_speed=0, auto_kill=True, random_seed=-1):
        self.connect(100, 'SAIDA_INIT')

        init_val = self.read()

        if init_val[0] != "Create":
            print('Initializing failed.. message is not for Create.', init_val[0])
            sys.exit(0)

        msg = cmn.InitReq()
        msg.content=gym_name
        msg.content2=name
        msg.version=version
        msg.local_speed=0 if no_gui or 0 > local_speed else 1000 if local_speed > 1000 else local_speed
        msg.no_gui=no_gui
        msg.auto_kill_starcraft=auto_kill
        msg.random_seed=random_seed

        self.write(env.MESSAGE_TYPE["Init"], msg)

        self.read(end_on_close=False)

    def read(self, end_on_close=True):

        self.shm.seek(0)
        msg = self.shm.read(1).decode('ascii')

        sleepTime = 0
        while msg != 'S':
            self.shm.seek(0)
            msg = self.shm.read(1).decode('ascii')
            sleep(0.001)
            sleepTime += 0.001
            # if sleepTime > 30 :
            #     msg = None
            #     break

        if msg is None:
            return None

        if self.verbose > 2:
            self.shm.seek(0)  # 파일의 현재 위치 설정
            print("[Read]", self.shm.read(min(self.memory_size, 100)))

        self.shm.seek(2)

        msg = self.shm.read(4).decode('ascii')
        if msg == "Init" or msg == "Step":
            self.shm.seek(7) # S;Init;
            bData = self.shm.read()
        else:
            self.shm.seek(2)
            msg = self.shm.read(5).decode('ascii')
            if msg == "Close": # S;Close;
                self.shm.close()

                if end_on_close:
                    print('close msg is received.')
                    sys.exit(0)

                return None
            elif msg == "Reset":
                self.shm.seek(8) # S;Reset;
                bData = self.shm.read()
            else:
                self.shm.seek(2)
                msg = self.shm.read(6).decode('ascii')

                if msg == "Render":
                    self.shm.seek(9)  # S;Render;
                    bData = self.shm.read()
                elif msg == "Create":
                    self.shm.seek(9)  # S;Create;
                    bData = self.shm.read()

        idx = 0

        while bData[idx] != ord(';'):
            idx += 1

        msgLength = int(bData[0:idx])

        return [msg, bData[idx+1:idx +1+ msgLength]]

    def write(self, message_type, write_msg):
        if message_type is None:
            print("Please set messge_type")
            return False

        if self.shm is None:
            print("Can't find shared memory.")
            return False

        if type(write_msg) == str:
            message_length = len(write_msg)
        else:
            message_length = write_msg.ByteSize()

        msg = "S;" + message_type+";" + str(message_length) + ";"
        self.shm.seek(0)
        self.shm.write(bytes(msg, 'UTF-8'))

        byteMSg = bytes(msg, 'UTF-8')
        self.shm.seek(byteMSg.__len__())
        if type(write_msg) is str:
            self.shm.write(bytes(write_msg, 'UTF-8'))
        else:
            self.shm.write(write_msg.SerializeToString())

        if self.verbose > 2:
            self.shm.seek(1)
            print('[send]', "P", self.shm.read(min(self.memory_size, 100)))

        self.shm.seek(0)
        self.shm.write(bytes("P", 'UTF-8'))

        return True

    def close(self, msg):
        if not self.shm.closed:
            self.write(env.MESSAGE_TYPE["Close"], msg)
