# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun

from core.common.util import OPS
import core.common.util as saida_util
import os


if __name__ == '__main__':
    params = {}

    params[OPS.NO_GUI.value] = [True]
    #params[OPS.DOUBLE.value] = [None]
    params[OPS.DUELING.value] = [True]
    params[OPS.DISCOUNT_FACTOR.value] = [0.99]
    params[OPS.BATCH_SIZE.value] = [700]
    params[OPS.REPLAY_MEMORY_SIZE.value] = [1000000]
    params[OPS.LEARNING_RATE.value] = [0.0005]
    params[OPS.WINDOW_LENGTH.value] = [2]
    params['-move-angle'] = [15, 20, 30]

    filename = 'vulture_vs_zealot_v0_DQN.py'

    saida_util.auto_executor(params, os.path.realpath(filename))
