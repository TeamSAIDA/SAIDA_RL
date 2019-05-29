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
    params[OPS.MOVE_ANG.value] = [10, 15]
    params[OPS.MOVE_DIST.value] = [2]
    params[OPS.GAMMA.value] = [0.99]
    params[OPS.EPOCHS.value] = [10]
    filename = 'avoid_reavers_PPO.py'

    saida_util.auto_executor(params, os.path.realpath(filename))
