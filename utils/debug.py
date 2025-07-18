# -*- coding: utf-8 -*-
import pdb
import time
from utils.comm import is_leader


def set_trace():
    if is_leader():
        pdb.set_trace()
    else:
        time.sleep(10000000)
