# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 13:48:04 2023

@author: fmotoyama
"""
import traceback
import numpy as np
import multiprocessing as mp
from game.ssbu_config import R2D2config, Game


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def worker():
    env_type = 'emmu_local'
    config = R2D2config()
    game = Game(0, env_type)
    
    config.state_size = game.state_size
    config.action_size = game.action_size
    
    game.reset()
    prev_reward = 0
    while True:
        #action = int(input('action:'))
        action = np.random.randint(game.action_size)
        obs,reward,done = game.step(action)
        if reward != 0 or prev_reward != 0:
            print(reward)
            #break
        if done:
            game.reset()
            game.close()
            break
        prev_reward = reward


if __name__ == '__main__':
    
    
    
    """
    worker(config,game)
    """
    child_process = Process(
        target=worker,
        args=()
        )
    
    # 各サブプロセスを開始
    child_process.start()
    child_process.join()
    e = child_process.exception
    
    #"""