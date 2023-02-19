# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 13:48:04 2023

@author: fmotoyama
"""
import numpy as np
from game.ssbu_config import R2D2config, Env_emmu_local
from R2D2 import Process




def worker(game,config):
    game.get_emmu_Handles()
    game.reset()
    
    reward_sum = 0
    while True:
        action = np.random.randint(config.action_size)
        action = 0
        obs,reward,done,_ = game.step(action)
        
        reward_sum += reward
        if reward != 0:
            print(reward)
        if done:
            break
    
    print('reward_sum =', reward_sum)
    game.close()


if __name__ == '__main__':
    game = Env_emmu_local(0)
    config = R2D2config()
    
    #"""
    worker(game,config)
    """
    child_process = Process(
        target=worker,
        args=(game,config)
        )
    
    # 各サブプロセスを開始
    child_process.start()
    child_process.join()
    e = child_process.exception
    
    #"""