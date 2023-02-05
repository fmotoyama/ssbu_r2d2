# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:08:20 2022

@author: motoyama
"""
import datetime
from pathlib import Path
import numpy as np

from .pole import Pole



class R2D2config:
    def __init__(self):
        # game
        self.name = 'Pole'
        
        self.save_path = Path("checkpoints") / (f'{self.name}_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.load_file = None
        #self.load_file = r'checkpoints\Pole_2022-12-21-14-55-51\100000.chkpt'
        
        # Network
        env = Pole()
        self.state_size = np.prod(env.observation_shape)
        self.action_size = np.prod(env.action_shape)
        self.action_shape = env.action_shape
        #self.hidden_size = 32
        self.hidden_size = 256
        
        # Learner
        self.steps = 500000
        #self.steps = 12
        self.learn_on_gpu = True
        self.batch_size = 32
        self.memory_max = 1e5
        self.save_every = self.steps//4     # チェックポイントをセーブする間隔
        self.epsilon_min = 0.01             # epsilon-greety
        self.epsilon_max = 0.4
        self.gamma = 0.997                  # Q学習の割引率
        self.lr = 0.00025                   # optimizerの学習率
        self.burnin = self.batch_size * 100 # トレーニングを始める前に行うステップ数
        self.sync_every = 1e3               # Q_targetにQ_onlineを同期する間隔
        self.send_param_every = 10          # learnerが最新のパラメータを置く間隔
        # PER
        self.PER_epsilon = 0.001            # 重さが0になることを防ぐ微小量
        self.PER_alpha = 0.6                # 0~1 0のとき完全なランダムサンプリング
        # 重要度サンプリング
        self.IS_beta = 0.4                  # 補正の強さ 固定値
        # R2D2
        self.r2d2_burnin = 5
        self.input_length = 10              # 学習に用いる時系列長
        
        # Actor
        self.actors_emmu_local = 1
        self.actors_emmu_remote = 0
        self.actors_switch = 0
        self.actors = self.actors_emmu_local + self.actors_emmu_remote + self.actors_switch
        self.play_on_gpu = False
        self.sync_param_every = 100         # モデルをLearnerと同期するステップ間隔
        self.send_size = 50                 # memoryに送る遷移情報のサイズ
        
        self.env_ids = [f'emmu_local{i}' for i in range(self.actors_emmu_local)]\
                         + [f'emmu_remote{i}' for i in range(self.actors_emmu_remote)]\
                         + [f'switch{i}' for i in range(self.actors_switch)]
        



"""
class Game:
    def __init__(self, seed):
        self.env = Pole()
    
    def step(self, action):
        observation, reward, done = self.env.step(action)
        return observation, reward, done
    
    def reset(self):
        return self.env.reset()
    
    def render(self, outpath=None):
        if outpath == None:
            self.env.render()
        else:
            self.env.render(outpath)
"""

class Game:
    def __init__(self, env_name, env_num):
        self.env = Pole()
        self.env_name = env_name
        self.env_num = env_num
        self.env_id = env_name + str(env_num)
        
        def dummy():
            pass
        self.env.get_emmu_Handles = dummy
    
    def step(self, action):
        observation, reward, done = self.env.step(action)
        observation = np.ravel(observation)     # observationを平坦化
        return observation, reward, done
    
    def reset(self):
        observation = np.ravel(self.env.reset())
        return observation
    
    def close(self):
        return self.env.close()