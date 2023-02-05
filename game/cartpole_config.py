# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:08:20 2022

@author: motoyama
"""
import datetime
from pathlib import Path

from .cartpole import CartPole

class config:
    # game
    state_dim = CartPole.state_dim
    action_dim = CartPole.action_dim
    
    # learn
    episodes = 20000
    batch_size = 32
    memory_max = 1e5
    save_every = 2e6    # チェックポイントをセーブするステップ数の間隔
    gamma = 0.98        # Q学習の割引率
    lr = 0.00025        # optimizer
    burnin = 5000       # トレーニングを始める前に行うステップ数
    learn_every = 5     # Q_onlineを更新するステップ数の間隔
    sync_every = 1e4    # Q_targetにQ_onlineを同期するステップ数の間隔
    
    save_path = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    load_path=None
    #load_path = r'checkpoints\2022-09-29-04-20-27\4000000.chkpt'
    
    # epsilon_greety法 1のとき常にランダム
    exploration_rate = 1
    exploration_rate_decay = 0.99999975
    exploration_rate_min = 0.1
    
    # PER
    PER_epsilon = 0.001 # 重さが0になることを防ぐ微小量
    PER_alpha = 0.8     # 0~1 0のとき完全なランダムサンプリング
    # 重要度サンプリング
    PER_beta = 0.4      # 補正の強さ
    PER_beta_max = 1.0
    PER_beta_steps = 7000000
    # multi step learning
    multi_steps = 3
    
    """
    PER_epsilon = 0.001
    PER_alpha = 0
    PER_beta = 0
    PER_beta_max = 0
    PER_beta_steps = 1
    """


class ApeX_config:
    def __init__(self):
        # game
        self.state_dim = CartPole.state_dim
        self.action_dim = CartPole.action_dim
        
        self.save_path = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.load_path=None
        #self.load_path = r'checkpoints\2022-09-29-04-20-27\4000000.chkpt'
        
        # Learner
        self.steps = 20000
        self.learn_on_gpu = True
        self.batch_size = 32
        self.memory_max = 1e5
        self.save_every = self.steps//4   # チェックポイントをセーブするステップ間隔
        self.epsilon_min = 0.01
        self.epsilon_max = 0.5
        self.gamma = 0.98            # Q学習の割引率
        self.lr = 0.00025            # optimizerの学習率
        self.burnin = self.batch_size * 100   # トレーニングを始める前に行うステップ数
        self.learn_every = 5         # Q_onlineを更新するステップ間隔
        self.sync_every = 1e4        # Q_targetにQ_onlineを同期するステップ間隔
        # PER
        self.PER_epsilon = 0.001     # 重さが0になることを防ぐ微小量
        self.PER_alpha = 0.6         # 0~1 0のとき完全なランダムサンプリング
        # 重要度サンプリング
        self.PER_beta = 0.4          # 補正の強さ 固定値
        
        # Actor
        self.actors = 3
        self.play_on_gpu = False
        self.sync_net_every = 400    # モデルをLearnerと同期するステップ間隔
        # multi step learning
        self.multi_steps = 3


class Game:
    def __init__(self,seed):
        self.env = CartPole()
    
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

