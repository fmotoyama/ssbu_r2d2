# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:27:13 2022

@author: f.motoyama
"""
import torch
import numpy as np

from learner import LSTMDuelingNetwork as Network
from game.pole_config import Game, R2D2config2


class play_against_AI:
    def __init__(self, load_path, config):
        self.use_cuda = torch.cuda.is_available()
        
        # Network
        self.net = Network(config.state_dim, config.action_dim, config.hidden_size).float()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.net.to(device=self.device)
        load = torch.load(load_path)
        self.net.load_state_dict(load['model'])
    
    

    def act(self, state, lstm_state):
        # ネットワークが出力するQ値の分布から最適のアクションを選択
        state = state.__array__()   # new reference to self
        state = torch.tensor(state, device=self.device).view(1,1,-1)
        action_values,lstm_state = self.net(state, lstm_state)
        action_idx = torch.argmax(action_values, axis=1).item()
        return action_idx, lstm_state



if __name__ == '__main__':
    load_path = './checkpoints/Pole_2022-11-05-16-24-22/100000.chkpt'
    
    # 環境の初期化
    config = R2D2config2()
    e = Game(0)
    obs = e.reset()
    
    agent = play_against_AI(load_path, config)
    lstm_state = None
    action_list = []
    while True:
        # アクションを決定
        action,lstm_state = agent.act(obs,lstm_state)
        # アクションを実行
        obs_next, reward, done = e.step(action)
        
        action_list.append(action)
        if done:
            break
        obs = obs_next
    
    action_list = np.array(action_list)
    e.render()










