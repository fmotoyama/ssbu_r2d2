# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:52:48 2022

@author: motoyama
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingNetwork(nn.Module):
    # Qを、状態価値関数V(s)とアドバンテージ関数A(s,a)に分ける
    # 状態sだけで決まる部分と、行動aしだいで決まる部分を分離する
    # https://ailog.site/2019/10/29/torch11/
    def __init__(self, n_in, n_out):
        super().__init__()
        n_mid = 64
        
        self.block = nn.Sequential(
            nn.Linear(n_in, n_mid),
            #nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True)
        )
        
        self.adv = nn.Sequential(
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, n_out)
        )
        self.val = nn.Sequential(
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, n_out)
        )
    
    def forward(self, x):
        h = self.block(x)
        adv = self.adv(h)
        val = self.val(h).expand(-1, adv.size(1))
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return output


class LSTMDuelingNetwork(nn.Module):
    # lstm_state = (1step前の出力(hidden), 1step前のセル状態(cell))
    # LSTMの入力を(L,N,input_size)　→ (N,L,input_size)に変更
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_in = config.state_size
        n_mid = config.hidden_size
        n_out = config.action_size
        self.n_mid = n_mid
        
        self.block = nn.Sequential(
            nn.Linear(n_in, n_mid),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True)
        )
        
        # input: n_mid + actionの各次元のサイズ-1の合計 + 1(reward)
        self.lstm = nn.LSTM(
            input_size = n_mid + (sum(config.action_shape)-len(config.action_shape)) + 1,
            hidden_size = n_mid,
            num_layers = 1,
            batch_first = True
            )
        
        self.adv = nn.Sequential(
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, n_out)
        )
        self.val = nn.Sequential(
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, 1)
        )
        

    def forward(self, x, lstm_state, prev_action, prev_reward):
        # (N,L,Hin) → (N,L,Hmid)
        h = self.block(x)
        
        """
        # prev_action: (N,L) → (N,L,action_onehot)
        prev_action_multi = F.one_hot(prev_action, num_classes=self.config.action_size)
        """
        # prev_actionをaction_shapeの次元に分割
        # (N,L) → [(N,L),(N,L),...]
        prev_action = prev_action.long().detach().clone()
        prev_action_multi = []
        base = self.config.action_size
        for l in self.config.action_shape:
            base = base // l
            prev_action_multi.append(prev_action // base)
            prev_action = prev_action % base
        # 各次元でワンホット化 長さはその次元のサイズ-1
        #　[(N,L),(N,L),...] → [(N,L,onehot1),(N,L,onehot2),...]
        prev_action_multi = [
            F.one_hot(mat, num_classes=l)[:,:,1:]
            for mat,l in zip(prev_action_multi,self.config.action_shape)
            ]
        # [(N,L,onehot1),(N,L,onehot2),...] → (N,L,onehot)
        prev_action_multi = torch.cat(prev_action_multi, 2)
        
        # prev_reward: (N,L) → (N,L,1)
        prev_reward = prev_reward.unsqueeze(-1)
        # h: (N,L,Hmid) → (N,L,Hmid+action_onehot+1)
        h = torch.cat((h, prev_action_multi, prev_reward), 2)
        
        # _: (N,L,Hmid), next_lstm_state: ((1,N,Hmid),(1,N,Hmid))
        _, next_lstm_state = self.lstm(h,lstm_state)
        # h: (N,Hmid)
        h = next_lstm_state[0].view(-1,self.n_mid)
        
        # adv: (N,Hout), val: (N,1).expand(-1, adv.size(1))
        adv = self.adv(h)
        val = self.val(h).expand(-1, adv.size(1))
        
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return output, next_lstm_state




if __name__ == '__main__':
    import torch, random
    import numpy as np
    
    def torch_fix_seed(seed=42):
        # Python random
        random.seed(seed)
        # Numpy
        np.random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
    torch_fix_seed()
    model1 = LSTMDuelingNetwork(2,5,32)
    #model2 = LSTMDuelingNetwork(2,5,32).to('cuda')
    
    # (N,L,H) = (8,4,2)
    data = torch.tensor([
        [0,1],
        [2,3],
        [4,5],
        [6,7],
        ])
    
    batch_size = 8
    batch = torch.stack([data for _ in range(batch_size)]).float()
    
    none_state = (torch.zeros(1, batch_size, 32),torch.zeros(1, batch_size, 32))
    
    a11,astate1 = model1(batch,None)#none_state)
    a12,astate1 = model1(batch,none_state)
    a2,_       = model1(batch,astate1)
    #a3,astate2 = model2(batch.to('cuda'),None)
    
    bstate = None
    for i in range(4):
        b1,bstate = model1(batch[:,i,:].unsqueeze(1),bstate)
    
    c,cstate = model1(torch.unsqueeze(batch[0], dim=0),None)
    
    a11 = a11.tolist()          # a11 == a12であり、none_stateが正しい
    a12 = a12.tolist()
    #a31 = a3.tolist()           # a31 == a32であり、lstmの状態はデバイスに依らない
    #a32 = a3.to('cpu').tolist()
    b1 = b1.tolist()            # a1,a2と大体一致
    
    













