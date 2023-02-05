# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:08:22 2022

@author: f.motoyama
"""
import numpy as np

class Env:
    observation_space = (2,8,8)
    action_space = 65
    def __init__(self,me):
        # tableでは白/黒/空白を0/1/2で表す
        self.table = np.empty((8,8), dtype = 'i1')
        assert me in [0,1], f'Env.__init__:me={me}'
        self.me = me            # 基準とするプレイヤー
        self.op = 1-self.me
    
    
    def get_observation(self):
        # obs[0/1]  : 自分の駒位置/相手の駒位置
        obs = np.empty((2,8,8), dtype = 'f4')
        obs[0] = self.table == self.me
        obs[1] = self.table == self.op
        return obs
    
    
    def reset(self):
        self.table = np.full((8,8), 2, dtype = 'i1')
        self.table[np.ix_([3,4],[3,4])] = [[0,1],[1,0]]
        return self.get_observation()
    
    
    def step(self, action, me=None):
        if me != None:
            op = 1-me
        else:
            me = self.me
            op = self.op
        
        #違法手を検知
        if action not in self._legal_actions(me):
            print('illegal action')
            return self.get_observation(), -1, 1
        
        if action != 64:
            row = action//8
            col = action%8
            pos_tgt = np.array([row,col], dtype = 'i1')     #石を置く場所
            
            vectors = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]], dtype = 'i1')
            #vector方向で裏返す
            for vector in vectors:
                pos = pos_tgt + vector
                while np.all(0<=pos) and np.all(pos<8):
                    if self.table[pos[0],pos[1]] == 2:
                        break
                    if self.table[pos[0],pos[1]] == me:
                        #逆走して[row,col]までをmeにする
                        while ~np.all(pos==pos_tgt):
                            pos -= vector
                            self.table[pos[0],pos[1]] = me
                        break
                    pos += vector
        
        #勝敗判定
        if action == 64 and self._legal_actions(op) == [64]:
            #今回自分が置けておらず、相手も置く場所がない場合
            done = 1
            score_me = np.sum(self.table==self.me)
            score_op = np.sum(self.table==self.op)
            if score_me < score_op:
                reward = -1
            elif score_me == score_op:
                reward = 0
            else:
                reward = 1
        else:
            done = 0
            reward = 0
        
        #return self.table, reward, done
        return self.get_observation(), reward, done
    
    
    def _legal_actions(self, me):
        """self.tableを用いてme側の合法手を求める"""
        op = 1-me
        
        temp = np.where(self.table == op)       #相手の石のインデックス
        actions = []
        vectors = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]], dtype = 'i1')
        for row,col in zip(temp[0],temp[1]):
            #8方向のベクトルで置ける場所を探す
            for vector in vectors:
                pos_tgt = np.array([row,col], dtype = 'i1') + vector
                
                if ~(np.all(0<=pos_tgt) and np.all(pos_tgt<8)):
                    continue
                if self.table[pos_tgt[0],pos_tgt[1]] != 2:
                    continue
                
                pos = pos_tgt - vector * 2
                while np.all(0<=pos) and np.all(pos<8):
                    if self.table[pos[0],pos[1]] == 2:
                        break
                    if self.table[pos[0],pos[1]] == me:
                        actions.append(pos_tgt[0]*8 + pos_tgt[1])
                        break
                    #vectorの逆に進む
                    pos -= vector
        
        if actions == []:
            actions = [64]  #パス
        return actions
    
    
    @staticmethod
    def legal_actions(observation, me_in_observation=0):
        """observationを用いてme側の合法手を求める"""
        me = me_in_observation
        op = 1-me
        
        table = np.full((8,8), 2, dtype = 'i1')
        table[np.where(observation[me])] = me
        table[np.where(observation[op])] = op
        temp = np.where(table == op)       #相手の石のインデックス
        actions = []
        vectors = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]], dtype = 'i1')
        for row,col in zip(temp[0],temp[1]):
            #8方向のベクトルで置ける場所を探す
            for vector in vectors:
                pos_tgt = np.array([row,col], dtype = 'i1') + vector
                
                if ~(np.all(0<=pos_tgt) and np.all(pos_tgt<8)):
                    continue
                if table[pos_tgt[0],pos_tgt[1]] != 2:
                    continue
                
                pos = pos_tgt - vector * 2
                while np.all(0<=pos) and np.all(pos<8):
                    if table[pos[0],pos[1]] == 2:
                        break
                    if table[pos[0],pos[1]] == me:
                        actions.append(pos_tgt[0]*8 + pos_tgt[1])
                        break
                    #vectorの逆に進む
                    pos -= vector
        
        if actions == []:
            actions = [64]  #パス
        return np.array(actions, dtype=np.int64)
    
    
    def render(self):
        for line in self.table:
            temp = ''.join([str(n) for n in line])
            print(temp.replace('0', '○').replace('1', '●').replace('2', '□'))




if __name__ == '__main__':
    import random
    
    me = 0
    e = Env(me)
    obs = e.reset()
    
    turn = 1
    done = 0
    while not done:
        actions1 = e._legal_actions(turn)
        actions2 = e.legal_actions(obs, turn ^ me)  # turn,meが同じとき0,違うとき1
        assert np.all(actions1==actions2)
        action = random.choice(actions1)
        obs, reward, done = e.step(action, turn)
        #e.render()
        turn = 1-turn
    e.render()
    print(reward)
    
    
    