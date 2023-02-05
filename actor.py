# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 14:34:54 2022

@author: f.motoyama
"""
import time
import numpy as np

import torch
import queue        #queueu.Emptyをキャッチする用

from model import LSTMDuelingNetwork

class Actor:
    def __init__(self, config, env, epsilon, queue_memory, queue_log, stop_flag):
        self.config = config
        self.env = env
        self.env_id = env.env_id
        self.epsilon = epsilon
        self.queue_memory = queue_memory
        self.queue_log = queue_log
        self.stop_flag = stop_flag
        
        self.device = 'cuda' if config.play_on_gpu else 'cpu'
        assert torch.cuda.is_available() or not config.play_on_gpu, f"Actor {self.env_id} can't use gpu"
        
        self.net = LSTMDuelingNetwork(config).to(device=self.device).float()
        self.net.eval() # training_modeがデフォなので
    
    @torch.no_grad()
    def run(self):
        print(f'Actor {self.env_id}: run', flush=True)
        #self.recv_param()
        self.load_param()
        curr_step = 0
        states = []
        actions = [0]   #　ステップを開始する前の記録を1つ保持 action=0を無入力とみなしている
        rewards = [0]   #　ステップを開始する前の記録を1つ保持
        dones = []
        qs = []
        while True:
            # episode開始
            lstm_state = (
                torch.zeros(1, 1, self.config.hidden_size, device=self.device),
                torch.zeros(1, 1, self.config.hidden_size, device=self.device)
                )
            lstm_state_hs = [lstm_state[0]]
            lstm_state_cs = [lstm_state[1]]
            state = self.env.reset()
            episode_rewards = []
            episode_frames = []
            t = time.time()
            while True:
                curr_step += 1
                # アクションを決定
                action, q, lstm_state = self.act(state, lstm_state, actions[-1], rewards[-1])
                # アクションを実行
                next_state, reward, done, info = self.env.step(action)
                # transitionsを蓄積
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                qs.append(q)
                episode_rewards.append(reward)
                episode_frames.append(info['frame'])
                # transitionを送信
                #if curr_step % self.config.send_size == 0 or done:
                if done:
                    states.append(next_state)
                    priorities = self.calc_priorities(
                        qs,         # 1~t
                        lstm_state, # t+1
                        action,     # t
                        next_state, # t+1
                        rewards[1:],# 1~t
                        dones       # 1~t
                        )
                    lstm_state_hs = [lstm_state_h.tolist() for lstm_state_h in lstm_state_hs]
                    lstm_state_cs = [lstm_state_c.tolist() for lstm_state_c in lstm_state_cs]
                    transitions = {
                        'env_id': self.env_id,
                        'lstm_state_hs': lstm_state_hs, # (t,L=1,N=1,Hmid)
                        'lstm_state_cs': lstm_state_cs, # (t,L=1,N=1,Hmid)
                        'states': states,               # (t+1,state_size)
                        'actions': actions[1:],         # (t)
                        'rewards': rewards[1:],         # (t)
                        'dones': dones,                 # (t)
                        'priorities': priorities,       # (t)
                        }
                    
                    self.send_transition(transitions)
                    lstm_state_hs = []
                    lstm_state_cs = []
                    states = []
                    actions = [actions[-1]]
                    rewards = [rewards[-1]]
                    dones = []
                    qs = []
                
                lstm_state_hs.append(lstm_state[0])
                lstm_state_cs.append(lstm_state[1])
                state = next_state
                # Learnerのネットワークを同期
                if curr_step % self.config.sync_param_every == 0:
                    self.load_param()
                # episodeを終了
                if done or self.stop_flag.is_set():
                    # queue_memoryが満タンなら待機
                    #while self.queue_memory.full:
                    #    time.sleep(1)
                    break
            
            # episode単位の情報をlogに渡す
            self.logging(episode_rewards, episode_frames, time.time() - t)
            # 終了処理
            if self.stop_flag.is_set():
                break
        self.env.close()
    
        
    
    def act(self, state, lstm_state, prev_action, prev_reward):
        # state: (state_size) → (N=1, L=1, state_size)
        state = torch.tensor(np.array([[state]]), device=self.device)
        # prev_action, prev_reward: scalar → (N=1,L=1)
        prev_action = torch.tensor([[prev_action]], device=self.device)
        prev_reward = torch.tensor([[prev_reward]], device=self.device)
        
        action_values, lstm_state = self.net(state, lstm_state, prev_action, prev_reward)
        q = torch.max(action_values).item()
        if np.random.rand() < self.epsilon:
            # ランダムアクション
            action_idx = np.random.randint(self.config.action_size)
        else:
            # ネットワークが出力するQ値の分布から最適なアクションを選択
            action_idx = torch.argmax(action_values, axis=1).item()
        return action_idx, q, lstm_state
    
    def load_param(self):
        #https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
        load_file = self.config.save_path / 'param.pt'
        t = time.time()
        while True:
            try:
                if self.config.learn_on_gpu:
                    if self.config.play_on_gpu:
                        # gpu → gpu
                        self.net.load_state_dict(torch.load(load_file))
                    else:
                        # gpu → cpu
                        self.net.load_state_dict(torch.load(load_file, map_location='cpu'))
                        #self.net.load_state_dict(torch.load(load_file, map_location={'cuda:0': 'cpu'}))#!!!
                else:
                    if self.config.play_on_gpu:
                        # cpu → gpu
                        self.net.load_state_dict(torch.load(load_file, map_location='cuda'))#直後にmodel.to(device)が要るかが怪しい
                    else:
                        # cpu → cpu
                        self.net.load_state_dict(torch.load(load_file))
                
            except (FileNotFoundError,PermissionError):
                if 120 < time.time() - t:
                    raise Exception('Timeout')
                continue
            except EOFError: # 空ファイルを検知したとき
                print(f'Actor {self.env_id}: param is empty', flush=True)
                time.sleep(2)
                continue
            break
        t = time.time() - t
        if 1.0 < t:
            print(f"Actor {self.env_id}: waited {round(t, 2)} sec to load param", flush=True)
            
    
    # actorごとにqueue_memoryを用意
    def send_transition(self, transitions):
        # Learnerのmemoryに遷移情報を渡す
        try:
            t = time.time()
            self.queue_memory.put(transitions,timeout=60)
            t = time.time() - t
            if 0.3 < t:
                print(f"Actor {self.env_id}: waited {round(t, 2)} sec to put in memory", flush=True)
        except queue.Full as e:
            print(f"Actor {self.env_id}: timeout to put transition", flush=True)
            if not self.stop_flag.is_set():
                raise e
    
    
    def calc_priorities(self, qs, next_lstm_state, action, next_state, rewards, dones):
        # qs                : 1~t
        # next_lstm_state   : t+1
        # action            : t
        # next_state        : t+1
        # rewards           : 1~t
        # done              : 1~t
        
        # scalar → (N=1,L=1)
        action = torch.tensor([[action]], device=self.device)
        reward = torch.tensor([[rewards[-1]]], device=self.device)
        
        td_est = np.array(qs)
        
        next_Q = np.roll(td_est, -1)
        next_state = torch.tensor(next_state, device=self.device).view(1,1,-1)
        next_Q[-1] = torch.max(
            self.net(next_state, next_lstm_state, action, reward)[0]
            ).item()
        td_tgt = np.array(rewards) + (1 - np.array(dones)) * next_Q * self.config.gamma
        
        priorities = (np.abs(td_tgt-td_est) + self.config.PER_epsilon) ** self.config.PER_alpha
        return priorities.tolist()
        
    
    def logging(self,rewards,frames,timedelta):
        ep_length = len(rewards)
        ep_reward = sum(rewards)
        ep_frame_mean = sum([f2-f1 for f1,f2 in zip(frames[:-1],frames[1:])]) / (ep_length-1)
        try:
            t = time.time()
            self.queue_log.put(
                (
                    'actor',
                    (self.env_id, ep_reward, ep_length, timedelta, ep_frame_mean)),
                timeout=60
                )
            t = time.time() - t
            if 0.3 < t:
                print(f"Actor {self.env_id}: waited {round(t, 2)} sec to put log", flush=True)
        except queue.Full as e:
            print(f"Actor {self.env_id}: timeout to put log", flush=True)
            if not self.stop_flag.is_set():
                raise e
        
        





if __name__ == '__main__':
    #from multiprocessing import Process, Queue, Pipe
    from game.pole_config import R2D2_config, Game
    
    a = Actor(
        config=R2D2_config,
        num=0,
        env=Game,
        epsilon=0,
        queue_memory=None,
        queue_param=None,
        queue_log=None,
        end=None
        )

















