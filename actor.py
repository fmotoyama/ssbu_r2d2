# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 14:34:54 2022

@author: f.motoyama
"""
import time, copy
import numpy as np

import torch
import queue        #queueu.Emptyをキャッチする用

from model import LSTMDuelingNetwork

class Actor:
    def __init__(self, config, env, epsilon, memory, queue_log, stop_flag):
        self.config = config
        self.env = env
        self.env_id = env.env_id
        self.epsilon = epsilon
        self.memory = memory
        self.queue_log = queue_log
        self.stop_flag = stop_flag
        
        self.device = 'cuda' if config.play_on_gpu else 'cpu'
        assert torch.cuda.is_available() or not config.play_on_gpu, f"Actor {self.env_id} can't use gpu"
        
        self.online_net = LSTMDuelingNetwork(config).to(device=self.device).float()
        self.target_net = LSTMDuelingNetwork(config).to(device=self.device).float()
        self.online_net.eval() # training_modeがデフォなので
        self.target_net.eval()
    
    
    @torch.no_grad()
    def run(self):
        print(f'Actor {self.env_id}: run', flush=True)
        self.load_param()
        curr_step = 0
        local_memory = {
            "lstm_state_hs": [],
            "lstm_state_cs": [],
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "best_actions": [],
            "q_currents": [],
            "q_targets": [],   # 未来のact()で求められるargmax(online_net(state(t+1)))を用いるため、データの追加が他より遅れる
            }
        sequence_length = 1 + self.config.r2d2_burnin + self.config.input_length + 1
        while True:
            # episode開始
            lstm_state = (
                torch.zeros(1, 1, self.config.hidden_size, device=self.device),
                torch.zeros(1, 1, self.config.hidden_size, device=self.device)
                )
            local_memory["lstm_state_hs"].append(lstm_state[0])
            local_memory["lstm_state_cs"].append(lstm_state[1])
            state = self.env.reset()
            episode_rewards = []
            episode_frames = []
            t = time.time()
            while True:
                curr_step += 1
                # アクションを決定
                prev_action = local_memory["actions"][-1] if local_memory["actions"] else 0
                prev_reward = local_memory["rewards"][-1] if local_memory["rewards"] else 0
                action, q_current, lstm_state, best_action = self.act(state, lstm_state, prev_action, prev_reward)
                # アクションを実行
                next_state, reward, done, info = self.env.step(action)
                # transitionsを蓄積
                local_memory['states'].append(state)
                local_memory['actions'].append(action)
                local_memory['rewards'].append(reward)
                local_memory['dones'].append(done)
                local_memory['best_actions'].append(best_action)
                local_memory['q_currents'].append(q_current)
                if 2 < len(local_memory['rewards']):
                    local_memory['q_targets'].append(
                        self.q_target(
                            state,
                            best_action,
                            local_memory['rewards'][-2],
                            local_memory['dones'][-2]
                            )
                        )
                episode_rewards.append(reward)
                episode_frames.append(info['frame'])
                
                if sequence_length + 1 < len(local_memory['states']):
                    self.memorry.add
                
                
                
                
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
                    lstm_state_hs = [lstm_state_h.numpy() for lstm_state_h in lstm_state_hs]
                    lstm_state_cs = [lstm_state_c.numpy() for lstm_state_c in lstm_state_cs]
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
        
    
        
    
    def act(self, state, lstm_state, prev_action, prev_reward):
        # state: (state_size) → (N=1, L=1, state_size)
        state = torch.tensor(np.array([[state]]), device=self.device)
        # prev_action, prev_reward: scalar → (N=1,L=1)
        prev_action = torch.tensor([[prev_action]], device=self.device)
        prev_reward = torch.tensor([[prev_reward]], device=self.device)
        
        action_values, lstm_state = self.online_net(state, lstm_state, prev_action, prev_reward)
        q_current = torch.max(action_values).item()
        best_action = torch.argmax(action_values, axis=1).item()
        if np.random.rand() < self.epsilon:
            # ランダムアクション
            action = np.random.randint(self.config.action_size)
        else:
            # ネットワークが出力するQ値の分布から最適なアクションを選択
            action = best_action
        return action, q_current, lstm_state, best_action
    
    
    def load_param(self):
        #https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
        load_file = self.config.save_path / 'param.pt'
        t = time.time()
        while True:
            try:
                if self.config.learn_on_gpu:
                    if self.config.play_on_gpu:
                        # gpu → gpu
                        self.online_net.load_state_dict(torch.load(load_file))
                    else:
                        # gpu → cpu
                        self.online_net.load_state_dict(torch.load(load_file, map_location='cpu'))
                        #self.online_net.load_state_dict(torch.load(load_file, map_location={'cuda:0': 'cpu'}))#!!!
                else:
                    if self.config.play_on_gpu:
                        # cpu → gpu
                        self.online_net.load_state_dict(torch.load(load_file, map_location='cuda'))#直後にmodel.to(device)が要るかが怪しい
                    else:
                        # cpu → cpu
                        self.online_net.load_state_dict(torch.load(load_file))
                
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
    
    

    def q_target(self,state,best_action,reward,done):
        state = torch.tensor(state, device=self.device).view(1,1,-1)
        next_q = self.target_net(state)[0][0,best_action]
        next_q = self.rescaling_inv(next_q)
        target_q = reward + (1-done)*next_q*self.config.gamma
        target_q = self.rescaling(target_q).float()
    @staticmethod
    def rescaling(x):
        epsilon = 0.001
        return torch.sign(x)*(torch.sqrt(torch.abs(x) + 1) - 1) + epsilon*x
    @staticmethod
    def rescaling_inv(x):
        epsilon = 0.001
        return torch.sign(x)*(((torch.sqrt(1 + 4*epsilon*(torch.abs(x) + 1 + epsilon)) - 1)/(2*epsilon))**2 - 1)

    
    
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
            self.online_net(next_state, next_lstm_state, action, reward)[0]
            ).item()
        td_tgt = np.array(rewards) + (1 - np.array(dones)) * next_Q * self.config.gamma
        
        priorities = np.abs(td_tgt-td_est)
        return priorities.numpy()
        
    
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

















