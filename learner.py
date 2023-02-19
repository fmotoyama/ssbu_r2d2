# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 14:29:32 2022

@author: f.motoyama
"""
import time, copy
import numpy as np
from collections import defaultdict

import torch

from sumtree import SumTree
from model import LSTMDuelingNetwork


class Learner:
    def __init__(self, config, queue_memory, queue_log, stop_flag):
        self.config = config
        self.queue_memory = queue_memory
        self.queue_log = queue_log
        self.stop_flag = stop_flag
        
        self.device = torch.device('cuda') if config.learn_on_gpu else torch.device('cpu')
        assert torch.cuda.is_available() or not config.learn_on_gpu, "Learner can't use gpu"
        
        self.memory = ReplayMemory(config, self.device)
        
        # LSTM + Dueling Network
        self.online_net = LSTMDuelingNetwork(config).float().to(self.device)
        self.target_net = LSTMDuelingNetwork(config).float().to(self.device)
        for p in self.target_net.parameters():
            p.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        # checkpointの呼び出し
        if config.load_file:
            self.load(config.load_file)
        
        self.curr_step = 0
        
        
    def run(self):
        print('Learner: run', flush=True)
        self.burnin()
        print('Learner: burnin end', flush=True)
        losses = []
        qs = []
        timedeltas = []
        while True:
            self.curr_step += 1
            t = time.time()
            # queue_memoryにたまってる遷移情報を保存
            self.recv_transitions()
            # 最新のネットワークをActorに渡す
            if self.curr_step % self.config.send_param_every == 0:
                self.save_param()
            # onlineをtargetへ同期
            if self.curr_step % self.config.sync_every == 0:
                self.sync_Q_target()
            # ネットワークを保存
            if self.curr_step % self.config.save_every == 0:
                old_files = list(self.config.save_path.glob('**/*.chkpt'))
                if old_files:
                    old_files[0].unlink()
                self.save()
            
            # memoryからサンプリング
            batch, indices, weights = self.memory.get_batch(self.config.batch_size)
            # Get TD Estimate and TD Target
            td_est, td_tgt = self.get_estimate_and_target(batch)
            # Q_onlineで逆伝搬
            loss = self.update_Q_online(td_est, td_tgt, weights)
            # メモリの優先度の更新
            self.memory.update(indices, (td_tgt - td_est).tolist())
            # ロギング
            losses.append(loss)
            qs.append(td_est.mean().item())
            timedeltas.append(time.time()-t)
            if self.curr_step % 100 == 0:
                self.logging(losses, qs, timedeltas)
                losses = []
                qs = []
                timedeltas = []
            
            # 終了処理
            if self.config.steps <= self.curr_step or self.stop_flag.is_set():
                if self.config.steps <= self.curr_step:
                    self.save(True)
                break
    
    
    def burnin(self):
        # memoryがある程度たまるまで学習しない 60秒かかるならタイムアウト
        for _ in range(600):
            self.save_param()
            self.recv_transitions()
            if self.config.burnin <= self.memory.count_add:
                break
            if self.stop_flag.is_set():
                raise Exception('burnin cancelled')
            #print(self.memory.count_add, flush=True)
            time.sleep(0.1)
        else:
            raise Exception('Timeout')
        
    def recv_transitions(self):
        # transitionsをmemoryに保存
        for _ in range(self.queue_memory.qsize()):
            t = time.time()
            transitions = self.queue_memory.get()
            t = time.time() - t
            if 1.0 < t:
                print(f"Learner: waited {round(t, 2)} sec to get transition", flush=True)
            self.memory.push(transitions)
    
    
    def save_param(self):
        t = time.time()
        file_new = self.config.save_path / 'param_new.pt'
        file = self.config.save_path / 'param.pt'
        torch.save(self.online_net.state_dict(), file_new)
        while True:
            try:
                if file.exists():
                    file.unlink()
            except PermissionError:
                if 3 < time.time() - t:
                    raise Exception('Timeout')
                continue
            break
        file_new.rename(file)
    
    
    def get_estimate_and_target(self, batch):
        states = batch['states'][:,:-1,:]           # (N, burnin + input_length, obs_shape)
        next_states = batch['states'][:,1:,:]
        prev_actions = batch['actions'][:,:-1]      # (N, burnin + input_length)
        prev_next_actions = batch['actions'][:,1:]
        prev_rewards = batch['rewards'][:,:-1]      # (N, burnin + input_length)
        prev_next_rewards = batch['rewards'][:,1:]
        assert states.size()[1] == self.config.r2d2_burnin + self.config.input_length
        
        # burnin
        with torch.no_grad():
            _, online_lstm_states = self.online_net(
                states[:,:self.config.r2d2_burnin,:],
                batch['lstm_states'],
                prev_actions[:,:self.config.r2d2_burnin],
                prev_rewards[:,:self.config.r2d2_burnin]
                )
        
        # td_est
        current_Q = self.online_net(
            states[:,self.config.r2d2_burnin:,:],
            online_lstm_states,
            prev_actions[:,self.config.r2d2_burnin:],
            prev_rewards[:,self.config.r2d2_burnin:]
            )[0][np.arange(0, self.config.batch_size), batch['actions'][:,-1]]
        
        # td_tgt
        with torch.no_grad():
            # onlineネットワークを用いて、合法手から最善手を選択
            next_state_Qs, _ = self.online_net(
                next_states, batch['next_lstm_states'], prev_next_actions, prev_next_rewards
                )
            best_action = torch.argmax(next_state_Qs, axis=1)
            # 最善手のQ値をtargetネットワークから取得
            next_Q = self.target_net(
                next_states, batch['next_lstm_states'], prev_next_actions, prev_next_rewards
                )[0][np.arange(0, self.config.batch_size), best_action]
            
            next_Q = self.rescaling_inv(next_Q)
            target_Q = batch['rewards'][:,-1] + (1-batch['dones'])*next_Q*self.config.gamma
            target_Q = self.rescaling(target_Q).float()
            #target_Q = (batch['rewards'] + (1-batch['dones'])*next_Q*self.config.gamma).float()
        
        return current_Q, target_Q
    
    
    
    def td_estimate(self, states, actions):
        current_Q = self.online_net(states)[np.arange(0, self.config.batch_size), actions]
        return current_Q

    @torch.no_grad()
    def td_target(self, next_states, rewards, dones):
        # onlineネットワークを用いて、合法手から最善手を選択
        next_state_Q = self.online_net(next_states)
        best_action = torch.argmax(next_state_Q, axis=1)
        # 最善手のQ値をtargetネットワークから取得
        next_Q = self.target_net(next_states)[np.arange(0, self.config.batch_size), best_action]
        
        #next_Q = self.rescaling_inv(next_Q)
        #target_Q = rewards + (1-dones.float())*next_Q*self.config.gamma
        #return self.rescaling(target_Q).float()
        return (rewards + (1-dones.float())*next_Q*self.config.gamma).float()
    

    def update_Q_online(self, td_est, td_tgt, weights):
        """onlineのパラメータを更新"""
        loss = self.loss_fn(td_est * weights, td_tgt * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    
    def sync_Q_target(self):
        """onlineパラメータをtargetパラメータにコピー"""
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    
    def save(self, sign=False):
        """チェックポイントをセーブ"""
        #if sign:
        if True:
            if self.config.learn_on_gpu:
                self.online_net.to('cpu')
            save_file = self.config.save_path / f'{self.curr_step}.chkpt'
            torch.save(
                dict(
                    online_model=self.online_net.state_dict(),
                    target_model=self.target_net.state_dict(),
                    memory=self.memory
                    ),
                save_file,
            )
            if self.config.learn_on_gpu:
                self.online_net.to(self.device)
            print(f"Network saved to {save_file} at step {self.curr_step}", flush=True)
        else:
            if self.config.learn_on_gpu:
                self.online_net.to('cpu')
            save_file = self.config.save_path / f'{self.curr_step}.chkpt'
            torch.save(
                dict(
                    online_model=self.online_net.state_dict(),
                    target_model=self.target_net.state_dict(),
                    #memory=self.memory
                    ),
                save_file,
            )
            if self.config.learn_on_gpu:
                self.online_net.to(self.device)
            print(f"Network saved to {save_file} at step {self.curr_step}", flush=True)
            
    
    
    def load(self, load_file):
        """セーブしたチェックポイントを呼び出す"""
        load = torch.load(load_file)
        self.online_net.load_state_dict(load['online_model'])
        self.target_net.load_state_dict(load['target_model'])
        self.memory = load['memory']
        print(f'load {load_file}', flush=True)
    
    def logging(self, losses, qs, timedeltas):
        t = time.time()
        self.queue_log.put(('learner',(losses, qs, timedeltas)))
        t = time.time() - t
        if 0.1 < t:
            print(f"Learner: waited {round(t, 2)} sec to put log", flush=True)
    
    @staticmethod
    def rescaling(x):
        epsilon = 0.001
        return torch.sign(x)*(torch.sqrt(torch.abs(x) + 1) - 1) + epsilon*x
    @staticmethod
    def rescaling_inv(x):
        epsilon = 0.001
        return torch.sign(x)*(((torch.sqrt(1 + 4*epsilon*(torch.abs(x) + 1 + epsilon)) - 1)/(2*epsilon))**2 - 1)






class ReplayMemory(SumTree):
    def __init__(self,config, device):
        super().__init__(config.memory_max)
        self.config = config
        self.device = device
        #self.sequence = [[] for _ in range(config.actors)]
        self.sequences = defaultdict(list)
        self.sequence_length = 1 + config.r2d2_burnin + config.input_length + 1
    
    
    def push(self, transitions):
        """
        transitionsを受け取り、sequenceを保存
        transitions = {
            'env_id',
            'lstm_state_hs', # (n,L=1,N=1,Hmid)
            'lstm_state_cs', # (n,L=1,N=1,Hmid)
            'states',        # (n+1,obs_shape)
            'actions',       # (n)
            'rewards',       # (n)
            'dones',         # (n)
            'priorities',    # (n)
            }
        sequences[env_id] = [
            [lstm_state_h, lstm_state_c, state, action, reward, done],  # = tran
            ...,
            [lstm_state_h, lstm_state_c, state, action, reward, done],
            [None, None, state, None, None, None]
            ]
        """
        env_id = transitions['env_id']
        
        # 時刻ごとのデータ(tran)に変形
        trans = list(map(list, zip(
            torch.tensor(transitions['lstm_state_hs'], device=self.device),
            torch.tensor(transitions['lstm_state_cs'], device=self.device),
            torch.tensor(np.array(transitions['states'][:-1]), device=self.device),
            torch.tensor(transitions['actions'], device=self.device),
            torch.tensor(transitions['rewards'], device=self.device),
            torch.tensor(transitions['dones'], device=self.device),
            )))
        # 最後の時刻のデータはstateしか与えられていないが問題ない
        trans.append([None, None, torch.tensor(transitions['states'][-1], device=self.device), None, None, None])
        
        # 前回pushで削除されたものを補完
        self.sequences[env_id].append(trans[0])
        # sequenceにt+1のデータを挿入し、tの優先度を割り当ててsumtreeに保存
        for tran,priority in zip(trans[1:],transitions['priorities']):
            # episode開始直後のとき、sequenceをためる
            if len(self.sequences[env_id]) < self.sequence_length:
                self.sequences[env_id].append(tran)
                continue
            self.sequences[env_id].pop(0)
            self.sequences[env_id].append(tran)
            self.add(copy.copy(self.sequences[env_id]),priority)
        # stateしか与えられていない最後のtranをsequenceから削除
        self.sequences[env_id].pop(-1)
        
        if transitions['dones'][-1]:
            self.sequences[env_id] = []
    
    
    def get_batch(self,batch_size):
        # 重複を許して取得
        lstm_state_hs = []      # N * (L=1, N=1, Hmid), torch.tensor
        lstm_state_cs = []      # N * (L=1, N=1, Hmid), torch.tensor
        states = []             # N * (burnin + input_length + 1, obs_shape), torch.tensor
        next_lstm_state_hs = [] # N * (L=1, N=1, Hmid), torch.tensor
        next_lstm_state_cs = [] # N * (L=1, N=1, Hmid), torch.tensor
        actions = []            # N * (burnin + input_length + 1), torch.tensor
        rewards = []            # N * (burnin + input_length + 1), torch.tensor
        dones = []              # N, torch.tensor
        
        indices = []            # (N), list
        weights = []            # (N), torch.tensor
        for _ in range(batch_size):
            # サンプリング
            sequence,index,priority = self.get()
            lstm_state_hs.append(sequence[1][0])
            lstm_state_cs.append(sequence[1][1])
            states.append(torch.stack([tran[2] for tran in sequence[1:]]))
            next_lstm_state_hs.append(sequence[2][0])
            next_lstm_state_cs.append(sequence[2][1])
            actions.append(torch.stack([tran[3] for tran in sequence[:-1]]))
            rewards.append(torch.stack([tran[4] for tran in sequence[:-1]]))
            dones.append(sequence[-2][5])
            
            indices.append(index)
            # 重要度サンプリングに用いるweightの計算
            weight = (priority * min(self.count_add, self.l)) ** (-self.config.IS_beta)
            weights.append(weight)
        
        batch = {
            'lstm_states': (torch.cat(lstm_state_hs, dim=1), torch.cat(lstm_state_cs, dim=1)),
            'states': torch.stack(states),
            'next_lstm_states': (torch.cat(next_lstm_state_hs, dim=1), torch.cat(next_lstm_state_cs, dim=1)),
            'actions': torch.stack(actions),
            'rewards': torch.stack(rewards),
            'dones': torch.stack(dones),
        }
        # 安定性の理由から最大値で正規化
        weights = torch.tensor(weights, device=self.device) / max(weights)
        
        return batch,indices,weights
    
    
    def update(self,indices,td_errors):
        # 各indexの優先度を更新
        for index, td_error in zip(indices,td_errors):
            priority = (abs(td_error) + self.config.PER_epsilon) ** self.config.PER_alpha
            self.propagate(index, priority)



if __name__ == '__main__':
    # 完璧なテスト環境を作る
    from game.pole_config import R2D2config
    memory = ReplayMemory(R2D2config)















