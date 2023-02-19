# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:45:03 2022

@author: f.motoyama
"""

import time, csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class MetricLogger:
    def __init__(self, config, env_ids, queue_log, stop_flag):
        self.config = config
        self.env_ids = env_ids
        self.queue_log = queue_log
        self.stop_flag = stop_flag
        
        self.save_dir = config.save_path
        assert config.save_path.exists()
        self.window = 100   #移動平均の長さ
        
        # ログを残すデータ
        self.actor_label = ['Episode','MeanReward','Length','TimeDelta','Frame_mean']
        self.learner_label = ['Step','loss','q','TimeDelta']
                
        self.actor_log_dirs = {
            env_id: self.save_dir / f'actor_log_{env_id}.csv'
            for env_id in env_ids
            }
        for actor_log in self.actor_log_dirs.values():
            with open(actor_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.actor_label)
        self.learner_log = self.save_dir / 'learner_log.csv'
        with open(self.learner_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.learner_label)
        
        # グラフとして描画するデータ
        # Actor
        self.ep_reward = defaultdict(list)
        self.ep_length = defaultdict(list)
        self.moving_avg_ep_reward = defaultdict(list)
        self.moving_avg_ep_length = defaultdict(list)
        self.moving_avg_ep_reward_plot = self.save_dir / "reward_plot.jpg"
        self.moving_avg_ep_length_plot = self.save_dir / "length_plot.jpg"
        # Learner
        self.loss = []
        self.q = []
        self.moving_avg_loss = []
        self.moving_avg_q = []
        self.moving_avg_loss_plot = self.save_dir / "loss_plot.jpg"
        self.moving_avg_q_plot = self.save_dir / "q_plot.jpg"
        
        
        self.t = time.time()
        
        
    
    
    def run(self):
        print('Logger : run', flush=True)
        count = 0
        while True:
            # ロギング
            for _ in range(self.queue_log.qsize()):
                name,data = self.queue_log.get()
                self.logging(name,data)
                count += 1
            if count % 50 == 1:
                self.draw()
            # 終了判定
            if self.stop_flag.is_set():
                break
    
    def close(self):
        print('Logger : start closing Logger', flush=True)
        self.end[-1] = 1
        # 全Actor、Learnerが終了処理開始後のとき、queue_logを空にする
        while True:
            if all([1 <= flag for flag in self.end[:-1]]):
                while not self.queue_log.empty():
                    name,data = self.queue_log.get()
                    self.logging(name,data)
                break
        self.draw()
        self.end[-1] = 2
        print('Logger : closed Logger')
        
    
    def logging(self, name, data):
        """
        Actorのepisodeの(env_id,rewards,timedelt)
        Learnerのstepの(loss,q),
        を受け取る
        """
        if name == 'actor':
            env_id, ep_reward, ep_length, timedelta, ep_frame_mean = data
            self.ep_reward[env_id].append(ep_reward)
            self.ep_length[env_id].append(ep_length)
            self.moving_avg_ep_reward[env_id].append(np.mean(self.ep_reward[env_id][-self.window:]))
            self.moving_avg_ep_length[env_id].append(np.mean(self.ep_length[env_id][-self.window:]))
            # ログに書き込み
            with open(self.actor_log_dirs[env_id], 'a', newline='') as f:
                writer = csv.writer(f)
                #['Episode','MeanReward','Length','TimeDelta','Time','Frame_mean']
                data = []
                writer.writerow((
                    len(self.ep_reward[env_id]),
                    ep_reward,
                    ep_length,
                    timedelta,
                    ep_frame_mean
                    ))
                
        elif name == 'learner':
            losses, qs, timedeltas = data
            for loss, q, timedelta in zip(losses, qs, timedeltas):
                self.loss.append(loss)
                self.q.append(q)
                self.moving_avg_loss.append(np.mean(self.loss[-self.window:]))
                self.moving_avg_q.append(np.mean(self.q[-self.window:]))
                # ログに書き込み
                with open(self.learner_log, 'a', newline='') as f:
                    writer = csv.writer(f)
                    #['Step','loss','q','TimeDelta','Time']
                    data = []
                    writer.writerow((
                        len(self.loss),
                        loss,
                        q,
                        timedelta,
                        ))
    
    

    def draw(self):
        # 描画
        for label in ['ep_reward','ep_length']:
            for env_id in self.env_ids:
                plt.plot(
                    np.arange(len(getattr(self, f'moving_avg_{label}')[env_id])),
                    getattr(self, f'moving_avg_{label}')[env_id],
                    label=str(env_id)
                    )        
            plt.xlabel('episode'); plt.ylabel(label[3:])
            plt.legend()
            plt.savefig(getattr(self, f'moving_avg_{label}_plot'))
            plt.clf()
        for label in ['loss','q']:
            plt.plot(getattr(self, f'moving_avg_{label}'))
            plt.xlabel('step'); plt.ylabel(label)
            plt.savefig(getattr(self, f'moving_avg_{label}_plot'))
            plt.clf()
    


















