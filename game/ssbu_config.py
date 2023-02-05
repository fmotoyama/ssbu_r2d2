import datetime
from pathlib import Path
import numpy as np

from .ssbu_emmu_host import Env_host
from .ssbu_env_emmu import Env, Env_param
    

class R2D2config:
    def __init__(self):
        # game
        self.name = 'ssbu_lv3'
        
        self.save_path = Path("checkpoints") / (f'{self.name}_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.load_file = None
        self.load_file = r'checkpoints\ssbu_lv3_2023-02-04-17-28-21\1000000.chkpt'
        
        # Network
        env_param = Env_param()
        self.state_size = np.prod(env_param.observation_shape)
        self.action_size = np.prod(env_param.action_shape)
        self.action_shape = env_param.action_shape
        #self.hidden_size = 32
        self.hidden_size = 256
        
        # Learner
        self.steps = 2000000
        #self.steps = 12
        self.learn_on_gpu = True
        self.batch_size = 64
        self.memory_max = 1e5
        self.save_every = self.steps//10    # チェックポイントをセーブする間隔
        self.epsilon_min = 0.01             # epsilon-greety
        self.epsilon_max = 0.4
        self.gamma = 0.997                  # Q学習の割引率
        #self.lr = 0.00025                   # optimizerの学習率
        self.lr = 0.00001                   # optimizerの学習率
        self.burnin = self.batch_size * 50  # トレーニングを始める前に行うステップ数
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
        self.actors_emmu_local = 2
        self.actors_emmu_remote = 0
        self.actors_switch = 0
        self.actors = self.actors_emmu_local + self.actors_emmu_remote + self.actors_switch
        self.play_on_gpu = False
        self.sync_param_every = 100         # モデルをLearnerと同期するステップ間隔
        self.send_size = 50                 # memoryに送る遷移情報のサイズ
        
        self.env_ids = [f'emmu_local{i}' for i in range(self.actors_emmu_local)]\
                         + [f'emmu_remote{i}' for i in range(self.actors_emmu_remote)]\
                         + [f'switch{i}' for i in range(self.actors_switch)]


class Game:
    def __init__(self, env_name, env_num):
        self.env_name = env_name
        self.env_num = env_num
        self.env_id = env_name + str(env_num)
        if env_name == 'emmu_local':
            self.env = Env(env_num)
        elif env_name == 'emmu_remote':
            self.env = Env_host(env_num)
        elif env_name == 'switch':
            self.env = 0
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.ravel(observation)     # observationを平坦化
        return observation, reward, int(done), info
    
    def reset(self):
        observation = np.ravel(self.env.reset())
        return observation
    
    def close(self):
        return self.env.close()


if __name__ == '__main__': 
    pass




















