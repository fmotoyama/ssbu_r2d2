# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:48:17 2022

@author: f.motoyama
"""
import traceback, shutil, time
import numpy as np
from torch import multiprocessing as mp
from torch.multiprocessing import Queue, Event

from mylogger import MetricLogger
from actor import Actor
from learner import Learner


import warnings
warnings.simplefilter('error')

# Windows,macOSでは"spawn"、Unixでは"fork"がデフォルト
if mp.get_start_method() == 'fork':
    mp.set_start_method('spawn', force=True)


# 子プロセス内のエラーログを取得する
#https://www.web-dev-qa-db-ja.com/ja/python/python-multiprocessing%EF%BC%9A%E8%A6%AA%E3%81%AE%E5%AD%90%E3%82%A8%E3%83%A9%E3%83%BC%E3%81%AE%E5%87%A6%E7%90%86/1043056036/
class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
 


def actor_work(**kwargs):
    stop_flag = kwargs['stop_flag']
    env = kwargs['env']
    env_id = env.env_id
    env.env.get_emmu_Handles()
    
    try:
        actor = Actor(**kwargs)
        actor.run()
        print(f'Actor {env_id}: closed', flush=True)
    except Exception as e:
        stop_flag.set()
        print(f'Actor {env_id}: errored', flush=True)
        raise e


def learner_work(**kwargs):
    stop_flag = kwargs['stop_flag']
    try:
        learner = Learner(**kwargs)
        learner.run()
        stop_flag.set()
        print('Learner: closed', flush=True)
    except Exception as e:
        stop_flag.set()
        print('Learner: errored', flush=True)
        raise e

    
def logger_work(**kwargs):
    stop_flag = kwargs['stop_flag']
    try:
        logger = MetricLogger(**kwargs)
        logger.run()
        print('Logger : closed', flush=True)
    except Exception as e:
        stop_flag.set()
        print('Logger : errored', flush=True)
        raise e



if __name__ == '__main__':
    #from game.pole_config import R2D2config, Game
    from game.ssbu_config import R2D2config, Game
    
    config = R2D2config()
    
    games = [Game('emmu_local',env_num) for env_num in range(config.actors_emmu_local)] \
            + [Game('emmu_remote',env_num) for env_num in range(config.actors_emmu_remote)] \
            + [Game('switch',env_num) for env_num in range(config.actors_switch)]
    
    
    # 記録用のディレクトリを作成
    config.save_path.mkdir(parents=True)
    
    # Actorが遷移情報を置き、Learnerが読む
    queue_memory = Queue(maxsize=6)
    # Actor,Learnerがログを置き、Loggerが読む
    queue_log = Queue(maxsize=(config.actors + 1) * 2)
    # 終了フラグ
    stop_flag = Event()
    
    

    # サブプロセスを作成
    epsilons = [epsilon for epsilon in np.linspace(config.epsilon_min, config.epsilon_max, config.actors)]
    actors = [
        Process(
            target=actor_work,
            kwargs={
                'config':config,
                'env':games[i],
                'epsilon':epsilons[i],
                'queue_memory':queue_memory,
                'queue_log':queue_log,
                'stop_flag':stop_flag
                }
            )
        for i in range(config.actors)
        ]
    
    learner = Process(
        target=learner_work,
        kwargs={
            'config':config,
            'queue_memory':queue_memory,
            'queue_log':queue_log,
            'stop_flag':stop_flag
            }
        )
    
    logger = Process(
        target=logger_work,
        kwargs={
            'config':config,
            'queue_log':queue_log,
            'stop_flag':stop_flag
            }
        )
    
    # 各サブプロセスを開始
    [actor.start() for actor in actors]
    learner.start()
    logger.start()
    
    
    # 終了フラグが立つかKeyboardInterruptされるまで待機
    try:
        while not stop_flag.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        stop_flag.set()
    # 終了フラグが立ったのち、全プロセスが終了するまでキューを空にし続ける
    while any([any([actor.is_alive() for actor in actors]), learner.is_alive(), logger.is_alive()]):
        for queue in [queue_memory, queue_log]:
            while not queue.empty():
                queue.get()
    
    
    [actor.join() for actor in actors]
    learner.join()
    logger.join()
    
    # エラー取得
    e_actors = [actor.exception for actor in actors]
    e_learner = learner.exception
    e_logger = logger.exception
    
    if any(e_actors) or e_learner or e_logger:
        #shutil.rmtree(config.save_path)
        print('[errored]')
    else:
        print('[closed successfully]')
    
    
    """
    print(stop_flag.is_set())
    #"""
























