# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:17:31 2022

@author: f.motoyama
"""
import random
from time import time
from pathlib import Path
import numpy as np

import torch
from torch import multiprocessing as mp

from model import LSTMDuelingNetwork

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

class R2D2config:
    def __init__(self):
        self.save_path = Path("checkpoints") / 'test' 
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)        
        # Network
        self.state_size = 7+12
        self.action_size = 17
        self.action_shape = (6,9,2)
        #self.hidden_size = 32
        self.hidden_size = 256


def test1():
    # gpu → cpu
    net1 = LSTMDuelingNetwork(config).float().to('cuda')
    net2 = LSTMDuelingNetwork(config).float().to('cpu')
    for _ in range(100):
        torch.save(net1.state_dict(), save_file)
        net2.load_state_dict(torch.load(save_file, map_location='cpu'))


def test2():
    # cpu → gpu
    net1 = LSTMDuelingNetwork(config).float().to('cpu')
    net2 = LSTMDuelingNetwork(config).float().to('cuda')
    for _ in range(100):
        torch.save(net1.state_dict(), save_file)
        net2.load_state_dict(torch.load(save_file, map_location='cuda'))


config = R2D2config()
save_file = config.save_path / 'temp'
net1 = LSTMDuelingNetwork(config).float().to('cuda')
net2 = LSTMDuelingNetwork(config).float().to('cpu')
net3 = LSTMDuelingNetwork(config).float().to('cpu')
torch.save(net1.state_dict(), save_file)
def work1(n):
    for _ in range(1000):
        torch.save(net1.state_dict(), save_file)
def work2(n):
    for _ in range(1000):
        net2.load_state_dict(torch.load(save_file, map_location='cpu'))
def work3(n):
    for _ in range(1000):
        net3.load_state_dict(torch.load(save_file, map_location='cpu'))
def test3():
    # マルチプロセスでgpu → cpu
    p1 = mp.Process(target=work1,args=(0,))
    p2 = mp.Process(target=work2,args=(0,))
    p3 = mp.Process(target=work3,args=(0,))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()


def test4():
    # gpu → gpu
    net1 = LSTMDuelingNetwork(config).float().to('cuda')
    net2 = LSTMDuelingNetwork(config).float().to('cpu')
    net3 = LSTMDuelingNetwork(config).float().to('cuda')
    net4 = LSTMDuelingNetwork(config).float().to('cuda')
    
    torch.save(net1.state_dict(), save_file)
    net2.load_state_dict(torch.load(save_file, map_location='cpu'))
    net2.to('cuda')
    net3.load_state_dict(torch.load(save_file, map_location='cpu'))
    net4.load_state_dict(torch.load(save_file, map_location='cpu'))





if __name__ == '__main__':
    config = R2D2config()
    
    save_file = config.save_path / 'temp'
     
    t1 = time()
    test3()   
    t1 = time() - t1












