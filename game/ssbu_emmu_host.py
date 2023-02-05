# -*- coding: utf-8 -*-

import socket, random, struct
import numpy as np

from .ssbu_env_emmu import Env_param

# ソケット通信(クライアント側)

ip = '192.168.11.18'
port = 50000

class MySocket:
    def __init__(self, soc):
        self.soc = soc
        
    def mysend(self, msg):
        l = len(msg)
        totalsent = 0
        while totalsent < l:
            sent = self.soc.send(msg[totalsent:])
            assert sent, "socket broken"
            totalsent = totalsent + sent
    
    def myrecv(self, l):
        self.soc.setblocking(True) # エラー回避： BlockingIOError: [WinError 10035] ブロック不可のソケット操作をすぐに完了できませんでした。
        chunks = []
        bytes_recd = 0
        while bytes_recd < l:
            chunk = self.soc.recv(min(l - bytes_recd, 2048))    #ホストが死ぬと空白を受け取る
            assert chunk, "socket broken"
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return b''.join(chunks)
    
    def mysend_plus(self,msg):
        # 長さ標識(2bytes)を含めて送信
        l = len(msg).to_bytes(2,'big')
        self.mysend(l)
        self.mysend(msg)
        
    def myrecv_plus(self):
        # 長さ標識(2bytes)を含めて受信
        recv = self.myrecv(2)
        l = int.from_bytes(recv,'big')
        recv = self.myrecv(l)
        return recv
    
    def close(self):
        self.soc.close()




class Env_host:
    def __init__(self,env_num):
        self.env_num = env_num
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(20)
        self.socket.connect((ip, port))
        self.mysocket = MySocket(self.socket)
        
        # serverから要求を受けて、env_numを返す
        recv = self.mysocket.myrecv_plus()
        assert recv.decode() == 'req_env_num'
        send = env_num.to_bytes(1, 'big')
        self.mysocket.mysend(send)
        print(f'host {env_num}: connected')
        
        env_param = Env_param()
        self.observation_shape = env_param.observation_shape
        self.action_shape = env_param.action_shape
        self.observation_datasize = np.prod(self.observation_shape) * 4
    
    # 1
    def step(self, action:int) -> (np.ndarray,float,bool):
        """
        send:
            operator: int8
            action: uint8
        recv:
            observation: np.float32 * (2,n)
            reward: float32
            done: uint8
            frame: uint32
        """
        operator = (1).to_bytes(1, 'big', signed=True)
        action = int(action).to_bytes(1, 'big', signed=False)
        send = operator + action
        
        self.mysocket.mysend(send)
        recv = self.mysocket.myrecv(self.observation_datasize+4+1+4)
        
        observation = np.reshape(
            np.frombuffer(recv[:self.observation_datasize], dtype='float32'),
            self.observation_shape
            )
        reward = struct.unpack('>f', recv[self.observation_datasize:-5])[0]
        done = recv[-5]
        frame = recv[-4:]
        assert done in [0,1]
        
        return observation, reward, done, {'frame': frame}

    # 2
    def legal_actions(self) -> list:
        """
        send:
            operator: int8
        recv:
            actions: uint16 * (n,)
        """
        send = (2).to_bytes(1, 'big', signed=True)
        self.mysocket.mysend(send)
        recv = self.mysocket.myrecv_plus()
        
        actions = [b for b in recv]
        return actions
    
    # 3
    def reset(self) -> np.float32:
        """
        send:
            operator: int8
        recv:
            observation: np.float32 * (2,n)
        """
        send = (3).to_bytes(1, 'big', signed=True)
        self.mysocket.mysend(send)
        recv = self.mysocket.myrecv(self.observation_datasize)
        
        observation = np.reshape(np.frombuffer(recv, dtype = 'f4'), self.observation_shape)
        return observation

    # 4
    def close(self):
        send = (4).to_bytes(1, 'big', signed=True)
        self.mysocket.mysend(send)
        recv = self.mysocket.myrecv_plus()
        assert recv.decode() == 'closed'
        self.socket.close()
        return
    
    
    def __del__(self):
        self.socket.close()





if __name__ == '__main__':
    env = Env_host(0)
    env.reset()
    reward2 = 0
    while True:
        legal_actions = env.legal_actions()
        #print(legal_actions)
        #action = input('action:') #0~118
        action = random.choice(legal_actions)
        #action = 0
        if action == -1:
            break
        #temp = time.time()
        observation,reward,done = env.step(action)
        #print(f'\r{time.time() - temp}(sec)', end = '')
        if reward2 < reward:
            print(f'\r{reward}', end = '')
            reward2 = reward
        if done:
            #observation = env.reset()
            print(f'reward:{reward}')
            break
    
    #通信終了
    env.close()


