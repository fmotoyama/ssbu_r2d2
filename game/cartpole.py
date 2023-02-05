# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 09:38:23 2022

@author: motoyama
"""

import math
import cv2
import numpy as np


class CartPole:
    state_dim = 4
    action_dim = 2
    
    def __init__(self):
        self.fps = 60
        self.frame_max = 10 * self.fps     # 10sec
        
        self.g = 9.8
        self.width = 10  #画面の幅 m
        
        # 台車
        self.M = 10     # kg
        # 棒
        self.m = 0.2      # kg
        self.l = 0.2      # m
        
        
        self.reset()
    
    
    def reset(self):
        self.frame_counter = 0
        # 台車
        self.x = 0
        self.dx = 0
        # 棒
        self.theta = -5
        self.dtheta = 0
        # 台車位置・棒角度を記憶
        self.history = [(self.x,self.theta)]
        return self.get_observation()
    
    
    def get_observation(self):
        return np.array([self.x, self.dx, self.theta, self.dtheta], dtype='f4')
    
    
    def action2F(self,action):
        F = [-200, 200]
        return F[action]
    
    
    def step(self, action):
        self.frame_counter += 1
        
        F = self.action2F(action)
        #F = 0

        # 物理演算
        s = math.sin(math.radians(self.theta))
        c = math.cos(math.radians(self.theta))
        g,M,m,l,dtheta = self.g,self.M,self.m,self.l,self.dtheta
        ddx = (F - m*l*s*dtheta**2 - m*g*s*c) / (M + m*s**2)
        ddtheta = (g*s/c - (F - m*l*s*dtheta**2)/(M + m)) / (l/c - m*l*c/(M + m))
        
        # 台車
        self.dx += ddx/self.fps
        self.x += self.dx/self.fps
        # 棒
        self.dtheta += ddtheta/self.fps
        self.theta += self.dtheta/self.fps
        
        self.history.append((self.x, self.theta))
        
        # 45度倒れる　or 場外で終了
        if 45 < abs(self.theta) or self.width/2 < abs(self.x):
            return(self.get_observation(), -self.frame_max, 1)
        # 規定回数まで耐えたら報酬
        if self.frame_max <= self.frame_counter:
            return(self.get_observation(), 1, 1)
        
        return(self.get_observation(), 1, 0)
    
    
    def render(self, outpath='animation.mp4'):
        # historyの内容を描画する
        # 描画の長さは、メートルの数値を10倍した数値を用いる
        fmt = cv2.VideoWriter_fourcc(*'mp4v')
        
        l = self.l * 500
        width = self.width * 100
        height = 300
        framesize = (width,height)
        y_horizon = height*2//3
        
        # 台車
        w, h = (40,20)    # 幅,高さの半分
        
        writer = cv2.VideoWriter(outpath, fmt, self.fps, framesize)     # ライター作成
        canvas =  np.zeros((height, width, 3), dtype="uint8")
        
        for x, theta in self.history:
            x = x*100 + width/2
            # 背景
            cv2.rectangle(canvas, (0,0), (width,height), (255, 255, 255), -1)
            # 水平線
            cv2.line(canvas, (0, y_horizon), (width, y_horizon), (0,0,0))
            # 台車
            pt1 = np.array([x - w, y_horizon - h], dtype = 'i4')
            pt2 = np.array([x + w, y_horizon + h], dtype = 'i4')
            cv2.rectangle(canvas, pt1, pt2, (0, 0, 0))
            # 棒
            pt1 = np.array([x, y_horizon - h], dtype='i4')
            pt2 = np.array([
                x + l*math.sin(math.radians(theta)),
                y_horizon - h - l*math.cos(math.radians(theta))
                ], dtype='i4')
            cv2.line(canvas,pt1,pt2,(0,0,0))
            cv2.circle(canvas, pt2, 5, (0,0,0), thickness=-1)
            # 描画
            writer.write(canvas)
            cv2.imshow("Canvas", canvas)
            # 5ms待機　Escを検知したらリセット
            key = cv2.waitKey(5)
            if key == 27:
                break
    
        writer.release()




if __name__ == "__main__":
    c = CartPole()
    
    c.reset()
    done = 0
    while not done:
        observation,reward,done = c.step(0)
        observation,reward,done = c.step(0)
    c.render()
    history = c.history
    