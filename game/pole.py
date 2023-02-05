# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 09:38:23 2022

@author: motoyama
"""

import math
#import cv2
import numpy as np


class Pole:
    observation_shape = (2,)
    action_shape = (3,)
    
    def __init__(self):
        self.fps = 60
        self.frame_max = 20 * self.fps
        
        self.g = 9.8
        self.m = 20
        self.l = 100
        
        self.reset()
    
    
    def reset(self):
        self.frame_counter = 0
        self.theta = 0             # 時計回りの角度
        self.dtheta = 0
        self.theta_history = [self.theta]
        return self.get_observation()
    
    def get_observation(self):
        return np.array([self.theta, self.dtheta], dtype='f4')
    
    def get_reward(self):
        return -math.cos(math.radians(self.theta))
    
    def action2F(self,action):
        F = [-1, 0, 1]
        return F[action]
    
    def step(self, action):
        # 棒の角度の計算
        F = self.action2F(action)
        ddtheta = F/(self.m) - self.g*math.sin(math.radians(self.theta))/self.l
        self.dtheta += ddtheta# / self.fps*20
        self.theta += self.dtheta# / self.fps*20
        
        self.theta_history.append(self.theta)
        self.frame_counter += 1
        done = 1 if self.frame_max <= self.frame_counter else 0
        
        return(self.get_observation(), self.get_reward(), done)
    
    
    def render(self, outpath='animation.mp4'):
        # theta_historyを動画として保存
        fmt = cv2.VideoWriter_fourcc(*'mp4v')
        framesize = (500, 500)
        width, height = framesize
        
        writer = cv2.VideoWriter(outpath, fmt, self.fps, framesize)     # ライター作成
        canvas =  np.zeros((height, width, 3), dtype="uint8")
        
        for theta in self.theta_history:
            # 背景
            cv2.rectangle(canvas, (0,0), (width,height), (255, 255, 255), -1)
            # 水平線
            cv2.line(canvas, (0, height//2), (width, height//2), (192,192,192))
            cv2.line(canvas, (width//2, 0), (width//2, height), (192,192,192))
            # 棒
            pt1 = np.array([width//2, height//2], dtype='i4')
            pt2 = np.array([
                width//2 - self.l*math.sin(math.radians(theta)),
                height//2 + self.l*math.cos(math.radians(theta))
                ], dtype='i4')
            cv2.line(canvas,pt1,pt2,(0,0,0))
            cv2.circle(canvas, pt2, 5, (0,0,0), thickness=-1)
            
            # 描画
            writer.write(canvas)
            cv2.imshow("Canvas", canvas)
            
            # 5ms待機　Escを検知したらリセット
            key = cv2.waitKey(1)
            if key == 27:
                break
        
        writer.release()
    
    def close(self):
        pass
        
        





if __name__ == "__main__":
    p = Pole()
    dx = []
    
    p.reset()
    done = 0
    while not done:
        observation,reward,done = p.step(0)
        dx.append(p.dtheta)
    
    p.render()

    a = np.array(dx)
