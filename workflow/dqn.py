import gym
from gym import error, spaces, utils
from gym.utils import seeding
from kmcsim.buildtools import make_fcc, write_latt
import kmc_env
import kmcsim
from kmcsim.sim import KMCModel
from kmcsim.sim import EventTree
from kmcsim.sim import RunSim
import os
import numpy as np
import collections
from kmc_env.envs.kmcsim_state_funcs import make_surface_proj,calc_roughness,get_state_reward,get_incremented_rates,gaussian
from kmc_env.envs.kmc_env import *
from matplotlib import pyplot as plt
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Reshape, Flatten, Add, Input
from keras.models import Model
from keras.layers import Concatenate
from IPython.display import clear_output
import time
import sys
from copy import deepcopy

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from kmcsim.buildtools import make_fcc, write_latt
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class DQNAgent:

    def __init__(self,action_size=3,gamma=0.5,epsilon=0.8,epsilon_min=0,epsilon_decay=0.992,
        box = [16, 32, 4],box_extension=32,target_roughness=0.98,episodes=150,wdir=r"C:\Users\ni1\Documents\RL\kmcsim\data\working",
        reward_type='gaussian',reward_multiplier=1000,reward_tolerance=2,rates_spread=0.1,rates_adjustment=1):
        
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon  
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model()
        self.batch_size = 1

        self.box=box
        self.box_extension=box_extension
        self.target_roughness=target_roughness
        self.episodes=episodes
        self.wdir=wdir
        self.env = KmcEnv(box=self.box,box_extension=self.box_extension,target_roughness=self.target_roughness,
             reward_type=reward_type,reward_multiplier=reward_multiplier,reward_tolerance=reward_tolerance,
             rates_spread=rates_spread,rates_adjustment=rates_adjustment,folder_with_params=self.wdir)

        self.state,self.reward = self.env.reset()
        self.state_size = self.env.state.shape
        self.action_size = (action_size,)

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, activation='tanh', kernel_size = (3,3), strides = (1,1), batch_input_shape=(1,32,32,1)))
        model.add(MaxPool2D((2,2)))
        model.add(Conv2D(64, activation='tanh', kernel_size = (3,3), strides = (1,1), batch_input_shape=(1,32,32,1)))
        model.add(MaxPool2D((2,2)))
        model.add(Conv2D(32, activation='tanh', kernel_size = (3,3), strides = (1,1)))
        model.add(MaxPool2D((2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(9, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = [np.random.randint(0, 3), np.random.randint(0, 3), np.random.randint(0, 3)]
            decision=0

        else:
            action = [np.argmax(self.model.predict(state[None,:,:,None])[0][:3]),
                      np.argmax(self.model.predict(state[None,:,:,None])[0][3:6]),
                      np.argmax(self.model.predict(state[None,:,:,None])[0][6:])]
            decision=1
   
        return action,decision 

    def replay(self, batch_size):
        state, action, reward_new, next_state, done = self.memory[-1]
        ns_s=next_state[None,:,:,None]-next_state[None,:,:,None].mean()
        target = (reward_new + self.gamma *
                      np.amax(self.model.predict(ns_s)))
        s_s=state[None,:,:,None]-state[None,:,:,None].mean()
        target_f = self.model.predict(s_s)[0]
        target_f[action] = target
        self.model.fit(s_s, target_f.reshape(-1, 9), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def start_session(self):
        self.state,self.reward = self.env.reset()       
    
    def run_train_session_once(self,verbose=False,plotit=False):
        self.state,self.reward = self.env.reset()
        self.done=self.env.end_flag
        count=0
        while not self.done:
            if verbose==True:
                if self.done:
                    print("RMS: {}, score: {}, e: {:.2}"
                        .format(self.rms_val, self.reward, self.epsilon))
                    break
                
            self.action, self.decision = self.act(self.state)
            self.next_state, self.reward, self.done = self.env.step(self.action, verbose=True)
            self.remember(self.state,self.action, self.reward, self.next_state, self.done)
            self.replay(self.batch_size)
            self.state = self.next_state
            self.rms_val = calc_roughness(self.state)
            self.thickness=np.mean(self.state)
            if plotit:
                clear_output(wait=True) 
                plt.title(count)
                plt.imshow(self.state,vmin=0,vmax=30,cmap='nipy_spectral')
                plt.colorbar()
                plt.show()
            if verbose:
                print("RMS: {:.4}, score: {:.4}, thickness: {:.4}, e: {:.4}, decision: {}, done: {}"
                        .format(self.rms_val, self.reward,self.thickness, self.epsilon, self.decision, self.done))
                count=count+1     
        
    def run_train_session(self,verbose=False,plotit=False):
        done = False
        for run in range(self.episodes):
            self.run_train_session_once(verbose=verbose,plotit=plotit)
            self.learning[run] = self.timestep
            
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
