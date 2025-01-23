import torch
import torch.nn as nn
import numpy as np
import random
import os
from collections import deque

import config
from env import Environment
from dqn_per_net import DQN_PER
from models import Encoder

import dataset5

dataset = dataset5


def train(seqs, index):
    # here we make a new agent and a new env for each seqs to train and evaluate seperately 
    # --------------------------------------------------------------------------------------
    env = Environment(seqs)
    dqn_per = DQN_PER(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
    # -------------------------------------------------------------------------------------
    total_steps = 0
    rs = []
    for episode in range(config.max_episode):
        state = env.reset()
        episode_reward = 0
        while True:
            action = dqn_per.select(state)
            reward, next_state, done = env.step(action)
            
            transition = (state, next_state, action, reward, done)
            dqn_per.replay_memory.push(transition)
            dqn_per.update()

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                break
        
        rs.append(episode_reward)
        
        dqn_per.update_epsilon()
        dqn_per.update()
        
        if episode % 1000 == 0:
            print('---------- trainining reached episode nbr:', episode)
            print('---------- reward for this episode = ', episode_reward)
            
    print("done training for this sequence of index: ", index)
    print("Episode: {}, Reward: {}, Epsilon: {:.4f}".format(episode, episode_reward, dqn_per.current_epsilon))
    
    # here you will save 25 model, each for the corresponding dataset
    fname = 'my_dqn_per_' + str(index) + '_'
    dqn_per.save(fname)
    # --------------------------------------------------------------
    
    print("done saving also for this sequence of index: ", index)
    print("\n\n")
    
    return rs


def train_all():
    import datetime
    now = datetime.datetime.now()
    print("----------- began at : ", now.time(), "--------------------\n")


    start= 0
    end= 25  #25 for comparison with zhang,  -1 for all datasets inside dataset5.py
    all_rewards = []
    for index, name in enumerate(dataset.datasets[start:end if end != -1 else len(dataset.datasets)], start):
        if not hasattr(dataset, name):
            continue
        seqs = getattr(dataset, name)
        
        rwds_each_seq = train(seqs, index)
        
        all_rewards.append(rwds_each_seq)
        
        # if index == 2:
        #     break
            
    import datetime
    now = datetime.datetime.now()
    print("\n----------- finished at : ", now.time(), "--------------------")

    return all_rewards


def main():
    # train 
    all_rs = train_all()

    # save all_rewards to csv file
    import pandas as pd
    cols = [str(i) for i in range(len(all_rs))]
    df = pd.DataFrame(all_rs, columns= cols)
    df.to_csv('all_rewards.txt', header=None, index=None, sep='\t', mode='a')


## 3ayat 3la main ta malk hhhh
# main()
