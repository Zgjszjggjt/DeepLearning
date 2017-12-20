#!/usr/bin/evn python
#-*- coding: utf-8 -*-

# ===================================
# Filename : cartpule.py
# Author : GT
# Create date : 17-12-19 15:37:48
# Description:
# ===================================


# Script starts from here

from __future__ import absolute_import
from __future__ import division
# this is for chinese characters
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

import gym
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as activation


STATE_DIM = 4
ACTION_DIM = 2
STEP = 2000
SAMPLE_NUM = 30


class actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = activation.relu(self.fc1(x))
        x = activation.relu(self.fc2(x))
        x = activation.log_softmax(self.fc3(x))
        return x

class value_network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(value_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = activation.relu(self.fc1(x))
        x = activation.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def sample(actor, value_network, initial_state, sample_num, game):
    states = []
    actions = []
    rewards = []
    terminal = False
    final_reward = 0
    state = initial_state
    
    for i in range(sample_num):
        states.append(state)
        at = actor(Variable(torch.Tensor([state])))
        at_softmax = torch.exp(at)
        action = np.random.choice(ACTION_DIM, p=at_softmax.cpu().data.numpy()[0])
        one_hot_actor = [int(k == action) for k in range(ACTION_DIM)]
        next_state, reward, terminal, _ = game.step(action)
        actions.append(one_hot_actor)
        rewards.append(reward)
        state = next_state
        if terminal:
            state = game.reset()
            break

    if not terminal:
        final_reward = value_network(Variable(torch.Tensor([state]))).cpu().data.numpy()

    return states, actions, rewards, final_reward, state


def discount_rewards(rewards, gamma, final_reward):
    d_reward = np.zeros_like(rewards)
    total = final_reward
    for i in reversed(range(len(rewards))):
        total = total * gamma + rewards[i]
        d_reward[i] = total
    return d_reward


def main():
    game = gym.make('CartPole-v0')
    initial_state = game.reset()

    vn = value_network(STATE_DIM, 1, 40)
    vm_op = torch.optim.Adam(vn.parameters(), lr=0.01)

    an = actor(STATE_DIM, ACTION_DIM, 40)
    an_op = torch.optim.Adam(an.parameters(), lr=0.01)


    steps = []
    game_episodes = []
    test_result = []

    for step in range(STEP):
        states, actions, rewards, final_reward, current_state = sample(an, vn, initial_state, SAMPLE_NUM, game)
        initial_state = current_state
        action_var = Variable(torch.Tensor(actions).view(-1, ACTION_DIM))
        states_var = Variable(torch.Tensor(states).view(-1, STATE_DIM))
        
        an_op.zero_grad()
        log_softmax_actions = an(states_var)
        vs = vn(states_var).detach()
        qs = Variable(torch.Tensor(discount_rewards(rewards, 0.9, final_reward)))
        advantage = qs - vs
        loss = - torch.mean(torch.sum(log_softmax_actions * action_var, 1) * advantage)
        loss.backward()
        torch.nn.utils.clip_grad_norm(an.parameters(), 0.5)
        an_op.step()

        vm_op.zero_grad()
        target_value = qs
        real_value = vn(states_var)
        msel = nn.MSELoss()
        vl = msel(real_value, target_value)
        vl.backward()
        torch.nn.utils.clip_grad_norm(vn.parameters(), 0.5)
        vm_op.step()

        if (step+1) % 50 == 0:
            result = 0
            test_game = gym.make('CartPole-v0')
            for test_episode in range(10):
                test_state = test_game.reset()
                for test_step in range(200):
                    t_a = torch.exp(an(Variable(torch.Tensor([test_state]))))
                    action = np.argmax(t_a.data.numpy()[0])
                    next_state, reward, terminal, _ = test_game.step(action)
                    result += reward
                    test_state = next_state
                    if terminal:
                        break
            print 'step:%d   reward:%d'%(step + 1, result)
            steps.append(step + 1)
            test_result.append(result / 10)
    
    print 'test'
    terminal = False
    initial_state = game.reset()
    while True:
        #if terminal:
         #   initial_state = game.reset()
          #  print 'reset'
        action = torch.exp(an(Variable(torch.Tensor([initial_state]))))
        action = np.argmax(action.data.numpy()[0])
        next_state, _, terminal, _ = game.step(action)
        game.render()

if __name__ == '__main__':
    main()
