import torch
import torch.nn as nn
import numpy as np
import random
import os
from collections import deque
import config
from models import Encoder

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(self, transition):
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[index] for index in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        state, next_state, action, reward, done = [], [], [], [], []

        for i in range(batch_size):
            s, ns, a, r, d = samples[i]
            state.append(s)
            next_state.append(ns)
            action.append(a)
            reward.append(r)
            done.append(d)

        return state, next_state, action, reward, done, indices, weights

    def update_priorities(self, indices, errors, epsilon=1e-6):
        for index, error in zip(indices, errors):
            priority = abs(error) + epsilon
            self.priorities[index] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)

    @property
    def size(self):
        return len(self.buffer)



class Net(nn.Module):
    """docstring for Net"""

    def __init__(self, seq_num, max_seq_len, action_number, max_value, d_model=64):
        super(Net, self).__init__()
        self.max_value = max_value
        dim = seq_num * (max_seq_len + 1)
        # layers:
        self.encoder = Encoder(6, d_model, dim)
        self.dropout = nn.Dropout()
        self.l1 = nn.Linear(dim * d_model, 1028)
        self.f1 = nn.LeakyReLU()
        self.l2 = nn.Linear(1028, 512)
        self.f2 = nn.LeakyReLU()
        self.l3 = nn.Linear(512, action_number)
        self.f3 = nn.Tanh()

        self.mask = lambda x, y: (x != y).unsqueeze(-2)

    def forward(self, x):
        x = self.encoder(x, self.mask(x, 0))
        # x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.f1(self.l1(x))
        x = self.f2(self.l2(x))
        x = self.f3(self.l3(x))
        x = torch.mul(x, self.max_value)

        return x


class DQN_PER:
    def __init__(self, action_number, seq_num, max_seq_len, max_value):
        super(DQN_PER, self).__init__()
        self.seq_num = seq_num
        self.max_seq_len = max_seq_len
        self.action_number = action_number

        self.eval_net = Net(seq_num, max_seq_len, action_number, max_value).to(config.device)
        self.target_net = Net(seq_num, max_seq_len, action_number, max_value).to(config.device)

        self.current_epsilon = config.epsilon

        self.update_step_counter = 0
        self.epsilon_step_counter = 0

        self.replay_memory = PrioritizedReplayMemory(config.replay_memory_size, config.alpha)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=config.alpha)
        self.loss_func = nn.SmoothL1Loss()

    def update_epsilon(self):
        self.epsilon_step_counter += 1
        if self.epsilon_step_counter % config.decrement_iteration == 0:
            self.current_epsilon = max(0, self.current_epsilon - config.delta)

    def select(self, state):
        if random.random() <= self.current_epsilon:
            action = np.random.randint(0, self.action_number)
        else:
            action_val = self.eval_net.forward(torch.LongTensor(state).unsqueeze_(0).to(config.device))
            action = torch.argmax(action_val, 1).cpu().data.numpy()[0]

        return action

    def predict(self, state):
        action_val = self.eval_net.forward(torch.LongTensor(state).unsqueeze_(0).to(config.device))
        return torch.argmax(action_val, 1).cpu().data.numpy()[0]

    def update(self):
        self.update_step_counter += 1
        if self.update_step_counter % config.update_iteration == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        if self.replay_memory.size < config.batch_size:
            return

        state, next_state, action, reward, done, indices, weights = self.replay_memory.sample(config.batch_size)

        batch_state = torch.LongTensor(state).to(config.device)
        batch_next_state = torch.LongTensor(next_state).to(config.device)
        batch_action = torch.LongTensor(action).to(config.device)
        batch_reward = torch.FloatTensor(reward).to(config.device)
        batch_done = torch.FloatTensor(done).to(config.device)
        batch_weights = torch.FloatTensor(weights).to(config.device)

        q_eval = self.eval_net(batch_state).gather(1, batch_action.unsqueeze_(-1)).squeeze_(1).to(config.device)

        q_next = self.target_net(batch_next_state).max(1)[0].to(config.device).detach_()
        q_target = batch_reward + batch_done * config.gamma * q_next

        loss = self.loss_func(q_eval, q_target)
        loss = (loss * batch_weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        errors = torch.abs(q_target - q_eval).detach().cpu().numpy()
        self.replay_memory.update_priorities(indices, errors)

    def save(self, filename, path=config.weight_path):
        torch.save(self.eval_net.state_dict(), os.path.join(path, "{}.pth".format(filename)))
        print("{} has been saved...".format(filename))
   
    def load(self, filename, path=config.weight_path):
        self.eval_net.load_state_dict(torch.load(os.path.join(path, "{}.pth".format(filename))))
        self.target_net.load_state_dict(self.eval_net.state_dict())
        print("{} has been loaded...".format(filename))




