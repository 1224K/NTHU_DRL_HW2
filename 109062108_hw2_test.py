# 1. Implement and train your agent in SuperMarioBros-v0 environment.
# 2. Use DQN as your core algorithm, but enhancements are permitted (DQN、Double DQN、DRQN、Dueling DQN、Prioritized Experience Replay, ...). Anything that is not fit with DQN is now allowed. (3C、PPO、Distributional RL...) 
# 3. You may store your learned model in an external file “./<Student_ID>_hw2_data” and access it with your program (during testing), the saved model file size should be <10MB.
import gym
import torch.nn as nn
import numpy as np
import torch
import copy
from collections import deque
import random
import cv2
import matplotlib.pyplot as plt


# Double Deep Q-Networks
class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DDQN, self).__init__()
        # w = h = 84
        c, w, h = input_dim
        self.online = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),

            # conv_out_size = 7 * 7 * 64 = 3136
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Freeze target network
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model):
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)
        
class Agent(object):
    """DDQN """
    def __init__(self, use_cuda=False, training=False):
        self.state_dim = (4, 84, 84)
        self.action_dim = 12
        self.memory = deque(maxlen=40000)
        self.batch_size = 64
        
        self.net = DDQN(self.state_dim, self.action_dim)
        
        self.use_cuda = use_cuda
        self.training = training
        if use_cuda:
            self.net = self.net.cuda()

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999975
        self.exploration_rate_min = 0.01
        self.curr_step = 0

        self.save_every = 5e5

        self.gamma = 0.9
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.frameStack = deque([], maxlen=4)
        self.frameStack.extend([np.zeros((1, 84, 84), dtype=np.float32) for _ in range(4)])
       
    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx =  np.random.choice(self.action_dim)
        else :
            if not self.training:
                # preporcessing the state
                # ProcessFrame84
                state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114
                resized_screen = cv2.resize(state, (84, 110), interpolation=cv2.INTER_AREA)
                x_t = resized_screen[18:102, :]
                x_t = np.reshape(x_t, [84, 84, 1])
                # # draw the state
                # plt.imshow(x_t, cmap='gray')
                # plt.show()

                # ImageToPyTorch
                state = np.moveaxis(x_t, 2, 0)

                # FrameStack
                self.frameStack.append(state)
                state = np.concatenate(self.frameStack, axis=0)


            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

            if self.use_cuda:
                state = state.cuda()
            q_values = self.net(state, model="online")
            action_idx = torch.argmax(q_values, axis=1).item() 

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate) 

        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        if self.use_cuda:
            state = state.cuda()
            next_state = next_state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            done = done.cuda()

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action, reward, done
    
    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if len(self.memory) < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return (td_est.mean().item(), loss)
    
    def save(self, path):
        torch.save(self.net.online.state_dict(), path)
    
    def load(self, path):
        self.net.online.load_state_dict(torch.load(path))
        self.net.target.load_state_dict(torch.load(path))
        for p in self.net.target.parameters():
            p.requires_grad = False

    def to(self, device):
        self.net.to(device)
        return self
