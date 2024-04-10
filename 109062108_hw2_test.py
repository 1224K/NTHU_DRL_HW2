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
# import matplotlib.pyplot as plt
import datetime
from torchvision import transforms as T

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
    def __init__(self):
        self.state_dim = (4, 84, 84)
        self.action_dim = 12
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        
        self.net = DDQN(self.state_dim, self.action_dim)
        
        self.use_cuda = False
        self.training = False

        self.exploration_rate = 0.1
        self.exploration_rate_decay = 0.9999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.gamma = 0.9
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 1  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001) # 0.00025
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.frameStack = deque(maxlen=4)
        self.action = 0

        self.init_state_hash = None
        self.load("./model_4000")
        # self.timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        self.random_seed = 2
       
    def act(self, observation):
        if not self.training:
            if self.init_state_hash is None:
                self.init_state_hash = hash(observation.tobytes())
                self.frameStack.extend([np.zeros((1, 84, 84), dtype=np.float32 ) for _ in range(4)])
                self.curr_step = 0
                self.action = 0
                random.seed(self.random_seed)
            elif self.init_state_hash == hash(observation.tobytes()):
                self.frameStack.extend([np.zeros((1, 84, 84), dtype=np.float32) for _ in range(4)])
                self.curr_step = 0
                self.action = 0
                random.seed(self.random_seed)

            if self.curr_step % 4 != 0:
                self.curr_step += 1
                return self.action
            
        if random.random() < self.exploration_rate:
            # self.action =  np.random.choice(self.action_dim)
            self.action =  random.choice([1, 2])
            if not self.training:
                observation = np.transpose(observation, (2, 0, 1))
                observation = torch.tensor(observation.copy(), dtype=torch.float)   
                transform = T.Grayscale()
                observation = transform(observation)
                # ResizeObservation
                transforms = T.Compose(
                    [T.Resize((84, 84), antialias=True), T.Normalize(0, 255)]
                )
                observation = transforms(observation)
               
                # FrameStack
                self.frameStack.append(observation)
        else :
            if not self.training:
                # preporcessing the state
                # GrayScaleObservation
                # permute [H, W, C] array to [C, H, W] tensor
                observation = np.transpose(observation, (2, 0, 1))
                observation = torch.tensor(observation.copy(), dtype=torch.float)   
                transform = T.Grayscale()
                observation = transform(observation)
                # ResizeObservation
                transforms = T.Compose(
                    [T.Resize((84, 84), antialias=True), T.Normalize(0, 255)]
                )
                observation = transforms(observation)
                # print(observation.shape)
                # print(observation.numpy().shape)
                # cv2.imwrite(f"./test/{self.curr_step}.png", observation.numpy()[0]*255)
                # FrameStack
                self.frameStack.append(observation)

                observation = np.concatenate(self.frameStack, axis=0)

            observation = np.array(observation)
            observation = torch.tensor(observation).unsqueeze(0)

            if self.use_cuda:
                observation = observation.cuda()
            q_values = self.net(observation, model="online")
            self.action = torch.argmax(q_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate) 

        self.curr_step += 1
        return self.action
    
    def cache(self, state, next_state, action, reward, done):
        state = np.array(state)
        next_state = np.array(next_state)
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
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
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
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if len(self.memory) < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return (td_est.mean().item(), loss)
    
    def save(self, path):
        torch.save(self.net.online.state_dict(), path)
    
    def load(self, path):
        self.net.to(torch.device('cuda:0' if self.use_cuda else 'cpu'))
        self.net.online.load_state_dict(torch.load(path, map_location=torch.device('cuda:0' if self.use_cuda else 'cpu')))
        self.net.target.load_state_dict(torch.load(path, map_location=torch.device('cuda:0' if self.use_cuda else 'cpu')))
        for p in self.net.target.parameters():
            p.requires_grad = False
    