# Source
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gym 
import math
import random
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from collections import namedtuple
from itertools import count
from PIL import Image

# Neuran network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T 


# Define environment
env = gym.make('CartPole-v0').unwrapped

# Setup matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

if is_ipython:
	from IPyton import display

plt.ion()

# If gpu is to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Experience replay memory for training the DQN. Stabilizes and improve DQN training procedure.

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Store the agent's experiences at each time-step
class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Save a transition"""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return(random.sample(self.memory, batch_size)) #Sample the experience

	def __len__(self):
		return(len(self.memory))

class DQN(nn.module):

	def __init__(self, h, w, outputs):
		super(DQN, self).__init__() # Returns a temporary object of the superclass that then allows you to call that superclass's methods

		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)


