import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
    	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    	self.states = []
    	self.actions = []
    	self.next_states = []
    	self.rewards = []
    	self.dones = []
    	self.device = device

    def add(self, state, action, reward,next_state,done):
    	self.states.append(state)
    	self.actions.append(action)
    	self.next_states.append(next_state)
    	self.rewards.append(reward)
    	self.dones.append(done)

    def sample(self, size, batch_size):
    	
    	ind = np.random.randint(0, size, size=batch_size)

    	return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.dones[ind]).to(self.device)
		)

    def save(self, save_folder):
    	np.save(f"{save_folder}_state.npy", self.states)
    	np.save(f"{save_folder}_action.npy", self.actions)
    	np.save(f"{save_folder}_next_state.npy", self.next_states)
    	np.save(f"{save_folder}_reward.npy", self.rewards)
    	np.save(f"{save_folder}_dones.npy", self.dones)

    def load(self, save_folder):
    	reward_buffer = np.load(f"{save_folder}_reward.npy")
		# Adjust crt_size if we're using a custom size
    	self.state = np.load(f"{save_folder}_state.npy")
    	self.action = np.load(f"{save_folder}_action.npy")
    	self.next_state= np.load(f"{save_folder}_next_state.npy")
    	self.reward = reward_buffer
    	self.dones = np.load(f"{save_folder}_dones.npy")

    	return self.state, self.action, self.next_state, self.reward, self.dones