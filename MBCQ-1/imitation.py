from algorithms.mbcq import MBCQ
import argparse
from tools import ReplayBuffer
from pathlib import Path
import torch
import numpy as np
from algorithms.maddpg import MADDPG
from utils.make_env import make_env
import os
from torch.autograd import Variable


def train_MBCQ(args):
    
    # For saving files
	setting = f"{args.env_id}_{args.seed}"
	buffer_name = f"{args.model_name}_{setting}"
	if not os.path.exists('./results'):
	    os.mkdir('./results')
	# Load buffer
	buffer_path= (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num)/ 'buffer')
	replay_buffer = ReplayBuffer()
	states, actions, next_states,rewards,dones= replay_buffer.load(buffer_path)
	# print('next stat',states.shape)
    # [n_data,agent,state_dim]
    # Initialize policy, we put all the agent's obs into the state and action, not sure??
	policy = MBCQ(state_dim=3*states.shape[2], action_dim=3*actions.shape[2], max_action=np.amax(actions))

	evaluations = []
	episode_num = 0
	done = True 
	training_iters = 0
	size=states.shape[0]
	while training_iters < args.max_timesteps: 
		pol_vals = policy.train(size,replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

		evaluations.append(eval_policy(policy, args.env_id))
		np.save(f"./results/MBCQ_{setting}", evaluations)

		training_iters += args.eval_freq
		print(f"Training iterations: {training_iters}")



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, eval_episodes=10):
	model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
	if config.incremental is not None:
		model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
	else:
		model_path = model_path / 'model.pt'
	maddpg = MADDPG.init_from_save(model_path)
	eval_env = make_env(env_name, discrete_action=maddpg.discrete_action)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state,done= eval_env.reset(),False
		# torch_obs = [Variable(torch.Tensor(state[i]).view(1, -1),
        #                         requires_grad=False)
        #                 for i in range(maddpg.nagents)]
		while not done:
			action = policy.select_action(np.array(state).reshape(-1,))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += np.mean(reward)

	avg_reward /= eval_episodes
	print('avg reward',avg_reward)
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward






if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("env_id", help="Name of environment")
	parser.add_argument("model_name",
                        help="Name of model")
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("run_num", default=1, type=int)
	parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
	parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
	parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
	parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
	parser.add_argument("--eval_freq", default=5e1, type=float)     # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
	parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
	config = parser.parse_args()
	
	train_MBCQ(config)