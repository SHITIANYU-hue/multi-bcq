import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
from tools import ReplayBuffer
from gym.spaces import Box, Discrete
import os

def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    buffer_path= (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num)/ 'buffer')
    # print(buffer_path)
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    ##build replay buffer
    replay_buffer = ReplayBuffer()
    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        if config.render:
            if config.save_gifs:
                frames = []
                frames.append(env.render('rgb_array')[0])
            env.render('human')
        ### how to save replay buffer?
        for t_i in range(config.episode_length):
            calc_start = time.time()
            ##rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            ## torch_obs=[agent_num, feature_num]
            # print('torch obs shape', torch_obs[1].shape)
            ##get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            ## torch action=[agent_num,action_dim]
            # print('torch actions shape',torch_actions)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            # print('actions',actions)
            next_obs, rewards, dones, infos = env.step(actions)
            # print('next obs',next_obs[0].shape)
            # print('obs',obs[0].shape)
            ## whether i should put agent actions into replay buffer? (with noise?)
            replay_buffer.add(np.asarray(obs), np.asarray(actions), np.asarray(rewards), np.asarray(next_obs), np.asarray(dones))
            obs = next_obs
            if config.render:
                if config.save_gifs:
                    frames.append(env.render('rgb_array')[0])
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < ifi:
                    time.sleep(ifi - elapsed)
                env.render('human')
        replay_buffer.save(buffer_path)
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--render", type=bool, default=False, help="whether to render")

    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    config = parser.parse_args()

    run(config)