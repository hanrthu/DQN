#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import torch
# import tensorflow as tf
# from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
#                           Permute)
# from keras.models import Model
# from keras.optimizers import Adam

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import AtariPreprocessor, HistoryPreprocessor, PreprocessorSequence
from deeprl_hw2.utils import make_atari, set_seed
from deeprl_hw2.models import LinearDQN, DeepDQN
from deeprl_hw2.core import NoMemory, ReplayMemory
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.policy import *
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import gym
from gym import spaces
import cv2


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    pass


'''def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir'''

def tovedio(frames, video_path, fps=10):
    isColor = (len(frames[0].shape) == 3)
    img_size = (frames[0].shape[1], frames[0].shape[0])
    print(img_size)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videowriter = cv2.VideoWriter(video_path, fourcc, fps, img_size, isColor=isColor)
    for frame in frames:
        videowriter.write(frame)
        # print(frame + " has been written!")
    videowriter.release()


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--path', default='LinearDQN', help='Directory to save data to')
    parser.add_argument('--load', default=None, help='Directory to save data to')
    parser.add_argument('--target_update_freq', default=10000, type=int, help='')
    parser.add_argument('--train_freq', default=1, type=int, help='')
    parser.add_argument('--num_burn_in', default=10000, type=int, help='')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--seed', default=952, type=int, help='Random seed')
    parser.add_argument('--gamma', default=0.99, type=float, help='Gamma')
    parser.add_argument('--lr', default=2e-4, type=float, help='Learning Rate')
    parser.add_argument('--history_length', default=4, type=int, help='how many frames to be a state')
    parser.add_argument('--reshape_size', default='84,84', type=str, help='shape for network input')
    parser.add_argument('--iterations', default=5000000, type=int, help='train iterations')
    parser.add_argument('--buffer_length', default=100000, help='train iterations')
    parser.add_argument('--deep', action='store_true', default=False)
    parser.add_argument('--noreplay', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--evaluate', action='store_true', default=False)

    args = parser.parse_args()
    args.device = 'cuda:0'
    set_seed(args.seed)
    # args.input_shape = tuple(args.input_shape)

    # args.output = get_output_folder(args.output, args.env)
    args.reshape_size = args.reshape_size.split(',')
    args.reshape_size = tuple([int(t) for t in args.reshape_size])

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    if 'NoFrameskip' in args.env:
        env = make_atari(args.env) # only use in no frameskip environment
    else:
        env = gym.make(args.env)
    print(env.action_space.n)
    historyProcessor = HistoryPreprocessor(args.history_length)
    preprocessor = AtariPreprocessor(args.reshape_size)
    processorSeq = PreprocessorSequence([preprocessor, historyProcessor])  # return 84 * 84 * 4 for one state
    if not args.deep:
        cls = LinearDQN
    else:
        cls = DeepDQN
    network = cls(env.action_space.n).to(args.device)
    if args.load is not None:
        network.load_state_dict(torch.load(args.load))
        args.path = 'video/' + args.load.replace('.bin', '.avi').replace('/', '-')
    target_network = cls(env.action_space.n).to(args.device)
    if not args.noreplay:
        memory = ReplayMemory(args.buffer_length)
    else:
        memory = NoMemory()
    if args.train:
        policy = LinearDecayGreedyEpsilonPolicy(1.0, 0.05, args.iterations, env.action_space.n)
    else:
        policy = LinearDecayGreedyEpsilonPolicy(0.05, 0.05, args.iterations, env.action_space.n)
    optimizer = torch.optim.RMSprop(network.parameters(), lr=args.lr, eps=0.001, alpha=0.95)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.999)
    agent = DQNAgent(network, target_network, processorSeq, memory, policy, optimizer, scheduler, args)
    if args.train:
        agent.fit(env, args.iterations)
    if args.evaluate:
        agent.evaluate(env, args.iterations)


if __name__ == '__main__':
    main()
