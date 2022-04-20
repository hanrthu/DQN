#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import gym
import torch.cuda
from deeprl_hw2.utils import make_env
import torch.optim as optim
import logging
import wandb
import numpy as np
import random

from deeprl_hw2.dqn import DeepQNet, LinearQNet
from deeprl_hw2.preprocessors import PreprocessorSequence, AtariPreprocessor, HistoryPreprocessor
from deeprl_hw2.memory import ReplayMemory
from deeprl_hw2.dqn import DQNAgent


def create_logger():
    # create logger
    logger = logging.getLogger('Q-Learning Logger')
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger


def init_wandb(lr, iterations, momentum, expname):
    wandb.login()
    wandb.init(project="Q-Learning", name=expname)
    config = wandb.config
    config.lr = lr
    config.iterations = iterations
    config.momentum = momentum
    config.seed = 260817
    config.log_interval = 10000


def init_random_state(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    You can use any DL library you like, including Tensorflow, Keras or PyTorch.

    If you use Tensorflow or Keras, we highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understand your network architecture if you are
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

      The Q-model.
    """
    pass


def get_output_folder(parent_dir, env_name):
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
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Games')
    parser.add_argument('--env', default='Enduro-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--input_shape', default=[210, 16, 3], help='Shape of the input')
    parser.add_argument('--large', default=False, action='store_true', help='Whether to use a larger DQN')
    parser.add_argument('--resize', default=84, help='The resize scale of input images')
    parser.add_argument('--train', default=False, action='store_true', help='Train option')
    parser.add_argument('--test', default=False, action='store_true', help='Test option')
    parser.add_argument('--deep', default=False, action='store_true', help='Use deep network')
    parser.add_argument('--num_episodes', default=100, help='The number of episodes when testing')
    parser.add_argument('--expname', default='LinearQN', help='The name of the experiment')
    parser.add_argument('--weights', default=None, help='The path to model weights')
    args = parser.parse_args()
    # print(args)
    args.input_shape = tuple(args.input_shape)
    args.output = get_output_folder(args.output, args.env)

    # hyper-parameters
    BATCH_SIZE = 32
    MAX_MEMORY_SIZE = 1000000
    HISTORY_LENGTH = 4
    TARGET_UPDATE_FREQ = 10000
    GAMMA = 0.99
    TRAIN_FREQ = 4
    EVAL_FREQ = 40000
    # EVAL_FREQ = 1000
    LR = 2.5e-4
    MOMENTUM = 0.95
    START_EPS = 1.0
    END_EPS = 0.1
    NUM_BURN_IN = 50000
    # NUM_BURN_IN = 100
    FINAL_EXPLORATION_FRAME = 500000
    ITERATIONS = 3000000

    # Initialize Logging and W&B Settings
    seed = 260817
    logger = create_logger()
    init_wandb(LR, ITERATIONS, MOMENTUM, args.expname)
    init_random_state(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    # env = gym.make("PongNoFrameskip-v4")
    env = gym.make('Enduro-v0')
    # env = gym.make('SpaceInvaders-v0')

    env = make_env(env)
    num_actions = env.action_space.n
    if args.deep:
        q_net = DeepQNet(HISTORY_LENGTH, num_actions, args.large).to(device)
        qminus_net = DeepQNet(HISTORY_LENGTH, num_actions, args.large).to(device)
    else:
        q_net = LinearQNet(HISTORY_LENGTH, num_actions).to(device)
        qminus_net = LinearQNet(HISTORY_LENGTH, num_actions).to(device)
    if args.weights is not None:
        q_net.load_state_dict(torch.load(args.weights))
    wandb.watch(q_net, log="all")
    optimizer = optim.RMSprop(params=q_net.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.999)
    atari_pro = AtariPreprocessor(args.resize)
    history_pro = HistoryPreprocessor(HISTORY_LENGTH)
    preprocessor = PreprocessorSequence([atari_pro, history_pro])
    memory = ReplayMemory(MAX_MEMORY_SIZE, device)
    agent = DQNAgent(
        q_net,
        qminus_net,
        preprocessor,
        memory,
        GAMMA,
        START_EPS,
        END_EPS,
        TARGET_UPDATE_FREQ,
        NUM_BURN_IN,
        TRAIN_FREQ,
        EVAL_FREQ,
        BATCH_SIZE,
        ITERATIONS,
        FINAL_EXPLORATION_FRAME,
        optimizer,
        scheduler,
        device,
        logger,
        args
    )
    if args.train:
        logger.info("Start Training Q Networks...")
        agent.fit(env, ITERATIONS)
        wandb.save('model.h5')
    if args.test:
        logger.info("Start Testing Q Networks...")
        agent.evaluate(env, args.num_episodes)


if __name__ == '__main__':
    main()
