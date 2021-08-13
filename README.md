# Deep Q-Network

Implementation of the DQN algorithm and six independent improvements as described in the paper Rainbow: Combining Improvements in Deep Reinforcement Learning [[1]](#references).

- DQN [[2]](#references)
- Double DQN [[3]](#references)
- Prioritized Experience Replay [[4]](#references)
- Dueling Network Architecture [[5]](#references)
- Multi-step Bootstrapping [[6]](#references)
- Distributional RL [[7]](#references)
- Noisy Networks [[8]](#references)

I provide a `main.py` as well as a Jupyter Notebook which demonstrate how to set up, train and compare multiple agents to reproduce the results of the aforementioned paper.

Don't hesitate to modify the default hyperparameters or the code to see how your new algorithm compare to the standard ones.

## Project Structure

    ├── README.md
    ├── main.py                             # Lab where agents are declared, trained and compared
    ├── .gitignore
    ├── agents
    │   ├── dqn_agent.py                    # DQN agent
    │   ├── ddqn_agent.py                   # Double DQN agent
    │   └── rainbow_agent.py                # Rainbow ageent
    ├── replay_memory
    │   ├── replay_buffer.py                # The standard DQN replay memory
    │   ├── prioritized_replay_buffer.py    # Prioritized replay memory using a sum tree to sample
    │   └── sum_tree.py                     # Sum tree implementation used by the prioritized replay memory
    └── utils
        ├── network_architectures.py        # A collection of network architectures including standard, dueling, noisy or distributional
        ├── wrappers.py                     # Wrappers and utilities to create Gym environments
        └── plot.py                         # Plot utilities to display agents' performances

In `wrappers.py`, I also provide a clean implementation of a CartPole Swing Up environment. The pole starts hanging down and the cart must first swing the pole to an upright position before balancing it as in normal CartPole.

## Instructions

First download the source code.
```
git clone https://github.com/maxencefaldor/dqn.git
```
Finally setup the environment and install dqn's dependencies
```
pip install -U pip
pip install -r dqn/requirements.txt
```

### Requirements

- [PyTorch](http://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Gym](https://gym.openai.com/)
- [atari-py](https://github.com/openai/atari-py)
- [Matplotlib](https://matplotlib.org/)

## Acknowledgements

- [@openai](https://github.com/openai) for [Baselines](https://github.com/openai/baselines)
- [@google](https://github.com/google) for [a sum tree implementation](https://github.com/google/dopamine/blob/master/dopamine/replay_memory/sum_tree.py)
- [@higgsfield](https://github.com/higgsfield) for [RL-Adventure](https://github.com/higgsfield/RL-Adventure)

## References

[1] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298), Hessel et al., 2017.  
[2] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602), Mnih et al., 2013.  
[3] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461), Hasselt et al., 2015.  
[4] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952), Schaul et al., 2015.  
[5] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581), Wang et al., 2015.  
[6] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html), Sutton and Barto, 1998.  
[7] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), Bellemare et al., 2017.  
[8] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295), Fortunato et al., 2017.  
