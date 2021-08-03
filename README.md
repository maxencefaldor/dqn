# Deep Q-Network

Implementation of the DQN algorithm and six independent improvements as described in the paper Rainbow: Combining Improvements in Deep Reinforcement Learning [[1]](#references).

- DQN [[2]](#references)
- Double DQN [[3]](#references)
- Prioritized Experience Replay [[4]](#references)
- Dueling Network Architecture [[5]](#references)
- Multi-step Bootstrapping [[6]](#references)
- Distributional RL [[7]](#references)
- Noisy Networks [[8]](#references)

# Requirements

- [PyTorch](http://pytorch.org/)
- [atari-py](https://github.com/openai/atari-py)
- [OpenCV Python](https://pypi.python.org/pypi/opencv-python)

# Acknowledgements

- [@openai](https://github.com/openai) for [Baselines](https://github.com/openai/baselines)
- [@google](https://github.com/google) for [a sum tree implementation](https://github.com/google/dopamine/blob/master/dopamine/replay_memory/sum_tree.py)
- [@jaara](https://github.com/jaara) for [AI-blog](https://github.com/jaara/AI-blog)

# References

[1] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298), Hessel et al., 2017.  
[2] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602), Mnih et al., 2013.  
[3] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461), Hasselt et al., 2015.  
[4] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952), Schaul et al., 2015.  
[5] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581), Wang et al., 2015.  
[6] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html), Sutton and Barto, 1998.  
[7] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), Bellemare et al., 2017.  
[8] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295), Fortunato et al., 2017.  
