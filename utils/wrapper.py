#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack


def make_atari(env_id):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    env = AtariPreprocessing(env,
                             noop_max=0,
                             frame_skip=4,
                             terminal_on_life_loss=True,
                             grayscale_obs=True,
                             scale_obs=True)
    env = FrameStack(env, 4)
    return env

class PongWrapper(AtariPreprocessing):
    def __init__(self, env, **kwargs):
        super(PongWrapper, self).__init__(env, **kwargs)

    def step(self, action):
        return super(PongWrapper, self).step(4 + action)
