#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack


class AtariWrapper(AtariPreprocessing):
    def __init__(self, env, action_map, **kwargs):
        super(AtariWrapper, self).__init__(env, **kwargs)
        self.action_map = action_map
    
    def step(self, action):
        return super(AtariWrapper, self).step(self.action_map[action])

def make_atari(env_id, action_map):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    env = AtariWrapper(env,
                       action_map=action_map,
                       noop_max=0,
                       frame_skip=4,
                       terminal_on_life_loss=True,
                       grayscale_obs=True,
                       scale_obs=True)
    env = FrameStack(env, 4)
    return env
