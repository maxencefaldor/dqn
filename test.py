#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:33:43 2021

@author: max
"""

from prioritized_replay_buffer import ReplayBuffer

rb = ReplayBuffer(64)

for i in range(100):
    rb.add(i, i, i, i, i)
