#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np

class Simulation:
    @staticmethod
    def exact_bs_sim(St: float, mu: float, sigma: float, T: float, num: int, random_state: int = 0) -> np.array:
        '''
        Monte Carlo GBM for stock price given initial price St
        '''
        np.random.seed(random_state)
        steps = mu*(T/(num-1)) + sigma*np.sqrt(T/(num-1))*np.random.normal(size=num-1) + 1
        steps = np.insert(steps, 0, St)
        return steps.cumprod()

    @staticmethod
    def exact_bs_sim_mid(mid_pos: int, St: float, mu: float, sigma: float, T: float, num: int, random_state: int = 0) -> np.array:
        '''
        Monte Carlo GBM for stock price given mid point price St
        '''
        np.random.seed(random_state)
        steps = mu*(T/(num-1)) + sigma*np.sqrt(T/(num-1))*np.random.normal(size=num-1) + 1
        steps = np.insert(steps, mid_pos-1, St)
        temp = 1 / steps[:mid_pos-1][::-1]

        return np.hstack([np.insert(temp, 0, 10).cumprod()[::-1][:-1], steps[mid_pos-1:].cumprod()])
