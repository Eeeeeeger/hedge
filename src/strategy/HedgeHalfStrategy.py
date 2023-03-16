#!/usr/bin/env python
# coding=utf-8

import pandas as pd

from .BaseStrategy import BaseStrategy
import numpy as np


class HedgeHalfStrategy(BaseStrategy):

    def cal_position(self, greek_df):
        hedge_all_position = -greek_df['cash_delta'].values
        trade_days = len(self.spot_position)
        position = np.zeros(trade_days)
        position[0] = hedge_all_position[0]
        for t in range(1, trade_days):
            position[t] = (hedge_all_position[t] + position[t - 1]) / 2
        position = pd.DataFrame(position, index=greek_df.index)
        position_w = self.hedge_weight.mul(position, axis=0)
        self.spot_position = round(position_w / self.multiplier) * self.multiplier
        return  self.spot_position