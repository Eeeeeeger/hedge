#!/usr/bin/env python
# coding=utf-8

from .BaseStrategy import BaseStrategy


class HedgeAllStrategy(BaseStrategy):

    def cal_position(self, greek_df):
        # cash_delta / index点位，乘上weight，再按multiplier规整
        position = -greek_df['delta']
        position_w = self.hedge_weight.loc[greek_df.index].mul(position, axis=0)
        self.spot_position = round(position_w/self.multiplier)*self.multiplier
        # self.spot_position = position_w
        return self.spot_position