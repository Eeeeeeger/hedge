#!/usr/bin/env python
# coding=utf-8

import pandas as pd
from abc import abstractmethod
from src.data.Container import _container
from src.option.BaseOption import BaseOption
from src.option.Portfolio import OptionPortfolio


class BaseStrategy:

    def __init__(self, hedge_asset: str = None, hedge_code: list = None):
        # option underlying
        self.spot_position: pd.DataFrame = None

        # hedge asset
        self.hedge_asset = hedge_asset
        self.hedge_code = hedge_code
        self.set_hedge_asset()
        self.multiplier = 100

    def set_hedge_asset(self):
        if self.hedge_code is not None:
            hedge_asset_data = pd.DataFrame()
            for _code in self.hedge_code:
                hedge_asset_data = pd.concat([hedge_asset_data, _container.get_data(self.hedge_asset, _code).assign(CODE=_code)], axis=0)
            self.hedge_price = hedge_asset_data.loc[:, ['CLOSE', 'CODE']].set_index('CODE',append=True)['CLOSE'].unstack()

            # 股票的hedge_weight是全为1的列
            if 'VOLUME' in hedge_asset_data.columns:
                hedge_volume = hedge_asset_data.loc[:, ['VOLUME', 'CODE']].set_index('CODE',append=True)['VOLUME'].unstack()
                tol_volume = hedge_volume.sum(axis=1)
                self.hedge_weight = hedge_volume.div(tol_volume, axis=0)
            else:
                self.hedge_weight = pd.DataFrame(index=self.hedge_price.index,
                                                 columns=self.hedge_code,
                                                 data=1 / len(self.hedge_code))

    @abstractmethod
    def cal_position(self, greek_df):
        pass

    # def pnl_decompose(self):
    #     if self.spot_position is None:
    #         self.cal_position()
    #     self.hedge_pnl_df = pd.DataFrame(index=self.spot_position.index,
    #                                      columns=['CA', 'MA', 'AA', 'delta_AA', 'init_MA', 'min_AA', 'max_AA', 'TC', 'IC', 'delta_nav', 'nav', 'is_trade'])
    #     # 对于hedge code是多个来说，需要在axis=1的维度上求和，得到所有资产的总市值
    #     self.hedge_pnl_df['AA'] = (self.spot_position * self.hedge_price).sum(axis=1)
    #     self.hedge_pnl_df['delta_AA'] = (self.hedge_price.diff() * self.spot_position.shift()).sum(axis=1)
    #     self.hedge_pnl_df['TC'] = (self.spot_position.diff() * self.hedge_price * self.r_tc).sum(axis=1)
    #     self.hedge_pnl_df['min_AA'] = self.hedge_pnl_df['AA'] * self.min_mr
    #     self.hedge_pnl_df['max_AA'] = self.hedge_pnl_df['AA'] * self.max_mr
    #     self.hedge_pnl_df['init_AA'] = self.hedge_pnl_df['AA'] * self.init_mr
    #     self.hedge_pnl_df['is_trade'] = (self.spot_position.diff() != 0).astype(int)
    #     bt_date_list = list(self.spot_position.index)
    #     for _i, _date in enumerate(bt_date_list):
    #         if _i == 0:
    #             self.hedge_pnl_df.loc[_date, 'delta_AA'] = 0
    #             self.hedge_pnl_df.loc[_date, 'TC'] = (self.spot_position.loc[_date] * self.hedge_price.loc[_date] * self.r_tc).sum()
    #             self.hedge_pnl_df.loc[_date, 'IC'] = 0
    #             self.hedge_pnl_df.loc[_date, 'MA'] = self.hedge_pnl_df.loc[_date, 'AA'] * self.init_mr
    #             self.hedge_pnl_df.loc[_date, 'CA'] = - self.hedge_pnl_df.loc[_date, 'MA']
    #             self.hedge_pnl_df.loc[_date, 'is_trade'] = 1
    #             continue
    #         _prv_date = bt_date_list[_i-1]
    #         # 更新MA，是否需要从CA借入资金，是否需要提取资金存入CA
    #         tmp_MA = self.hedge_pnl_df.loc[_prv_date, 'MA'] + self.hedge_pnl_df.loc[_date, 'delta_AA'] - self.hedge_pnl_df.loc[_date, 'TC']
    #         if tmp_MA > self.hedge_pnl_df.loc[_date, 'max_AA']:
    #             # delta_CA > 0：从MA取钱到CA存储，下一期会收获利息，使得当前的MA处于init_AA的水平
    #             delta_CA = tmp_MA - self.hedge_pnl_df.loc[_date, 'init_AA']
    #         elif tmp_MA < self.hedge_pnl_df.loc[_date, 'min_AA']:
    #             # delta_CA < 0：从CA取钱补充，下一期会扣除利息，使得当前的MA处于init_AA的水平
    #             delta_CA = tmp_MA - self.hedge_pnl_df.loc[_date, 'init_AA']
    #         else:
    #             delta_CA = 0
    #         self.hedge_pnl_df.loc[_date, 'MA'] = tmp_MA - delta_CA
    #         self.hedge_pnl_df.loc[_date, 'IC'] = self.hedge_pnl_df.loc[_prv_date, 'CA'] * self.r_ic
    #         self.hedge_pnl_df.loc[_date, 'CA'] = self.hedge_pnl_df.loc[_prv_date, 'CA'] + self.hedge_pnl_df.loc[_date, 'IC'] + delta_CA
    #
    #     # self.hedge_pnl_df['delta_nav'] = self.hedge_pnl_df['TC'] + self.hedge_pnl_df['IC']
    #     # self.hedge_pnl_df['nav'] = self.hedge_pnl_df['delta_nav'].cumsum()
    #     return self.hedge_pnl_df
