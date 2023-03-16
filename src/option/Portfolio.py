# !/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .AmericanCall import AmericanCall
from .AmericanPut import AmericanPut
from .EuropeanCall import EuropeanCall
from .EuropeanPut import EuropeanPut
from src.data.Container import _container


class OptionPortfolio():
    """
    portfolio在add_option_list()里初始化期权合约级参数
    关于日期的属性：
        all_trade_dates: 记录股价数据中所有的交易日
            - 类型: list
        trade_dates: 起息日到到期日
            - 类型: list
        look_back_dates: trade_date加上算vol的一小段时间窗口的时间
            - 类型: list
    """

    def __init__(self):
        self.decompose_df = {}
        self.greek_df = {}
        self.price_df = {}
        self.option_list = []
        self.fixbasiclist = ['sigma', 'left_days', 'left_times', 'sigma_T', 'stock_price', 'sigma_2', 'Delta_S',
                             'Delta_r']

    def add_option_list(self, para_dict):
        self.option_type = para_dict.get('option_type')

        # 初始化portfolio的合约级参数，这里以portfolio只有一种European为例
        self.underlying_asset = para_dict.get('underlying_asset')
        self.underlying_code = para_dict.get('underlying_code')
        self.strike_date = para_dict.get('strike_date')
        self.maturity_date = para_dict.get('maturity_date')
        self.K = para_dict.get('K')
        self.look_back_num = para_dict.get('look_back_num')
        self.option_position = para_dict.get('option_position', 1)
        self.set_all_trade_dates()

        if self.option_type in ['EuropeanCall', 'EuropeanPut', 'AmericanCall', 'AmericanPut']:
            option = eval(self.option_type)(**para_dict)
            option.calculate_option_greeks()
            self.option_list.append({'option_object': option, 'option_position': self.option_position})
        # elif self.option_type == "BullCallSpread":
        #     option1 = EuropeanCall()
        #     parameter = para_dict.copy()
        #     parameter['K'] = min(para_dict.get('K'))
        #     parameter['KS_ratio'] = min(para_dict.get('KS_ratio'))
        #     option1.set_paras_by_dict(parameter)
        #     self.option_list.append({'option_object': option1, 'option_position': 1})
        #     option2 = EuropeanCall()
        #     parameter = para_dict.copy()
        #     parameter['K'] = max(para_dict.get('K'))
        #     parameter['KS_ratio'] = max(para_dict.get('KS_ratio'))
        #     option2.set_paras_by_dict(parameter)
        #     self.option_list.append({'option_object': option2, 'option_position': -1})

    def set_all_trade_dates(self):
        self.all_trade_dates = _container.get_data(self.underlying_asset, self.underlying_code).index.tolist()

    def calculate_option_greeks(self):
        for i, element in enumerate(self.option_list):
            self.greek_df[i] = element.get('option_object').greek_df * element.get('option_position')
            self.price_df[i] = element.get('option_object').price_df

    def pnl_decompose(self):
        # TODO: American Greeks are not all finished
        self.calculate_option_greeks()
        for i, element in enumerate(self.option_list):
            self.decompose_df[i] = element.get('option_object').get_pnl_decompose_df() * element.get('option_position')

    def calculate_trade_dates(self):
        """计算起息日到期日和要用于计算vol的时间窗口

        trade_dates时间段为起息日到到期日，look_back_dates为trade_dates再加上往前look_back_num个交易日的时间窗口
        """
        self.start_idx = self.all_trade_dates.index(self.strike_date)
        self.end_idx = self.all_trade_dates.index(self.maturity_date) + 1
        self.trade_dates = self.all_trade_dates[self.start_idx:self.end_idx]
        self.look_back_date = self.all_trade_dates[self.start_idx - self.look_back_num]
        self.look_back_dates = self.all_trade_dates[self.start_idx - self.look_back_num:self.end_idx]
        self.trade_dates_length = len(self.trade_dates)

    def get_trade_dates(self):
        self.calculate_trade_dates()
        return self.trade_dates

    def get_greek_df(self):
        self.calculate_option_greeks()
        return self.greek_df

    def get_decomposition_df(self):
        self.pnl_decompose()
        return self.decompose_df

    def decomposition_visualize(self):
        df_plot = self.get_decomposition_df().copy()
        df_plot.index = np.linspace(start=0, stop=len(df_plot) / 252, num=len(df_plot))
        fig, ax1 = plt.subplots(figsize=(15, 10))
        ax1.set_xlabel('Time (Unit: year)')
        ax1.set_ylabel('Value (Unit: Yuan)')
        df_plot.loc[:,
        ['option_pnl', 'delta_pnl', 'gamma_pnl', 'theta_pnl', 'disc_pnl', 'carry_pnl', 'residual']].cumsum().plot(
            ax=ax1)
        plt.show()
