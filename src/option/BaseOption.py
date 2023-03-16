# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from loguru import logger
from abc import abstractmethod

# 读取数据
from src.data.Container import _container


class BaseOption:
    """期权基类

    所有方法列表
    ----------
        reset_paras:
            初始化参数设置，都设置为none，look_back_num设置为10个交易日
        set_paras_by_dict:
            根据传入的字典初始化合约级参数设置
        calculate_base_paras:
            计算与期限相关的参数，运行以下四个方法
        calculate_trade_dates:
            设置trade_dates和look_back_dates
        get_spot_price:
            读取保存look_back_dates时间段内的股价数据
        calculate_vols:
            计算trade_dates窗口期内的vol
        calculate_basic_paras:
            计算basic_paras_df里'sigma', 'left_days', 'left_times', 'sigma_T', 'delta_s', 'delta_r'
        pnl_decomposition:
            计算option端的pnl拆解，传入greeks，计算好各个pnl保存在decompose_df中

    """

    greek_columns = ['delta', 'gamma', 'vega', 'theta', 'option_value']
    underlying_asset_base_type = ['stock', 'futures']
    basic_paras_columns = ['sigma', 'left_days', 'left_times', 'sigma_T', 'stock_price', 'option_price', 'exercise']

    def __init__(self, underlying_asset: Optional[str] = None,
                 underlying_code: Optional[str] = None, strike_date: Optional[str] = None,
                 maturity_date: Optional[str] = None, K: float = 1,
                 look_back_num: float = 10, sigma: Optional[float] = None, r: float = 0.02,
                 *args, **kwargs):  # 合约级的参数需要初始化输入，随着交易日时变的参数后面再计算
        if underlying_asset in self.underlying_asset_base_type:
            self.underlying_asset = underlying_asset
        else:
            raise ValueError('Invalid underlying_asset!')
        self.underlying_code = underlying_code
        self.strike_date = pd.to_datetime(strike_date)
        self.maturity_date = pd.to_datetime(maturity_date)
        self.K = K
        self.look_back_num = look_back_num
        self.sigma = sigma
        self.r = r
        self.greek_df: pd.DataFrame = None
        self.set_all_trade_dates()
        logger.info(
            f'Initialize {self.__class__.__name__} with underlying: {self.underlying_asset}-{self.underlying_code} '
            f'start_date: {strike_date}, maturity_date: {maturity_date}, K: {self.K}')

    def set_all_trade_dates(self):
        self.all_trade_dates = _container.get_data(self.underlying_asset, self.underlying_code).index.tolist()

    def calculate_base_paras(self):
        self.calculate_trade_dates()
        self.get_spot_prices()
        self.calculate_vols()
        self.calculate_basic_paras()

    def calculate_trade_dates(self):
        """计算起息日到期日和要用于计算vol的时间窗口

        trade_dates时间段为起息日到到期日，look_back_dates为trade_dates再加上往前look_back_num个交易日的时间窗口
        """
        self.start_idx = self.all_trade_dates.index(self.strike_date)
        self.end_idx = self.all_trade_dates.index(self.maturity_date) + 1
        self.trade_dates = self.all_trade_dates[self.start_idx:self.end_idx]
        self.trade_dates_length = len(self.trade_dates)

    def get_spot_prices(self):
        """提取look_bakck_dates内的股票价格

        从single_stock_data里提取对应股票代码的CLOSE列收盘价，spot_price是look_back_dates内的收盘价
        """
        if self.underlying_code is None:
            logger.error('标的资产代码未设定')
            return -1
        if self.sigma is not None:
            self.spot_price = _container.get_data(self.underlying_asset, self.underlying_code).loc[
                self.trade_dates, 'CLOSE']
        else:
            self.spot_price = _container.get_data(self.underlying_asset, self.underlying_code).loc[
                self.all_trade_dates, 'CLOSE']
        self.ISP = self.spot_price.loc[self.strike_date]
        # 加上折现因子
        # self.spot_price = single_stock_data.get_data('stock', '300015.SZ').loc[self.look_back_dates, 'CLOSE']\
        #     *single_stock_data.get_data('stock', '300015.SZ').loc[self.look_back_dates, 'ADJFACTOR']

    def calculate_vols(self):
        """根据look_back_dates内的股票价格计算trade_dates内的vol

        用历史波动率表示隐含波动率
        """
        self.percent_change = self.spot_price.pct_change()
        # 移动窗口的长度是look_back_num的长度，计算std
        if self.sigma is None:
            self.volatility = self.percent_change.rolling(self.look_back_num).std()[self.look_back_num:].loc[
                                  self.trade_dates] * np.sqrt(252)
        else:
            self.volatility = pd.Series(index=self.trade_dates, data=self.sigma)

    def calculate_basic_paras(self):
        """计算sigma，期限以及年化的剩余期限

        sigma = volatility,用历史波动率代替隐含波动率
        left_days计算还有多少个交易日到期
        left_times计算年化的left_days
        sigma_T = sigma * sqrt(left_times) 对应用于之后BS公式计算
        """
        logger.info(f'{self.__class__.__name__} calculate basic paras')
        self.price_df = pd.DataFrame(data=None, columns=self.basic_paras_columns)
        self.price_df.loc[:, 'sigma'] = self.volatility.dropna()
        self.price_df.loc[:, 'sigma_2'] = self.price_df.loc[:, 'sigma'] ** 2
        self.price_df.loc[:, 'left_days'] = np.linspace(self.trade_dates_length - 1, 0, self.trade_dates_length)
        self.price_df.loc[:, 'left_times'] = self.price_df.loc[:, 'left_days'] / 252
        self.price_df.loc[:, 'sigma_T'] = self.price_df.loc[:, 'sigma'] * np.sqrt(
            self.price_df.loc[:, 'left_times'])
        self.price_df.loc[:, 'stock_price'] = self.spot_price.loc[self.trade_dates]
        self.price_df.loc[:, 'Delta_S'] = self.price_df.loc[:, 'stock_price'].diff().fillna(0)
        self.price_df.loc[:, 'Delta_r'] = self.price_df.loc[:, 'Delta_S'] / self.price_df.loc[:,
                                                                            'stock_price']
        self.price_df.loc[:, 'exercise'] = 0

    @abstractmethod
    def calculate_option_price(self):
        """计算price的参数，比如EuropeanOption里面的d1，d2等等
        """
        pass

    @abstractmethod
    def calculate_option_greeks(self):
        """计算期权对应的希腊字母
        """
        pass

    def pnl_decompose(self, greek_df):
        logger.info(f'{self.__class__.__name__} pnl decompose')
        self.decompose_df = pd.DataFrame(data=None,
                                         columns=['option_pnl', 'delta_pnl', 'gamma_pnl', 'theta_pnl', 'disc_pnl',
                                                  'carry_pnl', 'residual'])
        self.decompose_df['option_pnl'] = greek_df.loc[:, 'option_value'].diff().fillna(0)
        self.decompose_df['delta_pnl'] = greek_df.loc[:, 'delta'] * self.price_df.loc[:, 'Delta_S']
        self.decompose_df['gamma_pnl'] = greek_df.loc[:, 'gamma'] * np.power(
            self.price_df.loc[:, 'Delta_S'], 2) / 2
        self.decompose_df['theta_pnl'] = -greek_df.loc[:, 'gamma'] * np.power(self.price_df['stock_price'],
                                                                              2) * self.price_df.loc[:,
                                                                                   'sigma_2'] * 1 / 252 / 2
        self.decompose_df['disc_pnl'] = self.r * greek_df.loc[:, 'option_value'] / 252
        self.decompose_df['carry_pnl'] = -greek_df.loc[:, 'delta'] * self.price_df[
            'stock_price'] * self.r * 1 / 252
        self.decompose_df['residual'] = self.decompose_df.loc[:, 'option_pnl'] - self.decompose_df.loc[:,
                                                                                 'delta_pnl'] - self.decompose_df.loc[:,
                                                                                                'gamma_pnl'] \
                                        - self.decompose_df.loc[:, 'theta_pnl'] - self.decompose_df.loc[:,
                                                                                  'disc_pnl'] - self.decompose_df.loc[:,
                                                                                                'carry_pnl']
        return self.decompose_df

    def decomposition_visualize(self, decompose_df):
        df_plot = decompose_df.copy()
        df_plot.index = np.linspace(start=0, stop=len(df_plot) / 252, num=len(df_plot))
        fig, ax1 = plt.subplots(figsize=(15, 10))
        ax1.set_xlabel('Time (Unit: year)')
        ax1.set_ylabel('Value (Unit: Yuan)')
        df_plot.loc[:,
        ['option_pnl', 'delta_pnl', 'gamma_pnl', 'theta_pnl', 'disc_pnl', 'carry_pnl', 'residual']].cumsum().plot(
            ax=ax1)
        plt.show()

    @staticmethod
    def CRR(S0: float, r: float, K: float, sigma: float, T: float, type: str = 'Call'):
        N = int(250 * T)
        if N < 0.0:
            return float("nan")
        if N == 0.0:
            if type == 'Call':
                return max(S0 - K, 0)
            else:
                return max(K - S0, 0)
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        mu = np.array([list(range(N)) for _ in range(N)])
        md = mu.T
        mu = mu - md
        if type == 'Call':
            V = np.maximum(0, S0 * (u ** mu) * (d ** md) - K)
        else:
            V = np.maximum(0, K - S0 * (u ** mu) * (d ** md))
        for ii in range(N - 1, 0, -1):
            V[:ii, ii - 1] = np.maximum(V[:ii, ii - 1], (V[:ii, ii] * p + V[1:ii + 1, ii] * (1 - p)) / np.exp(r * dt))
        price = V[0, 0]
        return price
