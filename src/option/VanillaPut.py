#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
from typing import Optional
from src.data.Container import _container
from scipy import stats as st
from .BaseOption import BaseOption
from loguru import logger


class VanillaPut(BaseOption):
    """ 和call option方法类似，只是计算的公式不同

    gamma和vega是一样的
    """

    def __init__(self, underlying_asset: Optional[str] = None,
                 underlying_code: Optional[str] = None, strike_date: Optional[str] = None,
                 maturity_date: Optional[str] = None, K: float = 1,
                 look_back_num: float = 10, sigma: Optional[float] = None, r: float = 0.02,
                 *args, **kwargs):
        super().__init__( underlying_asset, underlying_code, strike_date,
                         maturity_date, K, look_back_num, sigma, r, *args, **kwargs)

    def calculate_option_price(self):
        logger.info(f'{self.__class__.__name__} calculate option price')
        self.calculate_base_paras()
        self.basic_paras_df.loc[:, 'd1'] = (np.log(
            self.basic_paras_df.loc[:, 'stock_price'] / self.K) + self.r * self.basic_paras_df.loc[:,
                                                                           'left_times']) / self.basic_paras_df.loc[:,
                                                                                            'sigma_T'] + 0.5 * self.basic_paras_df.loc[
                                                                                                               :,
                                                                                                               'sigma_T']
        self.basic_paras_df.loc[:, 'd2'] = self.basic_paras_df.loc[:, 'd1'] - self.basic_paras_df.loc[:, 'sigma_T']
        self.basic_paras_df.loc[:, 'nd1'] = st.norm.pdf(self.basic_paras_df.loc[:, 'd1'])
        self.basic_paras_df.loc[:, 'Nd1'] = st.norm.cdf(self.basic_paras_df.loc[:, 'd1'])
        self.basic_paras_df.loc[:, 'Nd2'] = st.norm.cdf(self.basic_paras_df.loc[:, 'd2'])
        self.basic_paras_df.loc[:, 'Delta_S'] = self.basic_paras_df.loc[:, 'stock_price'].diff().fillna(0)
        self.basic_paras_df.loc[:, 'Delta_r'] = self.basic_paras_df.loc[:, 'Delta_S'] / self.basic_paras_df.loc[:,
                                                                                        'stock_price']
        self.basic_paras_df.loc[:, 'option_price'] = (self.basic_paras_df.loc[:, 'stock_price'] * (self.basic_paras_df.loc[:, 'Nd1'] - 1)
                                                      - self.K * np.exp(
            -self.r * self.basic_paras_df.loc[:, 'left_times']) * (self.basic_paras_df.loc[:, 'Nd2'] - 1)
                                                      )
        self.basic_paras_df.loc[self.basic_paras_df.index[-1], 'exercise'] = int(self.basic_paras_df['stock_price'][-1] < self.K)


    def calculate_option_greeks(self):
        logger.info(f'{self.__class__.__name__} calculate option greeks')
        self.calculate_option_price()
        self.greek_df = pd.DataFrame(index=self.trade_dates, columns=self.greek_columns)
        self.greek_df.loc[:, 'option_value'] = self.basic_paras_df.loc[:, 'option_price']
        self.greek_df.loc[:, 'delta'] = self.basic_paras_df.loc[:, 'Nd1'] - 1
        self.greek_df.loc[:, 'gamma'] = self.basic_paras_df.loc[:, 'nd1'] / (
                    self.basic_paras_df.loc[:, 'stock_price'] * self.basic_paras_df.loc[:, 'sigma_T'])
        self.greek_df.loc[:, 'vega'] = self.greek_df.loc[:, 'gamma'] * self.basic_paras_df.loc[:,
                                                                       'stock_price'] * self.basic_paras_df.loc[:,
                                                                                        'stock_price'] * self.basic_paras_df.loc[
                                                                                                         :, 'sigma_T']
        self.greek_df.loc[:, 'theta'] = -(self.basic_paras_df.loc[:, 'stock_price'] * self.basic_paras_df.loc[:,
                                                                                      'nd1'] * self.basic_paras_df.loc[
                                                                                               :, 'sigma'] / (
                                                      2 * np.sqrt(self.basic_paras_df.loc[:, 'left_times']))) \
                                        + self.r * self.K * np.exp(
            -self.r * self.basic_paras_df.loc[:, 'left_times']) * self.basic_paras_df.loc[:, 'Nd2']
        # self.greek_df.loc[:,'cash_delta'] = self.greek_df.loc[:,'delta']*self.basic_paras_df.loc[:,'stock_price']*self.notional/self.ISP    #这样不考虑股票的最小交易单位
        # self.greek_df.loc[:, 'cash_delta'] = self.greek_df.loc[:, 'delta'] * self.notional * self.option_position # 这样不考虑股票的最小交易单位
        # self.greek_df.loc[:, 'cash_gamma'] = self.greek_df.loc[:, 'gamma'] * self.notional * self.basic_paras_df.loc[:,
        #                                                                                      'stock_price'] / 100 * self.option_position

    def get_basic_para_df(self):
        self.calculate_option_price()
        return self.basic_paras_df

    def get_greek_df(self):
        self.calculate_option_greeks()
        return self.greek_df

    def get_pnl_decompose_df(self):
        self.calculate_option_greeks()
        self.pnl_decompose(self.greek_df)
        return self.decompose_df
