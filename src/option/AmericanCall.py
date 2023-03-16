#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats as st
from .BaseOption import BaseOption
from src.data.Container import _container
from loguru import logger

class AmericanCall(BaseOption):
    """AmericanCall
    """

    def __init__(self, underlying_asset: Optional[str] = None,
                 underlying_code: Optional[str] = None, strike_date: Optional[str] = None,
                 maturity_date: Optional[str] = None, K: float = 1,
                 look_back_num: float = 10, sigma: Optional[float] = None, r: float = 0.02,
                 *args, **kwargs):
        super().__init__(underlying_asset, underlying_code, strike_date,
                         maturity_date, K, look_back_num, sigma, r, *args, **kwargs)

    def calculate_option_price(self):
        logger.info(f'{self.__class__.__name__} calculate option price')
        self.calculate_base_paras()
        self.price_df.loc[:, 'Delta_S'] = self.price_df.loc[:, 'stock_price'].diff().fillna(0)
        self.price_df.loc[:, 'Delta_r'] = self.price_df.loc[:, 'Delta_S'] / self.price_df.loc[:, 'stock_price']
        self.price_df.loc[:, 'option_price'] = self.price_df.apply(
            lambda x: self.CRR(x['stock_price'], self.r, self.K, x['sigma'], x['left_times'], 'Call'), axis=1)
        self.price_df.loc[:, 'exercise'] = self.price_df.apply(
            lambda x: x['option_price'] == x['stock_price'] - self.K, axis=1).astype(int)

    def calculate_option_greeks(self):
        logger.info(f'{self.__class__.__name__} calculate option greeks')
        self.calculate_option_price()
        self.greek_df = pd.DataFrame(index=self.trade_dates, columns=self.greek_columns)
        self.greek_df.loc[:, 'option_value'] = self.price_df.loc[:, 'option_price']
        self.greek_df.loc[:, 'delta'] = self.price_df.apply(lambda x: (self.CRR(x['stock_price'] + 0.05,
                                                                                      self.r,
                                                                                      self.K,
                                                                                      x['sigma'],
                                                                                      x['left_times'],
                                                                                      'Call')
                                                                             - self.CRR(x['stock_price'] - 0.05,
                                                                                        self.r,
                                                                                        self.K,
                                                                                        x['sigma'],
                                                                                        x['left_times'],
                                                                                        'Call')
                                                                             ) / 0.1,
                                                            axis=1)

        self.greek_df.loc[:, 'gamma'] = self.price_df.apply(lambda x: (self.CRR(x['stock_price'] + 0.05,
                                                                                self.r,
                                                                                self.K,
                                                                                x['sigma'],
                                                                                x['left_times'],
                                                                                'Call')
                                                                        - 2 * x['option_price']
                                                                        + self.CRR(x['stock_price'] - 0.05,
                                                                                   self.r,
                                                                                   self.K,
                                                                                   x['sigma'],
                                                                                   x['left_times'],
                                                                                   'Call')
                                                                        ) / 0.05 / 0.05,
                                                            axis=1)

    def get_price_df(self):
        self.calculate_option_price()
        return self.price_df

    def get_greek_df(self):
        self.calculate_option_greeks()
        return self.greek_df

    def get_pnl_decompose_df(self):
        self.calculate_option_greeks()
        self.pnl_decompose(self.greek_df)
        return self.decompose_df
