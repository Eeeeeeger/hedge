#!/usr/bin/env python
# coding=utf-8

import pandas as pd
from pathlib import Path
from WindPy import *


class Market:
    @staticmethod
    def load_data_from_wind(code: str, fields_list: list = ['open', 'close', 'volume', 'adjfactor', 'div_capitalization', 'div_stock'], start_date: str = '2010-01-01', end_date: str = '2011-01-01'):
        w.start()
        err, data = w.wsd(
            code,
            fields_list,
            start_date,
            end_date,
            Days="Trading",
            Period="D",
            usedf=True,
        )
        w.stop()
        if data.empty:
            raise ValueError('empty data')
        else:
            data.index = pd.to_datetime(data.index)
            return data

    @staticmethod
    def load_data_from_file(file_path: Path, start_date: str = '2010-01-01', end_date: str = '2010-01-01'):
        if file_path.suffix == '.csv':
            temp = pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_path.suffix == '.pkl':
            temp = pd.read_pickle(file_path)
        else:
            raise ValueError('wrong input file path')
        temp.index = pd.to_datetime(temp.index)
        return temp.loc[start_date:end_date]