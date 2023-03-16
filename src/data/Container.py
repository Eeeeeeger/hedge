#!/usr/bin/env python
# coding=utf-8

import pandas as pd

global _container

class Container:
    def __init__(self):
        self.data = {'stock': {},
                     'futures': {}
                     }

    def get_data(self, asset: str, code: str):
        return self.data[asset][code]

    def add_data(self, asset: str, code: str, data: pd.DataFrame):
        data.index = pd.to_datetime(data.index)
        self.data[asset][code] = data


global _container
_container = Container()
