import numpy as np
import pandas as pd
from .save_utils import *


def trading_days(start_date: str or pd.Timestamp, end_date: str or pd.Timestamp):
    days = pd.Index(load_pickle('./data/dates/trading_days.pkl'))
    return days[(days > start_date) & (days <= end_date)].tolist()
