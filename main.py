import matplotlib.pyplot as plt

from src.data.Container import _container
from src.data.Market import Market
from src.data.Simulation import Simulation
from src.option import *
from src.strategy.HedgeAllStrategy import HedgeAllStrategy
from src.backtest.backtest import Backtest

import pandas as pd
import numpy as np
import warnings

pd.set_option('display.max_columns', 100)
pd.set_option('expand_frame_repr', False)
warnings.filterwarnings('ignore')

start_date = '2020-01-01'
end_date = '2021-01-01'
data = Market.load_data_from_wind('000905.SH', ['close'], start_date, end_date)
trading_days = data.index.tolist()

'''
Data Generation
'''
for ii in range(10):
    para_dt = {'mid_pos': 11, 'St': 10, 'mu': 0, 'sigma': 0.5, 'T': 1, 'num': len(trading_days), 'random_state': ii}

    df = pd.DataFrame(index=trading_days,
                      columns=['CLOSE'],
                      data=Simulation.exact_bs_sim_mid(**para_dt))

    _container.add_data('stock', f'sim_{ii}', df)

'''
Portfolio Construction
'''
# op = OptionPortfolio()
# for code in [f'sim_{ii}' for ii in range(10)]:
#     paras_dt1 = {'option_type': 'VanillaPut', 'underlying_asset': 'stock',
#                 "underlying_code": code, "strike_date": trading_days[10],
#                 'maturity_date': trading_days[-1], "option_position": -5000,
#                 'K': 10, 'look_back_num': 10, 'r': 0.02}
#     # call1 = VanillaCall(**paras_dt1)
#     # call1.calculate_option_greeks()
#
#     op.add_option_list(paras_dt1)

'''
Hedge and Backtest
'''
# hedge_strategy = HedgeAllStrategy('stock', ['sim_0'])
# # :TODO hedge_asset and underlying asset?
# bt = Backtest()
# bt.set_strategy(hedge_strategy)
# bt.set_portfolio(op)
#
#
# df_backtest = [each for each in bt.run_backtest()]
# final_pnl = [df['final_pnl'][-1] for df in df_backtest]
# pd.concat([_container.get_data('stock', f'sim_{ii}') for ii in range(1, 10, 2)], axis=1).plot(figsize=(20, 15))
# plt.legend(final_pnl[1:10:2])
# plt.show()
# print([df['option_value'][0] for df in df_backtest])
#


paras_dt2 = {'option_type': 'VanillaCall', 'underlying_asset': 'stock',
             "underlying_code": "sim_0", "strike_date": "2020-01-16",
             'maturity_date': '2020-12-25', "option_position": -100,
             'K': 12, 'look_back_num': 10, 'r': 0.02}
call2 = VanillaCall(**paras_dt2)
call2.calculate_option_greeks()
print(pd.concat([call2.basic_paras_df,call2.greek_df['delta']],axis=1))


paras_dt2 = {'option_type': 'AmericanCall', 'underlying_asset': 'stock',
             "underlying_code": "sim_0", "strike_date": "2020-01-16",
             'maturity_date': '2020-12-25', "option_position": -100,
             'K': 12, 'look_back_num': 10, 'r': 0.02}
call2 = AmericanCall(**paras_dt2)
call2.calculate_option_greeks()
print(pd.concat([call2.basic_paras_df,call2.greek_df['delta']],axis=1))

paras_dt2 = {'option_type': 'VanillaPut', 'underlying_asset': 'stock',
             "underlying_code": "sim_0", "strike_date": "2020-01-16",
             'maturity_date': '2020-12-25', "option_position": -100,
             'K': 12, 'look_back_num': 10, 'r': 0.02}
call2 = VanillaPut(**paras_dt2)
call2.calculate_option_greeks()
print(pd.concat([call2.basic_paras_df,call2.greek_df['delta']],axis=1))


paras_dt2 = {'option_type': 'AmericanPut', 'underlying_asset': 'stock',
             "underlying_code": "sim_0", "strike_date": "2020-01-16",
             'maturity_date': '2020-12-25', "option_position": -100,
             'K': 12, 'look_back_num': 10, 'r': 0.02}
call2 = AmericanPut(**paras_dt2)
call2.calculate_option_greeks()
print(pd.concat([call2.basic_paras_df,call2.greek_df['delta']],axis=1))

# op.add_option_list(paras_dt2)
