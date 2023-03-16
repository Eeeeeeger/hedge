from src.option.Portfolio import OptionPortfolio as OP
from src.strategy import *
from loguru import logger
# from ..backtest.reportTemplate import ReportTemplate
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListItem, ListFlowable


class Backtest:
    def __init__(self):
        self.fee_rate = 0.0005
        self.r = 0.02
        self.strategy = None
        self.op = None

    def set_strategy(self, strategy: BaseStrategy):
        self.strategy = strategy

    def set_portfolio(self, portfolio: OP):
        self.op = portfolio
        self.op.calculate_option_greeks()

    def run_backtest(self):
        for i, element in enumerate(self.op.option_list):
            self.df_backtest = self.op.greek_df[i].copy()
            self.df_backtest.loc[:, 'name'] = element["option_object"].__class__.__name__
            self.df_backtest.loc[:, 'stock_price'] = self.op.price_df[i].loc[:, 'stock_price']
            self.df_backtest.loc[:, 'exercise'] = self.op.price_df[i].loc[:, 'exercise']
            self.df_backtest.loc[:, 'option_price'] = self.op.price_df[i].loc[:, 'option_price']
            # :TODO it should be hedge asset; considering multi assets
            self.df_backtest.loc[:, 'stock_position'] = self.strategy.cal_position(self.op.greek_df[i]).values
            self.df_backtest.loc[:, 'stock_value'] = self.df_backtest.loc[:, 'stock_price'] * self.df_backtest.loc[:,
                                                                                              'stock_position']

            self.df_backtest.loc[:, 'purchase_stock_position'] = self.df_backtest.loc[:, 'stock_position'].diff().fillna(self.df_backtest.loc[:, 'stock_position'][0])
            self.df_backtest.loc[:, 'purchase_stock_value'] = self.df_backtest.loc[:, 'purchase_stock_position'] * self.df_backtest.loc[:, 'stock_price']
            # self.df_backtest.loc[:, 'delta_stock_value'] = (self.df_backtest.loc[:, 'stock_position'].shift() * self.df_backtest.loc[:, 'stock_price'].diff()).fillna(-self.df_backtest.loc[:, 'stock_value'][0])

            self.df_backtest.loc[:, 'trading_pnl'] = self.df_backtest.loc[:, 'purchase_stock_value'].abs() * self.fee_rate
            self.df_backtest.loc[:, 'trading_value'] = self.df_backtest.loc[:, 'trading_pnl'].cumsum()
            self.df_backtest.loc[:, 'cash_account'] = -self.df_backtest['option_value'][0] - self.df_backtest.loc[:, 'purchase_stock_value'].cumsum()-self.df_backtest.loc[:, 'trading_value']
            self.df_backtest.loc[:, 'interest_pnl'] = (self.df_backtest.loc[:, 'cash_account'] * self.r / 252).shift().fillna(0)
            self.df_backtest.loc[:, 'interest_value'] = self.df_backtest.loc[:, 'interest_pnl'].cumsum()
            self.df_backtest.loc[:, 'net_cash_account'] = self.df_backtest.loc[:, 'cash_account'] + self.df_backtest.loc[:, 'interest_value']

            self.df_backtest.loc[:, 'exercise_value'] = 0
            # exercise or not
            if (self.df_backtest['exercise'] > 0).any():
                exercise_date = self.df_backtest['exercise'].idxmax()
                self.df_backtest = self.df_backtest.loc[self.df_backtest.index <= exercise_date]
                logger.debug(f'{element["option_object"].__class__.__name__} exercise at {exercise_date}')

                if 'Call' in element["option_object"].__class__.__name__:
                    self.df_backtest.loc[exercise_date, 'exercise_value'] = element['option_object'].K * element['option_position']
                else:
                    self.df_backtest.loc[exercise_date, 'exercise_value'] = - element['option_object'].K * element['option_position']
            else:
                logger.debug(f'{element["option_object"].__class__.__name__} not exercise')
            self.df_backtest.loc[:, 'final_pnl'] = self.df_backtest.loc[:, 'net_cash_account'] - self.df_backtest.loc[:, 'exercise_value']

            self.df_backtest = self.df_backtest.round(4)
            self.df_backtest.loc[:, 'trade_dummy'] = 1
            self.df_backtest.loc[self.df_backtest.loc[:, 'stock_position'].diff() == 0, 'trade_dummy'] = 0
            yield self.df_backtest


    # def hedge_pnl_analysis(self):
    #     self.trade_dates = pd.to_datetime(self.df_backtest.index.values)
    #     self.hedge_summary()
    #     self.hedge_pnl_plot()
    #     # self.pnl_decomposition_plot()

    # def hedge_summary(self):
    #     self.hedge_pnl_summary = pd.Series(self.df_backtest.loc[:, ['total_nav', 'stock_pnl']].sum().values, index=['total_pnl', 'stock_pnl'])
    #     self.hedge_pnl_summary['option_pnl'] = self.option_fee-self.df_backtest.loc[:, 'option_pnl'].sum()
    #     self.hedge_pnl_summary['trading_cost'] = self.df_backtest.loc[:, 'trading_cost'].sum()
    #     self.hedge_pnl_summary['min_cash'] = self.df_backtest.loc[:, 'cash_account'].min()
    #     self.hedge_pnl_summary['max_drawdown'] = self.cal_MDD(self.df_backtest.loc[:, 'total_nav'])
    #     self.hedge_pnl_summary = self.hedge_pnl_summary.round(2)
    #     self.decomposition_summary = pd.Series(self.df_backtest.loc[:, ['gamma_pnl', 'vega_pnl', 'theta_pnl', 'higher_order_pnl',
    #                                     'unhedged_pnl', 'trading_cost', 'total_nav']].sum().values, index=['total_gamma_pnl',
    #                                     'total_vega_pnl', 'total_theta_pnl', 'total_higher_order_pnl', 'total_unhedged_pnl',
    #                                     'total_trading_cost', 'total_profit'])
    #     self.decomposition_summary = self.decomposition_summary.round(decimals=2)

    # def hedge_pnl_plot(self):
    #     df_plot = self.df_backtest.loc[:, ['option_value', 'stock_value', 'cash_account', 'total_nav',
    #                                        'trade_dummy', 'stock_position']].copy()
    #     df_plot.loc[:, 'option_value'] = -df_plot.loc[:, 'option_value']
    #     fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.5, 4]}, figsize=(20, 12))
    #     ax0.set_axis_off()  # 除去坐标轴
    #     table = ax0.table(cellText=self.hedge_pnl_summary.values.reshape(1,-1),
    #                       bbox=(0, 0, 1, 1),  # 设置表格位置， (x0, y0, width, height)
    #                       rowLoc='right',  # 行标题居中
    #                       cellLoc='right',
    #                       colLabels=self.hedge_pnl_summary.index.tolist(),  # 设置列标题
    #                       colLoc='right',  # 列标题居中
    #                       edges='open'  # 不显示表格边框
    #                       )
    #     table.set_fontsize(13)
    #     ax1.set_xlabel('Date')
    #     ax1.set_ylabel('Value')
    #     ax1.plot(self.trade_dates, df_plot.loc[:, 'stock_value'] + df_plot.loc[:, 'cash_account'], linewidth=0.5,
    #              color='blue', label='hedge_value')
    #     ax1.plot(self.trade_dates, df_plot.loc[:, 'option_value'], linewidth=1, color='orange', label='option_value')
    #     ax1.plot(self.trade_dates, df_plot.loc[:, 'total_nav'], linewidth=1, color='red', label='total_value')
    #     for tradeday in pd.to_datetime(df_plot[df_plot['trade_dummy'] == 1].index):
    #         ax1.axvline(tradeday, linewidth=0.5, color='lightgrey', zorder=0)
    #     # ax1.legend(bbox_to_anchor=(1.025, 0.95), loc='upper left', borderaxespad=1, fontsize=8)
    #     ax1_lim = [df_plot.values.min(), df_plot.values.max()]
    #     # ax1.set_ylim(ax1_lim[0]-0.1*(ax1_lim[1]-ax1_lim[0]), ax1_lim[1]+(ax1_lim[1]-ax1_lim[0]))
    #     ax2 = ax1.twinx()
    #     ax2.plot(self.trade_dates, df_plot.loc[:, 'stock_position'], alpha=0.3, linewidth=1, color='green', drawstyle='steps-post', label='stock_position(右轴)')
    #     ax2_lim = [min(df_plot.loc[:, 'stock_position']), max(df_plot.loc[:, 'stock_position'])]
    #     # ax2.set_ylim(2*ax2_lim[0]-ax2_lim[1], ax2_lim[1]*1.1)
    #     # ax2.legend(bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=1, fontsize=8)
    #     h1, l1 = ax1.get_legend_handles_labels()
    #     h2, l2 = ax2.get_legend_handles_labels()
    #     plt.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.005, 1), fontsize=10, borderaxespad=1, loc='lower right', ncol=1)
    #     strategy_name = re.findall(r'\'(.*?)\'', str(type(self.strategy)))[0].split('.')[-1]
    #     ax1.set_title('收益情况图')
    #     # plt.show()
    #     fig.savefig('../03_img/股票对冲回测.jpg')
    #
    # def pnl_decomposition_plot(self):
    #     df_plot = self.df_backtest.loc[:, ['option_pnl', 'delta_pnl', 'gamma_pnl', 'theta_pnl', 'vega_pnl',
    #                                        'higher_order_pnl', 'trading_cost']].cumsum()
    #     df_plot.loc[:, ['position_rate', 'upper_bound', 'lower_bound']] = self.strategy.get_hedge_df().loc[:, ['position_rate', 'up_bound', 'low_bound']].values
    #     fig, ax1 = self.init_canvas([0.06, 0.07, 0.73, 0.85])
    #     ax1.set_xlabel('Date')
    #     ax1.set_ylabel('Value')
    #     ax1.plot(self.trade_dates, df_plot.loc[:, 'option_pnl'], linewidth=0.5, label='total_value')
    #     ax1.plot(self.trade_dates, df_plot.loc[:, 'delta_pnl'], linewidth=0.5, label='delta_value')
    #     ax1.plot(self.trade_dates, df_plot.loc[:, 'gamma_pnl'], linewidth=0.5, label='gamma_value')
    #     ax1.plot(self.trade_dates, df_plot.loc[:, 'theta_pnl'], linewidth=0.5, label='theta_value')
    #     ax1.plot(self.trade_dates, df_plot.loc[:, 'vega_pnl'], linewidth=0.5, label='vega_value')
    #     ax1.plot(self.trade_dates, df_plot.loc[:, 'higher_order_pnl'], linewidth=0.5, label='higher_order_value')
    #     ax1.plot(self.trade_dates, df_plot.loc[:, 'trading_cost'], linewidth=0.5, label='trade_cum_cost')
    #     for tradeday in pd.to_datetime(df_plot[df_plot['position_rate'].diff() != 0].index):
    #         ax1.axvline(tradeday, linewidth=0.5, color='lightgrey', zorder=0)
    #     ax1.legend(bbox_to_anchor=(1.03, 0.85), loc='upper left', borderaxespad=1, fontsize=9)
    #     ax2 = ax1.twinx()
    #     ax2.plot(self.trade_dates, df_plot.loc[:, 'position_rate'], linewidth=0.5, drawstyle='steps-post', label='asset_delta（右轴）', c='black')
    #     ax2.plot(self.trade_dates, df_plot.loc[:, 'upper_bound'], linewidth=0.5, label='delta_upper_bound', c='pink')
    #     ax2.plot(self.trade_dates, df_plot.loc[:, 'lower_bound'], linewidth=0.5, label='delta_lower_bound', c='pink')
    #     ax2_lim = [np.percentile(df_plot.loc[:, 'lower_bound'], 5), np.percentile(df_plot.loc[:, 'upper_bound'], 95)]
    #     ax2.set_ylim(2*ax2_lim[0]-ax2_lim[1], ax2_lim[1]*1.1)
    #     # ax2.set_ylim(-1, 2)
    #     ax2.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=1, fontsize=9)
    #     strategy_name = re.findall(r'\'(.*?)\'', str(type(self.strategy)))[0].split('.')[-1]
    #     ax1.set_title('收益分解图-对冲方法：{}-总利润：{}-gamma：{}-theta：{}\nvega：{}-高阶：{}-未对冲损益：{}-交易成本：{}'.format(
    #         strategy_name, self.decomposition_summary['total_profit'], self.decomposition_summary['total_gamma_pnl'],
    #                self.decomposition_summary['total_theta_pnl'], self.decomposition_summary['total_vega_pnl'],
    #                self.decomposition_summary['total_higher_order_pnl'], self.decomposition_summary['total_unhedged_pnl'],
    #                self.decomposition_summary['total_trading_cost']))
    #     # plt.show()
    #     fig.savefig('../03_img/对冲收益分解.jpg')
    #
    # def generate_report(self):
    #     rt = ReportTemplate()
    #     story = list()
    #     # story.append(Paragraph('<para><b>对冲回测报告</b></para>', style=rt.txt_style['标题1']))
    #     story.append(Paragraph('期权对冲回测报告', style=rt.txt_style['标题1']))
    #     story.append(Spacer(240, 20))
    #     story.append(Paragraph('1.参数详情', style=rt.txt_style['标题2']))
    #     story.append(Spacer(240, 20))
    #     table1_data = [
    #         ['参数名称', '参数数值'],
    #         ['期权类型', self.paras.get('option_type')],
    #         ['名义金额', self.paras.get('notional')],
    #         ['发行日期', self.paras.get('start_date')],
    #         ['行权日期', self.paras.get('end_date')],
    #         ['标的股票', self.paras.get('stock_code')],
    #         ['标的价格', self.paras.get('start_price')],
    #         ['对冲策略', re.findall(r'\'(.*?)\'', str(type(self.strategy)))[0].split('.')[-1]]
    #         ]
    #     story.append(rt.gen_table(table1_data))
    #     story.append(Spacer(240, 10))
    #     story.append(Paragraph('2.回测结果', style=rt.txt_style['标题2']))
    #     story.append(Spacer(240, 10))
    #     story.append(Paragraph('2.1 股票对冲回测', style=rt.txt_style['标题3']))
    #     story.append(Spacer(240, 10))
    #     table2_data = [
    #         ['总损益', '股票损益', '期权损益', '交易成本', '最小现金账户', '最大回撤'],
    #         self.hedge_pnl_summary.tolist()
    #     ]
    #     story.append(rt.gen_table(table2_data))
    #     story.append(Spacer(240, 10))
    #     story.append(rt.gen_img('../03_img/股票对冲回测.jpg'))
    #     # story.append(Spacer(240, 20))
    #     story.append(Paragraph('2.2 对冲收益分解', style=rt.txt_style['标题3']))
    #     story.append(Spacer(240, 10))
    #     table3_data = [
    #         ['gamma', 'vega', 'theta', '高阶项', '未对冲损益', '交易成本', '总损益'],
    #         self.decomposition_summary.tolist()
    #     ]
    #     story.append(rt.gen_table(table3_data))
    #     story.append(Spacer(240, 10))
    #     story.append(rt.gen_img('../03_img/对冲收益分解.jpg'))
    #     doc = SimpleDocTemplate('./report/对冲回测报告.pdf')
    #     doc.build(story)
    #



