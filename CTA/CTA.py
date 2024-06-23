import os as _os
import sys as _sys
from collections import namedtuple as _namedtuple
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import pandas as pd
import seaborn as _sns
from tabulate import tabulate as _tabulate

from bar_generator import BarGenerator

_plt.style.use('seaborn-v0_8')
_b = _sns.color_palette()[0]
_g = _sns.color_palette()[1]
_r = _sns.color_palette()[2]


class CTA:
    def __init__(self, strategy, asset, leverage, cost,
                 data_path_min, data_path_day, benchmark_path,
                 data_type='min', start='2000', end='2024', resample=None,
                 plot='download',
                 output_path=r"D:\实习\研报复现\因子\CTA策略库", verbose=False):
        self._intraday_balance = None
        self.res = None
        self.strategy = strategy
        self.data_path_min = data_path_min
        self.data_path_day = data_path_day
        self.benchmark_path = benchmark_path

        self._cost = cost
        self._leverage = leverage

        self._data_type = data_type
        self._start = start
        self._end = end
        self._resample = resample

        self._output_performance = _os.path.join(output_path, 'report')
        self._output_plot = _os.path.join(output_path, 'picture')
        if not _os.path.exists(self._output_performance):
            _os.makedirs(self._output_performance)

        if not _os.path.exists(self._output_plot):
            _os.makedirs(self._output_plot)

        self._output_performance = _os.path.join(self._output_performance,
                                                 self.strategy.name + '_' + asset + '_' + self._data_type + '.xlsx')
        self._output_plot = _os.path.join(self._output_plot,
                                          self.strategy.name + '_' + asset + '_' + self._data_type + '.png')

        self._plot = plot
        self._verbose = verbose

        self.account_info = []
        self._signal = None
        # 检查策略是否成功初始化
        # self._check_strategy()

    def _check_strategy(self):
        # 检查是否有signal模块
        if hasattr(self.strategy, 'calculate_trading_signal'):
            if self._verbose:
                print("初始化成功")
        else:
            print('策略中没有信号生产器')
            _sys.exit(1)

    @staticmethod
    def _round_time(time):
        if time.hour < 15 or (time.hour == 15 and time.minute <= 15):
            return _pd.Timestamp(time.year, time.month, time.day)  # 当天的开始
        else:
            return _pd.Timestamp(time.year, time.month, time.day) + _pd.Timedelta(days=1)  # 第二天的开始

    def settle(self, price):
        """
        每日开盘结算上一次交易的收益
        1. 如果账户未创建直接跳过
        2. 如果账户创建
            2.1 如果最新持仓量大于0，结算收益，更新最后一个账户
            2.2 其余情况直接跳过
        :param price:
        :return:
        """
        if len(self.account_info) > 0:
            balance, pos, direction, signal, enter_price, settle_price, returns, cost, date = self.get_account(-1)

            # 如果仓位不等于0，结算收益
            if pos != 0:
                # 收益，新账户，和结算价
                returns = pos * (price - enter_price) / enter_price * self._leverage
                balance = balance * (1 + returns)

                self.update_account(append=False, balance=balance, pos=pos, direction=direction, signal=signal,
                                    enter_price=enter_price, settle_price=price, returns=returns, cost=cost,
                                    date=date)
            else:
                return
        else:
            return

    def process_trade(self, time_idx, price):
        """
        处理交易指令
        1. 如果没有交易账户，跳过
        2. 如果有交易账户
            2.1 无信号+无持仓 跳过
            2.2 无信号+持仓，创建一个交易账户结算价为空的账户
            2.3 有信号+无持仓，创建一个交易账户结算价为空的账户
        :param price:
        :param time_idx:
        :return:
        """
        if len(self.account_info) > 0:
            balance, pos, direction, signal, enter_price, settle_price, returns, cost, date = self.get_account(-1)
            if signal == 0 and pos == 0:
                return
            elif signal == 0 and pos != 0:
                self.update_account(append=True, balance=balance, pos=pos, direction=direction, signal=signal,
                                    enter_price=settle_price, settle_price=_np.nan, returns=0, cost=0,
                                    date=time_idx)
            elif signal != 0:
                pos += direction * signal
                cost = self._cost * abs(signal * direction)
                self.update_account(append=True, balance=balance, pos=pos, direction=0, signal=0,
                                    enter_price=price, settle_price=_np.nan, returns=0, cost=cost,
                                    date=time_idx)
        else:
            return

    def process_signal(self, new_data):
        """
        处理交易信号
        1. 如果交易指令不为空
            1.1 如果账户未创建，创建账户，balance设置为1，pos为0，signal为最新信号，入场价和结算价为None
            1.2 如果账户已创建，更新账户，balance/pos均为前者，singal为最新，入场价前者(如果有持仓)，结算价为None
                1.2.1 如果有持仓，那么最新账户的入场价应为结算价
        2. 如果交易指令为空
            2.1 如果账户未创建，创建账户，balance设置为1，pos为0，signal为0，入场价和结算价为None
            2.2 如果账户已创建，更新账户，balance/pos均为前者，singal为0，入场价前者(如果有持仓)，结算价为None
                2.2.1 如果有持仓，那么最新账户的入场价应为结算价
        :return:
        """

        if self._signal is not None:
            if len(self.account_info) == 0:
                try:
                    date = new_data.name
                except AttributeError:
                    date = new_data[0].name
                self.update_account(append=True, balance=1, pos=0, direction=self._signal[0], signal=self._signal[-1],
                                    enter_price=_np.nan, settle_price=_np.nan, returns=0, cost=0, date=date)
            else:
                (balance, pos, direction, signal, enter_price,
                 settle_price, returns, cost, date) = self.get_account(-1)

                # 防止早盘平仓，尾盘出信号
                try:
                    new_date = new_data.name
                except AttributeError:
                    new_date = new_data[0].name

                if pos == 0 and new_date != date:
                    self.update_account(append=True, balance=balance, pos=pos, direction=self._signal[0],
                                        signal=self._signal[-1], enter_price=_np.nan, settle_price=_np.nan,
                                        returns=0, cost=0, date=new_date)

                # elif new_date == date:
                #     self.update_account(append=False, balance=balance, pos=pos, direction=self._signal[0],
                #                         signal=self._signal[-1], enter_price=_np.nan, settle_price=_np.nan,
                #                         returns=0, cost=0, date=date)
                else:
                    self.update_account(append=False, balance=balance, pos=pos, direction=self._signal[0],
                                        signal=self._signal[-1], enter_price=enter_price, settle_price=_np.nan,
                                        returns=0, cost=0, date=date)
            self._signal = None
        else:
            return

    def get_account(self, order):
        return self.account_info[order]

    def update_account(self, append=False, **kwargs):
        account_info = _namedtuple('AccountInfo',
                                   ['balance', 'pos',
                                    'direction', 'signal',
                                    'enter_price', 'settle_price', 'returns', 'cost', 'date'])

        balance = kwargs['balance']
        pos = kwargs['pos']
        direction = kwargs['direction']
        signal = kwargs['signal']
        enter_price = kwargs['enter_price']
        settle_price = kwargs['settle_price']
        returns = kwargs['returns']
        cost = kwargs['cost']
        time_idx = kwargs['date']

        account_info = account_info(balance=balance, pos=pos,
                                    direction=direction, signal=signal,
                                    enter_price=enter_price, settle_price=settle_price,
                                    returns=returns, cost=cost, date=time_idx)
        if not append:
            self.account_info[-1] = account_info
        else:
            self.account_info.append(account_info)

    def backtest(self):
        bg = BarGenerator(self._start, self._end, self.data_path_min, self.data_path_day, self._resample)
        for data_row in bg.bar_generator():
            # 开盘先结算前仓位
            if (self.data_path_min is not None and self.data_path_day is not None) \
                    or (self.data_path_min is not None):
                time_idx = data_row[0].name
                opn = data_row[0]['open']
            else:
                time_idx = data_row.name
                opn = data_row['open']

            self.settle(price=opn)
            self.process_trade(time_idx=time_idx, price=opn)

            self.strategy.main(new_data=data_row, account=self.account_info)
            self._signal = self.strategy.get_trading_signal()
            self.process_signal(new_data=data_row)
            self.strategy.set_trading_signal(None)

            if len(self.account_info) > 0:
                # self._trade_date.append(time_idx)
                _sys.stdout.write("\r{} \t{}".format(time_idx.strftime('%Y%m%d'), self.account_info[-1].balance))
                _sys.stdout.flush()
        self.res = _pd.DataFrame(self.account_info).set_index('date')
        self.res['rtns_aft_fee'] = self.res['returns'] - self.res['cost']
        self.res['aft_fee_balance'] = (self.res['rtns_aft_fee'] + 1).cumprod()
        self.res['long_only'] = (1 + self.res.apply(calculate_long_only, axis=1) * self._leverage).cumprod()
        self.res['short_only'] = (1 + self.res.apply(calculate_short_only, axis=1) * self._leverage).cumprod()

        if self._data_type == 'min':
            self.res.index = self.res.index.map(self._round_time)
            self.res = self.res.resample('D').agg({
                'direction': lambda x: x.abs().sum(),  # 对direction列取绝对值加总
                'signal': lambda x: _np.sign((x > 0).sum() - (x < 0).sum()),  # 对signal列取当日内符号最多的
                'pos': 'last',
                'returns': 'sum',  # returns列加总
                'rtns_aft_fee': 'sum',  # rtns_aft_fee列加总
                'cost': 'sum',  # cost列加总
                'balance': 'last',  # balance列取最后一个值
                'aft_fee_balance': 'last',  # aft_fee_balance列取最后一个值
                'long_only': 'last',  # long_only列取最后一个值
                'short_only': 'last'  # short_only列取最后一个值
            }).dropna(axis=0, subset=['balance'])
            self._intraday_balance = self.res['balance'].resample('D').sum()
            self._intraday_balance.name = 'intraday_balance'
            self.res = pd.concat([self.res, self._intraday_balance], axis=1).dropna()
            self.res['overnight_balance'] = self.res['intraday_balance'] / self.res['balance']
        elif self._data_type == 'day':
            self.res['intraday_balance'] = 1
            self.res['overnight_balance'] = 1
            print("Known Bug on Line 261")

    def _load_benchmark(self):
        _, file_extension = _os.path.splitext(self.benchmark_path)
        if file_extension == '.pkl':
            df = _pd.read_pickle(self.benchmark_path)
        elif file_extension == '.feather':
            df = _pd.read_feather(self.benchmark_path)  # .set_index('date')
        elif file_extension == '.csv':
            df = _pd.read_csv(self.benchmark_path, index_col=0)
            df.index = _pd.to_datetime(df.index, format='%Y%m%d')
        elif file_extension == '.xlsx':
            df = _pd.read_excel(self.benchmark_path, index_col=0)
            df.index = _pd.to_datetime(df.index)
        elif file_extension == '.parquet':
            df = _pd.read_parquet(self.benchmark_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        df = df.loc[self._start:self._end]

        df['benchmark'] = df['open'].pct_change()

        return df['benchmark'].shift(-1)

    def performance_panel(self, balance_name):
        indicator = self.res[balance_name].groupby(_pd.Grouper(freq='Y')).apply(performance_stats)
        idx = indicator.index
        indicator = _pd.concat([_pd.Series(x) for x in indicator], axis=1)

        indicator.index = ['Return (%)', 'Vol (%)', 'SR', 'Max Drawdown (%)',
                           'Drawdown Period', 'Winning Rate (%)', 'Calmar']

        indicator.columns = idx.strftime('%Y-%m-%d')

        if balance_name != 'benchmark':
            totaltrade = abs(self.res['direction']).groupby(_pd.Grouper(freq='Y')).sum()
            totaltrade.index = totaltrade.index.strftime('%Y-%m-%d')
            totaltrade.name = 'TotalTrade'
            # mean
            indicator = _pd.concat([indicator.T, totaltrade], axis=1)
            indicator.loc['mean'] = round(indicator.mean(), 3)
        else:
            indicator['mean'] = round(indicator.mean(axis=1), 3)
            indicator = indicator.T

        return indicator.dropna(axis=0)

    def print_performance(self, balance_name):
        # table
        table = _tabulate(self.performance_panel(balance_name), headers='keys', tablefmt='simple',
                          numalign='right')
        print(table)

    # plot backtest result
    def create_plot(self):
        """
        创建图像
        :return: fig,ax 各个图的图片对象和轴对象，用于保存或者在控制台直接绘制
        """
        # 创建颜色映射
        cmap = _plt.get_cmap('coolwarm')
        colors = [cmap(i) for i in _np.linspace(0, 1, 10)]

        # 创建包含两个子图的图表
        fig, axes = _plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        # 账户
        axes[0].plot(self.res['balance'], color=colors[-1], label='balance_bef_fee')
        axes[0].plot(self.res['aft_fee_balance'], color=colors[0], label='balance_aft_fee')
        axes[0].legend(ncol=2, loc='upper center')

        # 收益拆解，多-空
        axes[1].plot(self.res['long_only'], color=colors[-1], label='long_only')
        axes[1].plot(self.res['short_only'], color=colors[0], label='short_only')
        axes[1].legend(ncol=2, loc='upper center')

        # 收益拆解，日内-隔夜
        axes[2].plot(self.res['intraday_balance'], color=colors[-1], label='intraday_rtns')
        axes[2].plot(self.res['overnight_balance'], color=colors[0], label='overnight_rtns')
        axes[2].legend(ncol=2, loc='upper center')

        # 在第一个子图上绘制benchmark
        axes[3].plot(self.res['balance'], label='balance', color=colors[-1], zorder=3)
        axes[3].plot(self.res['benchmark'], label='benchmark', color=colors[0], zorder=1)
        axes[3].legend(ncol=3, loc='upper center')

        # 调整子图间距
        _plt.tight_layout()

        if self._plot == 'plot':
            # 显示图表
            _plt.show()
        elif self._plot == 'download':
            fig.savefig(self._output_plot)
            _plt.close(fig)
        else:
            _plt.show()
            fig.savefig(self._output_plot)
            _plt.close(fig)

    def output_result(self):
        start_date = self.res.index[0]
        end_date = self.res.index[-1]

        # if self._data_type == 'D':
        benchmark = self._load_benchmark()
        # self.res.to_excel('res.xlsx')
        # print(self.res.index[self.res.index.duplicated()])
        self.res = _pd.concat([self.res, benchmark], axis=1).loc[start_date:end_date].iloc[:-1]

        ########
        self.res[['direction', 'signal', 'pos', 'returns', 'rtns_aft_fee', 'cost']] = (
            self.res[['direction', 'signal', 'pos', 'returns', 'rtns_aft_fee', 'cost']].fillna(0))
        self.res[['balance', 'aft_fee_balance', 'long_only', 'short_only', 'intraday_balance', 'overnight_balance']] =(
            self.res[['balance', 'aft_fee_balance', 'long_only', 'short_only',
                      'intraday_balance', 'overnight_balance']].ffill())
        ########

        self.res['benchmark'] = (self.res['benchmark'] + 1).cumprod()
        self.res.dropna(subset=['balance'], axis=0, inplace=True)

        before_fee_pfm = self.performance_panel('balance')
        aft_fee_pfm = self.performance_panel('aft_fee_balance')
        benchmark_pfm = self.performance_panel('benchmark')
        with _pd.ExcelWriter(self._output_performance) as writer:
            # 将每个DataFrame写入不同的sheet
            before_fee_pfm.to_excel(writer, sheet_name='before_fee_pfm')
            aft_fee_pfm.to_excel(writer, sheet_name='aft_fee_pfm')
            benchmark_pfm.to_excel(writer, sheet_name='benchmark_pfm')
            self.res.to_excel(writer, sheet_name='detail')
        if self._verbose:
            print(f"\n结果已经存入{self._output_performance}")


# ==============useful tools============
def performance_stats(year_group):
    """
    Internal Function to calculate performance metric
    """
    # annual ret/std/IR
    AnuualReturn = (year_group.iloc[-1] - year_group.iloc[0]) / year_group.iloc[0]
    AnnualStd = year_group.pct_change(fill_method=None).std() * _np.sqrt(len(year_group))

    if AnnualStd == 0 or _np.isnan(AnnualStd):
        AnnualStd = 0.01

    SR = round(AnuualReturn / AnnualStd, 3)

    # win rate

    _ = year_group.pct_change(fill_method=None).replace(0, _np.nan).dropna()
    pos_return_len = (_ > 0).astype(int).sum()
    win_rate = pos_return_len / len(_)

    # drawdown
    Drawdown = (year_group - year_group.expanding().max()) / year_group.expanding().max()
    MaxDrawdown = Drawdown.min()
    if MaxDrawdown == 0 or _np.isnan(MaxDrawdown):
        MaxDrawdown = 0.001
    Drawdown.reset_index(drop=True, inplace=True)

    if len(Drawdown.loc[:Drawdown.idxmin()][Drawdown == 0]) > 0:
        start = Drawdown.loc[:Drawdown.idxmin()][Drawdown == 0].last_valid_index()
    else:
        start = Drawdown.loc[:Drawdown.idxmin()].idxmax()

    if Drawdown.loc[Drawdown.idxmin():].max() == 0:
        end = Drawdown.loc[Drawdown.idxmin():].idxmax()
    else:
        end = Drawdown.last_valid_index()

    calmar = AnuualReturn / abs(MaxDrawdown)

    return [round(AnuualReturn * 100, 2), round(AnnualStd * 100, 2), SR, round(MaxDrawdown * 100, 2),
            end - start, round(win_rate * 100, 2), round(calmar, 2)]


# 定义一个函数，用于根据pos的值来计算long_only
def calculate_long_only(row):
    if row['pos'] > 0:
        return row['pos'] * row['returns']
    else:
        return 0


# 定义一个函数，用于根据pos的值来计算long_only
def calculate_short_only(row):
    if row['pos'] < 0:
        return -row['pos'] * row['returns']
    else:
        return 0
