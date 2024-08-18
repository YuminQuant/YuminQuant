import os as _os
import numpy as _np
import numpy as np
import pandas as _pd
import matplotlib.pyplot as _plt
import bisect as _bisect


class EventDriven:
    def __init__(self, trade_list, name, event_period=60, tail=False, cost=0.001,
                 holding_period=20, st=True, up_down_limit=True, stock_pool='mkt', benchmark='mkt',
                 return_type='vwap', data_path=r'D:\实习\研报复现\dataAll\stockDayData',
                 output_path=r'D:\实习\研报复现\QuantResearch\stock_strategy', plot='download', verbose=False):

        self.weight = None
        self.res = None
        self.strategy_excess_rtn = None
        self.strategy_rtn = None
        self.period_rtn = None
        self.period_excess_rtn = None

        self._event_period = event_period
        self._tail = tail
        self._holding_period = holding_period
        self._stock_pool = stock_pool
        self._benchmark = benchmark
        self._plot = plot
        self._st = st
        self._up_down_limit = up_down_limit
        self._cost = cost
        self._verbose = verbose

        self._benchmark_path = _os.path.join(data_path, f'stockPool\\benchmark.pkl')
        self._ST_path = _os.path.join(data_path, 'pv\\st_dummy.pkl')
        self._upDownLimit_path = _os.path.join(data_path, 'pv\\upDownLimit.pkl')
        self._Ddates_path = _os.path.join(data_path, f'calendar\\Ddates.pkl')
        if type(self._holding_period) is int:
            self._calendar_path = _os.path.join(data_path, f'calendar\\Ddates.pkl')
        else:
            self._calendar_path = _os.path.join(data_path, f'calendar\\{self._holding_period}dates.pkl')

        self._status = _pd.read_pickle(_os.path.join(data_path, 'pv\\stock_status.pkl'))

        # output path
        self._output_performance = _os.path.join(output_path, 'report')
        self._output_plot = _os.path.join(output_path, 'picture')
        if not _os.path.exists(self._output_performance):
            _os.makedirs(self._output_performance)

        if not _os.path.exists(self._output_plot):
            _os.makedirs(self._output_plot)

        self._output_performance = _os.path.join(self._output_performance,
                                                 name + '.xlsx')
        self._output_plot = _os.path.join(self._output_plot,
                                          name + '.png')

        # data path
        if return_type == 'close':
            self._return_path = _os.path.join(data_path, 'pv\\adj_close_return.pkl')
        elif return_type == 'vwap':
            self._return_path = _os.path.join(data_path, 'pv\\adj_vwap_return.pkl')

        if self._stock_pool != '000985' and self._stock_pool != 'mkt':
            self._components_path = _os.path.join(data_path, f'stockPool\\stockPool_{self._stock_pool}.pkl')
            if _os.path.exists(self._components_path):
                pass
            else:
                print('文件不存在', self._components_path)

        if self._benchmark != 'mkt':
            self._benchmark_components_path = _os.path.join(data_path, f'stockPool\\weight_{self._benchmark}.pkl')
            if _os.path.exists(self._benchmark_components_path):
                pass
            else:
                print('文件不存在', self._benchmark_components_path)

        # 检查格式
        if verbose:
            print("检查表结构")
        self._trade_list = trade_list
        self._check_required_columns()
        self._trade_list = self._trade_list.sort_values('date')
        self._Ddates = _pd.read_pickle(self._Ddates_path)

        if verbose:
            print("获取交易日")
        self._trade_list['first_trade_date'] = self._trade_list['date'].map(self._find_nearest_trade_date)
        self._trade_list['last_trade_date'] = self._trade_list['first_trade_date'].map(self._find_last_trade_date)

        if self._st:
            self._clean_ST()

        if self._stock_pool != 'mkt' and self._stock_pool != '000985':
            self._get_stock_pool()

    def _check_required_columns(self):
        missing_columns = [col for col in ['date', 'stock_code', 'isEvent'] if col not in self._trade_list.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    def _get_stock_pool(self):
        if self._verbose:
            print(f"开始获取股票池{self._stock_pool}")
        components = _pd.read_pickle(self._components_path).stack()
        self._trade_list = self._trade_list[
            self._trade_list.set_index(['date', 'stock_code']).index.isin(components.index)]

    def _clean_ST(self):
        """
        清除ST
        :return:
        """

        if self._verbose:
            print("清除ST")
        ST = _pd.read_pickle(self._ST_path)
        ST = (ST != 1.0).astype(int).replace(0, _np.nan).stack()
        self._trade_list = self._trade_list[~self._trade_list.set_index(['date', 'stock_code']).index.isin(ST.index)]

    def _clean_upDownLimit(self):
        if self._verbose:
            print("清除涨跌停")
        upDownLimit = _pd.read_pickle(self._upDownLimit_path).stack()
        pass_trade_list = self._trade_list[self._trade_list.set_index(['date',
                                                                       'stock_code']).index.isin(upDownLimit.index)]
        notpass_trade_list = self._trade_list[~self._trade_list.set_index(['date',
                                                                           'stock_code']).index.isin(upDownLimit.index)]
        notpass_trade_list['first_trade_date'] = notpass_trade_list['first_trade_date'].map(
            self._find_nearest_trade_date)

        self._trade_list = _pd.concat([pass_trade_list, notpass_trade_list], axis=0)

    def _get_benchmark(self):
        if self._benchmark == 'mkt' or self._benchmark == '000985':
            benchmark = _pd.read_pickle(self._return_path).mean(axis=1)
        else:
            stock_pool = _pd.read_pickle(self._benchmark_components_path)
            benchmark = (_pd.read_pickle(self._return_path).mul(stock_pool, axis=0)
                         .dropna(axis=0, how='all').sum(axis=1).loc[:'2024-03'])
        return benchmark

    def _find_nearest_trade_date(self, date):
        idx = _bisect.bisect_left(self._Ddates, date)
        if idx < len(self._Ddates) and self._Ddates[idx] == date:
            # 如果日期是交易日，找到下一个交易日
            return self._Ddates[min(idx + 1, len(self._Ddates) - 1)]
        else:
            # 如果日期不是交易日，找到大于它的最小的交易日
            return self._Ddates[idx] if idx < len(self._Ddates) else self._Ddates[-1]

    def _find_last_trade_date(self, date):
        calendar = _pd.read_pickle(self._calendar_path)
        idx = _bisect.bisect_left(calendar, date)
        if idx < len(calendar) and calendar[idx] == date:
            # 如果日期是交易日，找到下一个交易日
            if type(self._holding_period) is int:
                return calendar[min(idx + self._holding_period, len(calendar) - 1)]
            else:
                return calendar[min(idx + 1, len(calendar) - 1)]
        elif idx < len(calendar):
            return calendar[min(idx, len(calendar) - 1)]
        else:
            return _np.nan

    def get_forward_backward_rtn(self):
        if self._verbose:
            print("开始统计事件收益")

        Ddates = _pd.read_pickle(self._Ddates_path).to_list()
        stock_rtn_adj = _pd.read_pickle(self._return_path)
        # stock_rtn = _pd.read_pickle(r"D:\实习\研报复现\dataAll\stockDayData\pv\stock_close.pkl").pct_change(
        #     fill_method=None)
        benchmark = self._get_benchmark()
        grouped = self._trade_list.groupby('date')
        sample = 0

        for event_day, event_list in grouped:
            trade_code = event_list['stock_code'].to_list()

            try:
                if self._tail:
                    # 判断回测考虑事件后，还是事件前后
                    # 如果是日历中找不到对应交易日，或者期间收益长度不满观察周期，
                    # 则抛弃样本
                    start_date = Ddates[Ddates.index(event_day) + 1]
                    end_date = Ddates[Ddates.index(start_date) + self._event_period - 1]
                else:
                    start_date = Ddates[int(Ddates.index(event_day) - self._event_period / 2)]
                    end_date = Ddates[int(Ddates.index(event_day) + self._event_period / 2)]
            except:
                continue

            new_period_rtn = (stock_rtn_adj[trade_code].loc[start_date:end_date]
                              .reset_index(drop=True).fillna(0).sum(axis=1))

            # event length checker
            if self._tail:
                if len(new_period_rtn) < self._event_period:
                    continue
            else:
                if len(new_period_rtn) <= self._event_period:
                    continue

            sample += len(trade_code)
            new_period_excess_rtn = stock_rtn_adj[trade_code].loc[start_date:end_date].reset_index(drop=True).sub(
                benchmark.loc[start_date:end_date].reset_index(drop=True), axis=0).sum(axis=1).fillna(0)

            if self.period_rtn is None:
                self.period_rtn = new_period_rtn.copy()
                self.period_excess_rtn = new_period_excess_rtn.copy()
            else:
                self.period_rtn += new_period_rtn
                self.period_excess_rtn += new_period_excess_rtn

        self.period_rtn /= sample
        self.period_rtn.loc[self.period_rtn.index[0] - 1] = 0
        self.period_rtn.sort_index(inplace=True)
        self.period_rtn.reset_index(drop=True, inplace=True)

        self.period_excess_rtn /= sample
        self.period_excess_rtn.loc[self.period_excess_rtn.index[0] - 1] = 0
        self.period_excess_rtn.sort_index(inplace=True)
        self.period_excess_rtn.reset_index(drop=True, inplace=True)
        if not self._tail:
            self.period_rtn.index = np.arange(-int(self._event_period / 2) - 1, int(self._event_period / 2 + 1))
            self.period_excess_rtn.index = np.arange(-int(self._event_period / 2) - 1, int(self._event_period / 2 + 1))

    def backtest(self):
        if self._verbose:
            print("开始回测策略")
        if self._up_down_limit:
            self._clean_upDownLimit()

        stock_group = self._trade_list.groupby('stock_code')
        stock_rtn = _pd.read_pickle(self._return_path)
        benchmark = self._get_benchmark()
        all_stock_rtn = []
        for stock, stock_event in stock_group:
            single_stock_rtn = []
            for idx, data in stock_event.iterrows():
                single_stock = stock_rtn[stock].loc[data['first_trade_date']:data['last_trade_date']]
                single_stock.iloc[0] -= self._cost
                # single_stock.iloc[-1] -= self._cost

                single_stock_rtn.append(single_stock)
            single_stock_rtn = (_pd.concat(single_stock_rtn).reset_index()
                                .drop_duplicates(subset='tradeDate', keep='first').set_index('tradeDate').sort_index())
            all_stock_rtn.append(single_stock_rtn)

        strategy_rtn = _pd.concat(all_stock_rtn, axis=1).sort_index()
        start_date = strategy_rtn.index[0]
        strategy_rtn = strategy_rtn.reindex(self._Ddates).loc[start_date:].dropna(axis=0, how='all')
        weight = strategy_rtn.where(strategy_rtn.isna(), 1.0).div(strategy_rtn.count(axis=1), axis=0).fillna(0)

        self.weight = weight.clip(upper=0.1)
        strategy_rtn = strategy_rtn.mul(self.weight, axis=0).sum(axis=1)
        self.res = _pd.concat([strategy_rtn, benchmark], axis=1).loc[start_date:]
        self.res.columns = ['strategy_rtn', 'benchmark']
        self.res['excess'] = (self.res['strategy_rtn'].replace(0.0, _np.nan).fillna(self.res['benchmark']) -
                              self.res['benchmark'])

    def describe_event(self, balance_name):
        yearly_sample = self._trade_list.set_index('date').groupby(_pd.Grouper(freq='Y')).size()

        indicator = (self.res[balance_name].fillna(0) + 1).cumprod()
        indicator.loc[indicator.index[0] - _pd.Timedelta('1D')] = 1
        indicator.sort_index(inplace=True)

        indicator = indicator.groupby(_pd.Grouper(freq='Y')).apply(performance_stats)
        idx = indicator.index
        indicator = _pd.concat([_pd.Series(x) for x in indicator], axis=1)

        indicator.index = ['Return (%)', 'Vol (%)', 'SR', 'Max Drawdown (%)',
                           'Drawdown Period', 'Winning Rate (%)', 'Calmar']

        indicator.columns = idx.strftime('%Y-%m-%d')
        yearly_sample.index = yearly_sample.index.strftime('%Y-%m-%d')
        yearly_sample.name = 'total_sample'
        turnover = round(self.weight.diff().abs().sum(axis=1).groupby(_pd.Grouper(freq='Y')).sum() * 100 / 250, 3)
        turnover.index = turnover.index.strftime('%Y-%m-%d')
        turnover.name = 'AvgTurnover(%)'
        # mean
        indicator = _pd.concat([indicator.T, turnover, yearly_sample], axis=1).dropna(axis=0, how='all')
        indicator.loc['mean'] = round(indicator.mean(), 3)

        return indicator.dropna(axis=0)

    def create_plot(self):
        """
        创建图像
        :return: fig,ax 各个图的图片对象和轴对象，用于保存或者在控制台直接绘制
        """
        # 创建颜色映射
        cmap = _plt.get_cmap('coolwarm')
        colors = [cmap(i) for i in _np.linspace(0, 1, 10)]

        cumulative_excess = (1 + self.res['excess']).cumprod() - 1
        cumulative_group = (1 + self.res['strategy_rtn'].replace(0.0, _np.nan).fillna(self.res['benchmark'])).cumprod()
        benchmark = (1 + self.res['benchmark']).cumprod()

        period_rtn = (self.period_rtn + 1).cumprod() - 1
        period_excess_rtn = (self.period_excess_rtn + 1).cumprod() - 1
        # 创建包含两个子图的图表
        fig, axes = _plt.subplots(3, 1, figsize=(8, 6))
        # 账户
        axes[0].plot(period_rtn, color=colors[-1], label='period_rtn')
        axes[0].plot(period_excess_rtn, color=colors[0], label='period_excess_rtn')
        axes[0].legend(ncol=2, loc='upper center')

        line1, = axes[1].plot(cumulative_group, color=colors[-1], label='strategy_rtn', zorder=3)
        line2, = axes[1].plot(benchmark, color=colors[0], label='benchmark', zorder=3)
        ax2 = axes[1].twinx()
        bar = ax2.fill_between(cumulative_excess.index, cumulative_excess, color='gray', alpha=0.3, zorder=1)
        ax2.grid(False)

        axes[1].legend([line1, line2, bar], ['strategy_rtn', 'benchmark', 'excess_rtn'], ncol=3,
                       loc='upper center', bbox_to_anchor=(0.5, 1.05))

        eventCount = self._trade_list.pivot(index='date', columns='stock_code', values='isEvent').count(axis=1)
        axes[2].fill_between(eventCount.index, eventCount, label='eventCount')
        axes[2].legend(ncol=1, loc='upper center')

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
        strategy_rtn_pfm = self.describe_event('strategy_rtn')
        strategy_excess_rtn_pfm = self.describe_event('excess')
        with _pd.ExcelWriter(self._output_performance) as writer:
            # 将每个DataFrame写入不同的sheet
            strategy_rtn_pfm.to_excel(writer, sheet_name='strategy_rtn_pfm')
            strategy_excess_rtn_pfm.to_excel(writer, sheet_name='excess_pfm')
            self.res.to_excel(writer, sheet_name='detail')
        if self._verbose:
            print(f"\n结果已经存入{self._output_performance}")

    def main(self):
        self.get_forward_backward_rtn()
        self.backtest()
        self.create_plot()
        self.output_result()


# ================== useful tool ==================
def performance_stats(year_group):
    """
    Internal Function to calculate performance metric
    """
    # annual ret/std/IR
    AnuualReturn = (year_group.iloc[-1] - year_group.iloc[0]) / year_group.iloc[0]
    AnnualStd = year_group.pct_change().std() * _np.sqrt(len(year_group))
    SR = round(AnuualReturn / AnnualStd, 3)

    # win rate
    _ = year_group.pct_change()
    pos_return_len = (_ > 0).astype(int).sum()
    win_rate = pos_return_len / len(_)

    # drawdown
    Drawdown = (year_group - year_group.expanding().max()) / year_group.expanding().max()
    MaxDrawdown = Drawdown.min()
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
    try:
        return [round(AnuualReturn * 100, 2), round(AnnualStd * 100, 2), SR, round(MaxDrawdown * 100, 2),
                end - start, round(win_rate * 100, 2), round(calmar, 2)]
    except TypeError:
        return [0, 0, 0, 0, 0, 0, 0]


# 定义一个函数来找到小于Series值的最大值
def find_min_greater_than_date(value, calendar):
    index = _bisect.bisect_right(calendar, value)
    if index < len(calendar) and calendar[index - 1] == value:
        return calendar[index - 1]
    elif index < len(calendar) and calendar[index - 1] != value:
        return calendar[index]

    return _np.nan  # 如果没有找到小于value的值，返回None


def _weight(row):
    row_count = len(row.dropna())
    if row_count >= 10:
        return row / row_count
    else:
        return 0.1 * row


def toTimingfctBase(fct, fct_name, base_path=f'D:\\实习\\研报复现\\QuantResearch\\timing_alpha'):
    path = _os.path.join(base_path, fct_name + '.feather')
    fct.to_feather(path)

