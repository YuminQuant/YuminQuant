# quantoolv3.0.0
# Yumin 20240501


# %%
import os as _os
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import seaborn as _sns
from tqdm import tqdm as _tqdm
from tabulate import tabulate as _tabulate

_plt.style.use('seaborn-v0_8')
_b = _sns.color_palette()[0]
_r = _sns.color_palette()[2]


# =====================================stock_factor=====================================
class Alpha:
    """
    选股因子回测框架

    1. 因子描述性统计
    2. RankIC
    3. 分层回测
    4. 导出结果
    """

    def __init__(self, factor, name, freq='W', layer=10, stock_pool='000985',
                 st=True, up_down_limit=True, weight_limit=0.05, benchmark='mkt',
                 rolling_ic=True, plot='download',
                 return_type='vwap', data_path=r'D:\实习\研报复现\stockData',
                 output_path=r'D:\实习\研报复现\因子\选股因子回测结果', verbose=False):

        # setting
        self._factor = factor.dropna(axis=0, how='all')

        self._name = name
        self._freq = freq
        self._layer = layer
        self._stock_pool = stock_pool
        self._st = st
        self._up_down_limit = up_down_limit
        self._benchmark = benchmark
        self._rolling_ic = rolling_ic
        self._weight_limit = weight_limit

        self._plot = plot
        self._verbose = verbose

        # output path
        self._output_performance = _os.path.join(output_path, name + '_' +self._freq + '.xlsx')
        self._output_plot = _os.path.join(output_path, name + '_' + self._freq + '.png')

        # data path
        if return_type == 'close':
            self._return_path = _os.path.join(data_path, 'pv\\adj_close_return.pkl')
        elif return_type == 'vwap':
            self._return_path = _os.path.join(data_path, 'pv\\adj_vwap_return.pkl')

        self._ST_path = _os.path.join(data_path, 'pv\\st_dummy.pkl')
        self._upDownLimit_path = _os.path.join(data_path, 'pv\\upDownLimit.pkl')

        self._dates_path = _os.path.join(data_path, f'calendar\\{self._freq}dates.pkl')
        self._Ddates_path = _os.path.join(data_path, f'calendar\\Ddates.pkl')

        self._components_path = _os.path.join(data_path, f'stockPool\\stockPool_{self._stock_pool}.pkl')
        self._benchmark_path = _os.path.join(data_path, f'stockPool\\benchmark.pkl')

        self._status = _pd.read_pickle(_os.path.join(data_path, 'pv\\stock_status.pkl'))
        self._factor = self._factor.mul(self._status, axis=0).dropna(axis=0, how='all').dropna(axis=1, how='all')
        # 检查路径或文件是否存在
        for _ in [self._return_path, self._upDownLimit_path,
                  self._ST_path, self._dates_path, self._Ddates_path, self._benchmark_path]:
            if _os.path.exists(_):
                continue
            else:
                print('文件不存在', _)

        if self._stock_pool != '000985' and self._stock_pool != 'mkt':
            if _os.path.exists(self._components_path):
                pass
            else:
                print('文件不存在', self._components_path)

    def backtest(self):
        self._describe_factor()
        self._ic()
        self._get_trading_panel()
        self._group(self._layer)
        self._excess_group()

    def _describe_factor(self):
        self._fct_mean = _np.mean(_np.nanmean(self._factor, axis=1))
        self._fct_std = _np.mean(_np.nanstd(self._factor, axis=1))
        self._fct_skew = _np.mean(self._factor.skew(axis=1))
        self._fct_kurt = _np.mean(self._factor.kurt(axis=1))
        self._fct_median = _np.mean(_np.nanmedian(self._factor, axis=1))
        self._fct_max = _np.median(_np.nanmax(self._factor, axis=1))
        self._fct_min = _np.median(_np.nanmin(self._factor, axis=1))
        self._avg_coverage = (self._factor.count(axis=1) / self._status.count(axis=1)).mean()
        self._period = len(self._factor)

    def _ic(self):
        if self._verbose:
            print("开始计算RankIC")
        _stock_return_daily = _pd.read_pickle(self._return_path)
        if self._rolling_ic:

            d_ic = self._factor.shift(2).corrwith(_stock_return_daily, axis=1, method='spearman').dropna()
            w_ic = self._factor.shift(5).corrwith(_stock_return_daily.rolling(4).sum()
                                                  .dropna(axis=0, how='all'), axis=1, method='spearman').dropna()
            m_ic = self._factor.shift(20).corrwith(_stock_return_daily.rolling(19).sum()
                                                   .dropna(axis=0, how='all'), axis=1, method='spearman').dropna()
            self.RankIC = _pd.concat([d_ic, w_ic, m_ic], axis=1)
            self.RankIC.columns = ['D', 'W', 'M']
        else:
            if self._freq == 'W':
                interval = 5
            elif self._freq == 'M':
                interval = 20
            dates = _pd.read_pickle(self._dates_path)
            _stock_return_daily = _pd.read_pickle(self._return_path).replace(0, _np.nan).rolling(
                interval - 1).sum().dropna(axis=0, how='all')
            mask = _stock_return_daily.index.isin(dates)
            _stock_return_daily[~mask] = _np.nan
            self.RankIC = self._factor.shift(1).corrwith(_stock_return_daily, axis=1, method='spearman').dropna()

        if isinstance(self.RankIC, _pd.Series):
            self.ic = self.RankIC.mean()
            self.icir = self.ic / self.RankIC.std()
        else:
            self.ic = self.RankIC[self._freq].mean()
            self.icir = self.ic / self.RankIC[self._freq].std()

        # 改变因子方向
        if self.ic < 0:
            self._factor *= -1

    def _get_trading_panel(self):

        dates = list(_pd.read_pickle(self._dates_path))
        Ddates = list(_pd.read_pickle(self._Ddates_path))

        if self._freq != 'D':
            self._factor = self._factor.loc[self._factor.index.isin(dates)]
        else:
            self._factor = self._factor.shift(2).dropna(axis=0, how='all')

        if self._stock_pool:
            self._get_stock_pool()
        if self._st:
            self._clean_st()
        if self._up_down_limit:
            self._clean_upDownLimit()

        del_idx = self._factor.index
        # get trading day from frequency calendar
        first_signal_day = self._factor.index[0]
        last_signal_day = self._factor.index[-1]
        first_trading_day_id = Ddates.index(first_signal_day)
        last_trading_day_id = dates.index(last_signal_day) + 1

        try:
            last_trading_day = dates[last_trading_day_id]
        except:
            last_trading_day = last_signal_day

        # get trading day id from daily-calendar
        last_trading_day_id = Ddates.index(last_trading_day)
        trading_period = Ddates[first_trading_day_id:last_trading_day_id + 1]
        trading_period.sort()

        # rebuild factor_panel
        self._factor = zscore(self._factor.reindex(trading_period, method='ffill').dropna(axis=0, how='all'))
        if self._freq != 'D':
            self._factor.loc[del_idx] = _np.nan
            self._factor = self._factor.shift(1).dropna(axis=0, how='all')
        self.first_trading_day = trading_period[0]
        self.last_trading_day = trading_period[-1]

        if self._verbose:
            print('获取交易面板完成')

    def _get_stock_pool(self):
        if self._stock_pool == 'mkt' or self._stock_pool == '000985':
            return
        else:
            index = _pd.read_pickle(self._components_path)
            index = index[index.index.isin(self._factor.index)]
            self._factor = index.mul(self._factor, axis=0).dropna(axis=1, how='all').dropna(axis=0, how='all')

        if self._verbose:
            print('获取股票池完成', self._stock_pool)

    def _clean_st(self):
        _ST = _pd.read_pickle(self._ST_path)
        _ST = _ST[_ST.index.isin(self._factor.index)]
        self._factor = self._factor.mul(_ST, axis=0).dropna(axis=1, how='all').dropna(axis=0, how='all')

        if self._verbose:
            print('清除ST')

    def _clean_upDownLimit(self):
        _upDownLimit = _pd.read_pickle(self._upDownLimit_path)
        _upDownLimit = _upDownLimit[_upDownLimit.index.isin(self._factor.index)]
        self._factor = self._factor.mul(_upDownLimit, axis=0).dropna(axis=1, how='all').dropna(axis=0, how='all')

        if self._verbose:
            print('清除涨跌停')

    def _group(self, layer):
        if self._verbose:
            print('开始分层回测...')
        # 使用stack函数将其转换为长格式
        self._factor = self._factor.rank(axis=1, pct=True, method='dense').stack().reset_index()
        self._factor.columns = ['date', 'stock_code', 'factor_rank']

        # 对每天进行分组，并将股票等分为layer层
        self._factor['quantile'] = self._factor.groupby('date')['factor_rank'].transform(
            lambda x: _pd.qcut(x, layer, labels=False, duplicates='drop')
        )

        # 使用stack函数将其转换为长格式
        _stock_return_daily = _pd.read_pickle(self._return_path).stack().reset_index()
        _stock_return_daily.columns = ['date', 'stock_code', 'returns']
        # 根据日期和股票代码将收益数据与因子数据合并
        self._factor = _pd.merge(self._factor, _stock_return_daily, on=['date', 'stock_code'])

        # 计算每层的平均收益率
        grouped = self._factor.groupby(['date', 'quantile'])
        self.group_return = grouped['returns'].mean().unstack()
        self.group_return.columns = ['G' + str(_) for _ in range(1, layer + 1)]
        self.group_return['ls'] = self.group_return.iloc[:, -1] - self.group_return.iloc[:, 0]

        self._factor['weight'] = 1 / self._factor.groupby(['date', 'quantile']).transform('size')
        self.topGroupWeight = self._factor.query(""" quantile==9 """).pivot(index='date', columns='stock_code',
                                                                            values='weight').fillna(0)
        self.bottomGroupWeight = self._factor.query(""" quantile==1 """).pivot(index='date', columns='stock_code',
                                                                               values='weight').fillna(0)

    def _excess_group(self):
        if self._verbose:
            print('开始计算分层超额...')
        if self._benchmark == 'mkt':
            benchmark = _pd.read_pickle(self._return_path).mean(axis=1).loc[
                        self.first_trading_day:self.last_trading_day]
        else:
            benchmark = _pd.read_pickle(self._benchmark_path)[self._benchmark] \
                            .pct_change().loc[self.first_trading_day:self.last_trading_day]
        self.excess_group = self.group_return.sub(benchmark, axis=0).dropna(axis=0, how='all')
        self.excess_group.columns = [_ + '_excess' for _ in self.excess_group.columns]

    def performance_panel(self, portfolio_name='G1'):
        """
        External to get the group performance panel
        """
        # format the portfolio
        if '_excess' in portfolio_name:
            portfolio = (self.excess_group[portfolio_name] + 1).cumprod()
            portfolio.loc[portfolio.index[0] - _pd.Timedelta('1D')] = 1
            portfolio.sort_index(inplace=True)
        elif 'G' in portfolio_name or 'ls' in portfolio_name:
            portfolio = (self.group_return[portfolio_name] + 1).cumprod()
            portfolio.loc[portfolio.index[0] - _pd.Timedelta('1D')] = 1
            portfolio.sort_index(inplace=True)

        # calculate yearly performance
        indicator = portfolio.groupby(_pd.Grouper(freq='Y')).apply(performance_stats)
        idx = indicator.index
        indicator = _pd.concat([_pd.Series(x) for x in indicator], axis=1)

        if 'G' in portfolio_name or portfolio_name == 'top_excess':
            turnover = round(
                (_pd.DataFrame(abs(self.topGroupWeight.diff()).sum(axis=1).groupby(_pd.Grouper(freq='Y')).sum())) * 100,
                3)
        elif 'ls' in portfolio_name:
            temp_top, temp_bottom = self.topGroupWeight.align(self.bottomGroupWeight)
            temp_top.fillna(0, inplace=True)
            temp_bottom.fillna(0, inplace=True)

            turnover = round((_pd.DataFrame(
                abs((temp_top - temp_bottom).diff()).sum(axis=1).groupby(
                    _pd.Grouper(freq='Y')).sum())) * 100, 3)

        turnover.index = turnover.index.strftime('%Y-%m-%d')
        turnover.columns = ['Turnover(%)']
        if portfolio_name == 'top_excess':
            indicator.index = ['Return (%)', 'Vol (%)', 'IR', 'Max Drawdown (%)', 'Drawdown Period', 'Winning Rate (%)',
                               'Calmar']
        elif portfolio_name == 'bottom_excess':
            indicator.index = ['Return (%)', 'Vol (%)', 'IR', 'Max Drawdown (%)', 'Drawdown Period', 'Winning Rate (%)',
                               'Calmar']
        else:
            indicator.index = ['Return (%)', 'Vol (%)', 'SR', 'Max Drawdown (%)', 'Drawdown Period', 'Winning Rate (%)',
                               'Calmar']
        indicator.columns = idx.strftime('%Y-%m-%d')

        # mean
        indicator = _pd.concat([indicator.T, turnover], axis=1)
        indicator.loc['mean'] = round(indicator.mean(), 3)

        return indicator.dropna(axis=0)

    def print_performance(self, portfolio_name='G10_excess'):
        # table
        table = _tabulate(self.performance_panel(portfolio_name=portfolio_name), headers='keys', tablefmt='simple',
                          numalign='right')
        print(table)

    # plot backtest result
    def create_plot(self):
        """
        创建图像
        :return: fig,ax 各个图的图片对象和轴对象，用于保存或者在控制台直接绘制
        """
        # 计算累计超额收益率
        cumulative_excess = (1 + self.excess_group).cumprod().iloc[:, :-1] - 1

        # 创建颜色映射
        cmap = _plt.get_cmap('coolwarm')
        colors = [cmap(i) for i in _np.linspace(0, 1, len(cumulative_excess.columns))]

        # 创建包含两个子图的图表
        fig, axes = _plt.subplots(5, 1, figsize=(8, 10))

        # ic bar
        ic = self.RankIC.mean()

        if self._rolling_ic:
            axes[0].bar(ic.index, ic, color=_b)
        else:
            axes[0].bar(self._freq, ic, color=_b)

        # ic
        if self._rolling_ic:
            ic = self.RankIC[self._freq]
        else:
            ic = self.RankIC
        bar_container = axes[1].bar(ic.index, ic, alpha=0.6, label='RankIC (left)', width=5)
        axes2 = axes[1].twinx()
        line_container, = axes2.plot(ic.index, ic.cumsum(), label='Cummulative RankIC (right)', color=_r)
        axes2.grid(False)
        # 创建图例，包含两个y轴的图例
        axes[1].legend([bar_container, line_container], ['RankIC (left)', 'Cummulative RankIC (right)'], ncol=2,
                       loc='upper center')

        # 在第一个子图上绘制累计超额收益率曲线
        for i, (column, color) in enumerate(zip(cumulative_excess.columns, colors)):
            if i == 0 or i == len(cumulative_excess.columns) - 1:
                axes[2].plot(cumulative_excess.index, cumulative_excess[column], label=column, color=color, zorder=3)
            else:
                axes[2].plot(cumulative_excess.index, cumulative_excess[column], color=color, alpha=0.8, zorder=1)
        axes[2].legend(ncol=2, loc='upper center')

        # 头组尾组换手率
        top_turnover = abs(self.topGroupWeight.diff()).sum(axis=1).replace(0, _np.nan).dropna()
        bottom_turnover = abs(self.bottomGroupWeight.diff()).sum(axis=1).replace(0, _np.nan).dropna()
        axes[3].plot(bottom_turnover.index, bottom_turnover, color=colors[0], label='G1_turnover')
        axes[3].plot(top_turnover.index, top_turnover, color=colors[-1], label='G10_turnover')
        axes[3].legend(ncol=2, loc='upper center')

        # 分组柱状图
        return_group = self.group_return.groupby(_pd.Grouper(freq='Y')).sum().mean().iloc[:-1]
        for i, (idx, color) in enumerate(zip(return_group.index, colors)):
            if i == 0 or i == len(return_group.index) - 1:
                axes[4].bar(idx, return_group[idx], color=color)
            else:
                axes[4].bar(idx, return_group[idx], color=color)

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
        desc_info = _pd.Series([self._name, self.first_trading_day, self.last_trading_day,
                               self._fct_mean, self._fct_std, self._fct_skew, self._fct_kurt,
                               self._fct_median, self._fct_min, self._fct_max, self._avg_coverage,
                               self._period, self.ic, self.icir, self._stock_pool, self._benchmark],
                              index=['name', 'start_date', 'end_date', 'mean', 'std', 'skew',
                                     'kurt', 'median', 'min', 'max', 'coverage', 'efficient_period',
                                     f'{self._freq}_RankIC', f'{self._freq}_RankICIR', 'stock_pool', 'benchmark'])
        with _pd.ExcelWriter(self._output_performance) as writer:
            # 将每个DataFrame写入不同的sheet
            desc_info.to_excel(writer, sheet_name='fct_info')
            self.performance_panel('ls').to_excel(writer, sheet_name='L-S_pfm')
            self.performance_panel('G10_excess').to_excel(writer, sheet_name='G10_Epfm')
            self.group_return['ls'].to_excel(writer, sheet_name='L-S_Drtn')
            self.excess_group['G10_excess'].to_excel(writer, sheet_name='G10_Ertn')
            self.RankIC.to_excel(writer, sheet_name='RankIC')
        if self._verbose:
            print(f"结果已经存入{self._output_performance}")

    def main(self):
        self.backtest()
        self.create_plot()
        self.output_result()


# %%
class IC_analysis:
    """
    Class to analyze the information coefficient
    
    Params
    ------
    factor           -pandas.DataFrame            -factor
    interval         -int or list                 -depends on rolling
                                                  -if rolling == True, int
                                                  -else list '['W',5]'
    """

    def __init__(self, factor, interval=5, rolling=True,
                 return_type='vwap', data_path='D:/实习/研报复现/stockData'):
        # data path
        if return_type == 'close':
            self._return_path = _os.path.join(data_path, 'pv\\adj_close_return.pkl')
        elif return_type == 'vwap':
            self._return_path = _os.path.join(data_path, 'pv\\adj_vwap_return.pkl')

        if rolling == True:
            if interval > 1:
                self._stock_return_daily = _pd.read_pickle(self._return_path).replace(0, _np.nan).rolling(
                    interval - 1).sum().dropna(axis=0, how='all')
                self._factor = factor.dropna(axis=0, how='all').shift(interval).dropna(axis=0, how='all')
            else:
                self._stock_return_daily = _pd.read_pickle(self._return_path).replace(0, _np.nan).dropna(axis=0,
                                                                                                         how='all')
                self._factor = factor.dropna(axis=0, how='all').shift(2).dropna(axis=0, how='all')

        else:
            self._dates_path = _os.path.join(data_path, f'calendar\\{interval[0]}dates.pkl')
            dates = _pd.read_pickle(self._dates_path)
            self._stock_return_daily = _pd.read_pickle(self._return_path).replace(0, _np.nan).rolling(
                interval[-1] - 1).sum().dropna(axis=0, how='all')

            mask = self._stock_return_daily.index.isin(dates)
            self._stock_return_daily[~mask] = _np.nan

            self._factor = factor.dropna(axis=0, how='all').shift(1).dropna(axis=0, how='all')
        # print("Successfully initialized...")

    def IC(self):
        """
        Internal to calculate IC
        """
        # print("Start calculating IC...")
        self.IC = self._factor.corrwith(self._stock_return_daily, axis=1).dropna()

    def RankIC(self):
        """
        Internal to calculate RankIC
        """
        # print("Start calculating IC...")
        self.RankIC = self._factor.corrwith(self._stock_return_daily, axis=1, method='spearman').dropna()

    def plot_IC(self):
        """
        External function to plot the IC
        """
        ic = self.IC

        fig, ax1 = _plt.subplots(figsize=(6, 4))
        ax1.bar(ic.index, ic, alpha=0.6, label='IC (left)', width=5)
        ax1.grid(False)

        ax2 = ax1.twinx()
        ax2.plot(ic.index, ic.cumsum(), label='Cummulative IC (right)', color=_r)
        ax2.grid(False)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines + lines2, labels + labels2, ncol=2, loc='lower center')
        _plt.show()

    def plot_RankIC(self):
        """
        External function to plot the RankIC
        """
        ic = self.RankIC

        fig, ax1 = _plt.subplots(figsize=(6, 4))
        ax1.bar(ic.index, ic, alpha=0.6, label='RankIC (left)', width=5)
        ax1.grid(False)

        ax2 = ax1.twinx()
        ax2.plot(ic.index, ic.cumsum(), label='Cummulative RankIC (right)', color=_r)
        ax2.grid(False)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines + lines2, labels + labels2, ncol=2, loc='lower center')
        _plt.show()


# ==============================useful tools=================================
def zscore(df):
    """
    Internal function to normalize the factor value for every row

    Params
    ------
    row             -pandas.Series                  -row of factor panel

    Outputs
    -------
    row             -pandas.Series                  -row after cleaned

    """
    # mean
    mean = df.mean(axis=1)

    # std
    std = df.std(axis=1)

    return (df.sub(mean, axis=0)).div(std, axis=0)


def MAD(self, row):
    """
    Internal Function to use MAD to clean outliers

    Params
    ------
    row             -pandas.Series                  -row of factor panel

    Outputs
    -------
    row             -pandas.Series                  -row after cleaned
    """

    # distance from the median
    n = self._outlier[-1]

    # MAD
    row_median = row.median()
    D = abs(row - row_median).median()

    # limit
    mask_ceiling = row_median + n * D
    mask_flooring = row_median - n * D

    # mask
    row[row > mask_ceiling] = mask_ceiling
    row[row < mask_flooring] = mask_flooring

    return row


def winsor(self, row):
    """
    Internal Function to use winsorization to clean outliers

    Params
    ------
    row             -pandas.Series                  -row of factor panel

    Outputs
    -------
    row             -pandas.Series                  -row after cleaned
    """
    # calculate the quantile
    lower = row.quantile(self._outlier[-1])
    upper = row.quantile(1 - self._outlier[-1])

    # substitute
    row[row < lower] = lower
    row[row > upper] = upper

    return row


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

    return [round(AnuualReturn * 100, 2), round(AnnualStd * 100, 2), SR, round(MaxDrawdown * 100, 2),
            end - start, round(win_rate * 100, 2), round(calmar, 2)]


def to_fctDataBase(fct, fct_name):
    path = f'D:\\实习\\研报复现\\因子\\备用因子库\\{fct_name}.feather'
    fct.astype(_np.float32).reset_index(names='date').to_feather(path)


def load_fct(fct_name):
    path = f'D:\\实习\\研报复现\\因子\\备用因子库\\{fct_name}.feather'
    fct = _pd.read_feather(path).set_index('date')
    return fct


def load_data(data_name):
    path = f'D:\\实习\\研报复现\\stockData\\{data_name}.pkl'
    data = _pd.read_pickle(path)
    return data
