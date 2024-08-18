# quantoolv3.0.0
# Yumin 20240501
import os as _os
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import pandas as pd
import seaborn as _sns
from tabulate import tabulate as _tabulate
from tqdm import tqdm as _tqdm
import statsmodels.api as _sm

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
                 st=True, up_down_limit=True, weight_limit=0.1, benchmark='mkt',
                 rolling_ic=True, plot='download',
                 return_type='vwap', data_path=r'D:\实习\研报复现\dataAll\stockDayData',
                 output_path=r'D:\实习\研报复现\QuantResearch\alpha_result', verbose=False):

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
        self._output_performance = _os.path.join(output_path, 'report')
        self._output_plot = _os.path.join(output_path, 'picture')
        if not _os.path.exists(self._output_performance):
            _os.makedirs(self._output_performance)

        if not _os.path.exists(self._output_plot):
            _os.makedirs(self._output_plot)

        self._output_performance = _os.path.join(self._output_performance,
                                                 name + '_' + self._freq + '.xlsx')
        self._output_plot = _os.path.join(self._output_plot,
                                          name + '_' + self._freq + '.png')

        # data path
        if return_type == 'close':
            self._return_path = _os.path.join(data_path, 'pv\\adj_close_return.pkl')
        elif return_type == 'vwap':
            self._return_path = _os.path.join(data_path, 'pv\\adj_vwap_return.pkl')

        self._ST_path = _os.path.join(data_path, 'pv\\st_dummy.pkl')
        self._upDownLimit_path = _os.path.join(data_path, 'pv\\upDownLimit.pkl')
        self._mv_path = _os.path.join(data_path, f'pv\\stock_mv.pkl')

        self._dates_path = _os.path.join(data_path, f'calendar\\{self._freq}dates.pkl')
        self._Ddates_path = _os.path.join(data_path, f'calendar\\Ddates.pkl')

        self._index_weight_path = _os.path.join(data_path, f'stockPool\\weight_{self._benchmark}.pkl')
        self._components_path = _os.path.join(data_path, f'stockPool\\stockPool_{self._stock_pool}.pkl')

        self._benchmark_path = _os.path.join(data_path, f'stockPool\\benchmark.pkl')
        self._barra_path = _os.path.join(data_path, f'Barra')

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
        if self._stock_pool:
            self._get_stock_pool()
        self._describe_factor()
        self._barra_corr()
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

        if self._stock_pool == '000985' or self._stock_pool == 'mkt':
            self._avg_coverage = (self._factor.count(axis=1) / self._status.count(axis=1)).mean()
        else:
            stock_pool = _pd.read_pickle(self._components_path)
            self._avg_coverage = (self._factor.count(axis=1) / stock_pool.count(axis=1)).mean()

    def _barra_corr(self):
        if self._verbose:
            print("开始计算Barra相关性")
        barra_corr = {}
        barra_list = _os.listdir(self._barra_path)
        for i in barra_list:
            barra = _os.path.join(self._barra_path, i)
            barra = _pd.read_feather(barra).set_index("date")
            barra_corr[i.split('.')[0]] = barra.corrwith(self._factor, axis=1).mean()
        self.barra_corr = _pd.Series(barra_corr)

    def _ic(self):
        if self._verbose:
            print("开始计算RankIC")
        _stock_return_daily = _pd.read_pickle(self._return_path).replace(0, _np.nan)
        if self._rolling_ic:
            if self._freq == 'D':
                self.RankIC = self._factor.shift(2).corrwith(_stock_return_daily, axis=1, method='spearman').dropna()
            else:
                if self._freq == 'W':
                    interval = 5
                elif self._freq == 'M':
                    interval = 20
                elif self._freq == 'Q':
                    interval = 60
                elif self._freq == 'Y':
                    interval = 250
                else:
                    raise ValueError("Wrong frequency")

                self.RankIC = self._factor.shift(interval).corrwith(_stock_return_daily.rolling(interval - 1).sum()
                                                                    .dropna(axis=0, how='all'), axis=1,
                                                                    method='spearman').dropna()
        else:
            if self._freq == 'W':
                interval = 5
            elif self._freq == 'M':
                interval = 20
            elif self._freq == 'Q':
                interval = 60
            elif self._freq == 'Y':
                interval = 250
            else:
                raise ValueError("Wrong frequency")

            dates = _pd.read_pickle(self._dates_path)
            _stock_return_daily = _pd.read_pickle(self._return_path).replace(0, _np.nan).rolling(
                interval - 1).sum().dropna(axis=0, how='all')
            mask = _stock_return_daily.index.isin(dates)
            _stock_return_daily[~mask] = _np.nan
            self.RankIC = self._factor.shift(1).corrwith(_stock_return_daily, axis=1, method='spearman').dropna()

        self.ic = self.RankIC.mean()
        self.icir = self.ic / self.RankIC.std()

        # # 改变因子方向
        # if self.ic < 0:
        #     self._factor *= -1

    def _get_trading_panel(self):

        dates = list(_pd.read_pickle(self._dates_path))
        Ddates = list(_pd.read_pickle(self._Ddates_path))

        if self._freq != 'D':
            self._factor = self._factor.loc[self._factor.index.isin(dates)]
        else:
            self._factor = self._factor.shift(2).dropna(axis=0, how='all')

        ####
        self._period = len(self._factor)
        ####

        # if self._stock_pool:
        #     self._get_stock_pool()
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
            stock_pool = _pd.read_pickle(self._components_path)
            stock_pool = stock_pool[stock_pool.index.isin(self._factor.index)]
            self._factor = stock_pool.mul(self._factor, axis=0).dropna(axis=1, how='all').dropna(axis=0, how='all')

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

    def _ls_rtn(self, group):
        rtns = (group['returns'] * group['ls_weight']).sum()
        return rtns

    def _group(self, layer):
        if self._verbose:
            print('开始分层回测...')
        # 使用stack函数将其转换为长格式
        self.ls_w = self._factor.div(self._factor.abs().sum(axis=1), axis=0).stack().to_frame()
        self.ls_w.columns = ['ls_weight']
        self._factor = self._factor.rank(axis=1, pct=True, method='dense').stack().to_frame()
        self._factor.columns = ['factor_rank']
        self._factor = self._factor.join(self.ls_w, how='inner')
        self._factor.rename_axis(['date', 'stock_code'], inplace=True)
        # 对每天进行分组，并将股票等分为layer层
        self._factor['quantile'] = self._factor.groupby(level=0)['factor_rank'].transform(
            lambda x: _pd.qcut(x, layer, labels=False, duplicates='drop')
        )

        # 使用stack函数将其转换为长格式
        _stock_return_daily = _pd.read_pickle(self._return_path).stack().to_frame()
        _stock_return_daily.columns = ['returns']
        _stock_return_daily.rename_axis(['date', 'stock_code'], inplace=True)

        # 根据日期和股票代码将收益数据与因子数据合并
        self._factor = self._factor.join(_stock_return_daily, how='inner')
        # 计算每层的平均收益率
        date = self._factor.index.get_level_values(0)
        grouped = self._factor.groupby([date, 'quantile'])
        self.group_return = grouped['returns'].mean().unstack()
        self._factor['weight'] = 1 / grouped.transform('size')

        self.group_return.columns = ['G' + str(_) for _ in range(1, layer + 1)]
        self.group_return['factor_return'] = self._factor.groupby(level=0).apply(self._ls_rtn)

        if self.ic > 0:
            query_top = f""" quantile=={self._layer - 1} """
            query_bottom = """ quantile==0 """
        else:
            query_top = """ quantile==0 """
            query_bottom = f""" quantile=={self._layer - 1} """
            self.group_return['factor_return'] *= -1
        self.topGroupWeight = self._factor.query(query_top)['weight'].unstack().fillna(0)
        self.bottomGroupWeight = self._factor.query(query_bottom)['weight'].unstack().fillna(0)
        self.lsWeight = self._factor['ls_weight'].unstack().fillna(0)

    def _excess_group(self):
        if self._verbose:
            print('开始计算分层超额...')
        if self._benchmark == 'mkt' or self._benchmark == '000985':
            benchmark = _pd.read_pickle(self._return_path).mean(axis=1).loc[
                        self.first_trading_day:self.last_trading_day]
        else:
            stock_pool = _pd.read_pickle(self._index_weight_path)
            benchmark = (_pd.read_pickle(self._return_path).mul(stock_pool, axis=0).dropna(axis=0, how='all').sum(axis=1)
                         .loc[self.first_trading_day:self.last_trading_day])
        self.excess_group = self.group_return.sub(benchmark, axis=0).dropna(axis=0, how='all')
        self.excess_group.columns = [_ + '_excess' for _ in self.excess_group.columns]

    def performance_panel(self, portfolio_name):
        """
        External to get the group performance panel
        """
        # format the portfolio
        if '_excess' in portfolio_name:
            portfolio = (self.excess_group[portfolio_name] + 1).cumprod()
            portfolio.loc[portfolio.index[0] - _pd.Timedelta('1D')] = 1
            portfolio.sort_index(inplace=True)
        elif 'G' in portfolio_name or 'factor_return' in portfolio_name:
            portfolio = (self.group_return[portfolio_name] + 1).cumprod()
            portfolio.loc[portfolio.index[0] - _pd.Timedelta('1D')] = 1
            portfolio.sort_index(inplace=True)
        else:
            raise ValueError("Wrong portfolio")

        # calculate yearly performance
        indicator = portfolio.groupby(_pd.Grouper(freq='Y')).apply(performance_stats)
        idx = indicator.index
        indicator = _pd.concat([_pd.Series(x) for x in indicator], axis=1)

        if 'G' in portfolio_name or portfolio_name == 'top_excess':
            turnover = round(
                (_pd.DataFrame(self.topGroupWeight.diff().abs().sum(axis=1)
                               .groupby(_pd.Grouper(freq='Y')).sum())) * 100, 3) / 250
        elif 'factor_return' in portfolio_name:
            turnover = round(
                (_pd.DataFrame(self.lsWeight.abs().diff().abs().sum(axis=1)
                               .groupby(_pd.Grouper(freq='Y')).sum())) * 100, 3) / 250
        else:
            raise ValueError("Wrong portfolio")

        turnover.index = turnover.index.strftime('%Y-%m-%d')
        turnover.columns = ['Turnover(%)']

        indicator.index = ['Return (%)', 'Vol (%)', 'SR', 'Max Drawdown (%)', 'Drawdown Period', 'Winning Rate (%)',
                           'Calmar']
        indicator.columns = idx.strftime('%Y-%m-%d')

        # mean
        indicator = _pd.concat([indicator.T, turnover], axis=1)
        indicator.loc['mean'] = round(indicator.mean(), 3)
        indicator.index.name = portfolio_name
        return indicator.dropna(axis=0)

    def print_performance(self, portfolio_name):
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
        cumulative_group = (1 + self.group_return).cumprod()

        # 创建颜色映射
        cmap = _plt.get_cmap('coolwarm')
        colors = [cmap(i) for i in _np.linspace(0, 1, len(cumulative_excess.columns))]

        # 创建包含两个子图的图表
        fig, axes = _plt.subplots(6, 1, figsize=(8, 12))

        # 在第一个子图上画IC
        if not self._rolling_ic:
            bar_container = axes[0].bar(self.RankIC.index, self.RankIC, alpha=0.6, label='RankIC (left)', width=5)
        else:
            bar_container = axes[0].fill_between(self.RankIC.index, self.RankIC, alpha=0.8, zorder=1)

        axes2 = axes[0].twinx()
        line_container, = axes2.plot(self.RankIC.index, self.RankIC.cumsum(), label='Cummulative RankIC (right)',
                                     color=_r)
        axes2.grid(False)
        # 创建图例，包含两个y轴的图例
        axes[0].legend([bar_container, line_container], ['RankIC (left)', 'Cummulative RankIC (right)'], ncol=2,
                       loc='upper center', bbox_to_anchor=(0.5, 1.05))

        # 在第二个子图上绘制累计收益率曲线
        lines = []
        labels = []
        for i, (column, color) in enumerate(zip(cumulative_group.columns[:-1], colors)):
            if i == 0 or i == len(cumulative_group.columns[:-1]) - 1:
                line, = axes[1].plot(cumulative_group.index, cumulative_group[column],
                                     label=column, color=color, zorder=3)
                lines.append(line)
                labels.append(column)
            else:
                axes[1].plot(cumulative_group.index, cumulative_group[column], color=color, alpha=0.8, zorder=1)
        axes[1].set_ylim(ymin=0)
        # 副轴画L-S
        ax_twin = axes[1].twinx()
        line, = ax_twin.plot(cumulative_group.index, cumulative_group['factor_return'],
                             label='FR', color='forestgreen', alpha=1)

        left_ticks = axes[2].get_yticks()
        num_intervals = len(left_ticks) - 1
        right_ticks = _np.linspace(ax_twin.get_ybound()[0], ax_twin.get_ybound()[1], num_intervals + 1)
        ax_twin.set_yticks(right_ticks)

        lines.append(line)
        labels.append('factor_return (right)')
        ax_twin.grid(False)

        axes[1].legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3,
                       fancybox=True, shadow=True)

        # 在第三个子图上绘制累计超额收益率曲线
        for i, (column, color) in enumerate(zip(cumulative_excess.columns, colors)):
            if i == 0 or i == len(cumulative_excess.columns) - 1:
                axes[2].plot(cumulative_excess.index, cumulative_excess[column], label=column, color=color,
                             zorder=3)

            else:
                axes[2].plot(cumulative_excess.index, cumulative_excess[column], color=color, alpha=0.8, zorder=1)

        # 显示图例，设置为3列，位置为上部中心
        axes[2].legend(ncol=2, loc='upper center')

        # 头组尾组换手率
        top_turnover = self.topGroupWeight.diff().abs().sum(axis=1)  #.replace(0, _np.nan).dropna()
        bottom_turnover = self.bottomGroupWeight.diff().abs().sum(axis=1)  #.replace(0, _np.nan).dropna()
        # ls_turnover = self.lsWeight.abs().diff().abs().sum(axis=1)#.replace(0, _np.nan).dropna()

        axes[3].fill_between(bottom_turnover.index, bottom_turnover, where=(bottom_turnover >= 0),
                             interpolate=True, color=colors[0], label='G1_turnover', alpha=0.8)
        axes[3].fill_between(top_turnover.index, top_turnover, where=(top_turnover >= 0),
                             interpolate=True, color=colors[-1], label=f'G{self._layer}_turnover', alpha=0.8)
        # axes[3].plot(ls_turnover.index, top_turnover, color='forestgreen', label=f'L-S_turnover', alpha=0.5)
        axes[3].legend(ncol=2, loc='upper center')

        # 分组柱状图
        return_group = self.group_return.groupby(_pd.Grouper(freq='Y')).sum().mean().iloc[:-1]
        for i, (idx, color) in enumerate(zip(return_group.index, colors)):
            if i == 0 or i == len(return_group.index) - 1:
                axes[4].bar(idx, return_group[idx], color=color)
            else:
                axes[4].bar(idx, return_group[idx], color=color)

        # barra相关性图
        axes[5].bar(self.barra_corr.index, self.barra_corr, color=_b)
        axes[5].set_xticks(_np.arange(0, len(self.barra_corr.index)), self.barra_corr.index, rotation=45)

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
        desc_info = _pd.Series([self._name, self.first_trading_day.strftime('%Y-%m-%d'),
                                self.last_trading_day.strftime('%Y-%m-%d'),
                                self._fct_mean, self._fct_std, self._fct_skew, self._fct_kurt,
                                self._fct_median, self._fct_min, self._fct_max, self._avg_coverage,
                                self._period, self.ic, self.icir, self._stock_pool, self._benchmark],
                               index=['name', 'start_date', 'end_date', 'mean', 'std', 'skew',
                                      'kurt', 'median', 'min', 'max', 'coverage', 'efficient_period',
                                      f'{self._freq}_RankIC', f'{self._freq}_RankICIR', 'stock_pool', 'benchmark'],
                               name='info')

        with _pd.ExcelWriter(self._output_performance) as writer:
            # 将每个DataFrame写入不同的sheet
            desc_info.to_excel(writer, sheet_name='fct_info')
            topGroupName = f'G{self._layer}_excess' if self.ic > 0 else 'G1_excess'

            self.performance_panel('factor_return').to_excel(writer, sheet_name='LS_Performance')
            self.performance_panel(topGroupName).to_excel(writer, sheet_name='TopGroupPerformance')
            data = _pd.concat([self.group_return['factor_return'], self.excess_group[topGroupName],
                               self.RankIC], axis=1)
            data.to_excel(writer, sheet_name='DailySeries')

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
                 return_type='vwap', data_path='D:/实习/研报复现/dataAll/stockDayData'):
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
def row_rolling_std(arr, window_size):
    # 计算每一列的滚动标准差
    # 首先，我们需要一个窗口视图
    shape = (arr.shape[0] - window_size + 1, window_size) + arr.shape[1:]
    strides = (arr.strides[0],) + arr.strides
    windowed_arr = _np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    # 计算每个窗口的标准差
    return _np.nanstd(windowed_arr, axis=1, ddof=1)


def row_rolling_mean(arr, window_size):
    # 计算每一列的滚动标准差
    # 首先，我们需要一个窗口视图
    shape = (arr.shape[0] - window_size + 1, window_size) + arr.shape[1:]
    strides = (arr.strides[0],) + arr.strides
    windowed_arr = _np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    # 计算每个窗口的标准差
    return _np.nanmean(windowed_arr, axis=1)


def row_rolling_sum(arr, window_size):
    # 计算每一列的滚动标准差
    # 首先，我们需要一个窗口视图
    shape = (arr.shape[0] - window_size + 1, window_size) + arr.shape[1:]
    strides = (arr.strides[0],) + arr.strides
    windowed_arr = _np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    # 计算每个窗口的标准差
    return _np.nansum(windowed_arr, axis=1)


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


def style_reg_neutral(fct, styles: list):
    fct_df = fct.copy()
    for style in styles:
        if style == 'SECTOR':
            style_df = load_data("ind_citics")
        elif style == 'cap':
            style_df = load_data("pv\\stock_mv")
        elif style in ['BETA', 'BTOP', 'EARNYILD', 'GROWTH', 'LEVERAGE',
                       'LIQUIDTY', 'MOMENTUM', 'RESVOL', 'SIZE', 'SIZENL']:
            style_df = load_data(f'Barra\\{style}')
        else:
            raise ValueError('Wrong Style')

        res = []
        for idx in _tqdm(fct_df.index):
            if style == 'SECTOR':
                sector_df = _pd.get_dummies(style_df.loc[idx], drop_first=True, dtype=int)
                temp = _pd.concat([fct_df.loc[idx], sector_df], axis=1).dropna()
            else:
                temp = _pd.concat([fct_df.loc[idx], style_df.loc[idx]], axis=1).dropna()

            X = temp.iloc[:, 1:]
            X = _sm.add_constant(X)
            y = temp.iloc[:, 0]

            model = _sm.OLS(y, X)
            results = model.fit()
            res.append(results.resid)

        fct_df = _pd.concat(res, axis=1).T
        fct_df.index = fct.index

    return fct_df


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


def to_fctDataBase(fct, fct_name, base_path=f'D:\\实习\\研报复现\\QuantResearch\\pv_alpha'):
    path = _os.path.join(base_path, fct_name + '.feather')
    fct = fct.astype(_np.float32)
    fct.index.name = 'date'
    fct.to_feather(path)


def load_fct(fct_name, fct_path='D:\\实习\\研报复现\\QuantResearch\\备用因子库'):
    path = _os.path.join(fct_path, fct_name + '.feather')
    fct = _pd.read_feather(path)
    return fct


def load_data(data_name, start=None, end=None):
    try:
        path = f'D:\\实习\\研报复现\\dataAll\\stockDayData\\{data_name}.pkl'
        data = _pd.read_pickle(path)
    except:
        path = f'D:\\实习\\研报复现\\dataAll\\stockDayData\\{data_name}.feather'
        data = _pd.read_feather(path).set_index('date')

    if start is not None:
        data = data.loc[start:]
    if end is not None:
        data = data.loc[:end]

    return data
