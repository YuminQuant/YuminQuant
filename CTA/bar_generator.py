import os as _os
import pandas as _pd


class BarGenerator:
    def __init__(self, start=None, end=None, data_path_min=None, data_path_day=None, resample=None):
        self.data_path_min = data_path_min
        self.data_path_day = data_path_day
        self._start = start
        self._end = end
        self.resample = resample
        self.time_mask_1515 = None

    def _load_data(self, file_path):
        _, file_extension = _os.path.splitext(file_path)

        # 根据文件后缀名选择读取方法
        if file_extension == '.pkl':
            df = _pd.read_pickle(file_path)
        elif file_extension == '.feather':
            df = _pd.read_feather(file_path)  # .set_index('date')
        elif file_extension == '.parquet':
            df = _pd.read_parquet(file_path)
        elif file_extension == '.csv':
            df = _pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_extension == '.xlsx':
            df = _pd.read_excel(file_path, index_col=0, parse_dates=True)
            df.index = _pd.to_datetime(df.index)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        if self._start is not None:
            df = df.loc[self._start:]
        if self._end is not None:
            df = df.loc[:self._end]

        return df

    def _close_check(self, time_idx):
        """
        探测收盘
        :return:
        """
        # 定义要检查的时间范围
        time_start_1501 = _pd.Timestamp('1900-01-01 15:01:00').time()  # 使用一个不相关的日期，只关注时间
        time_end_1515 = _pd.Timestamp('1900-01-01 15:15:00').time()

        # 检查索引中的时间是否在指定范围内
        time_mask_1515 = (time_idx.time >= time_start_1501) & (time_idx.time <= time_end_1515)

        # 如果有匹配的数据，打印True
        if time_mask_1515.any():
            self.time_mask_1515 = True
        else:
            self.time_mask_1515 = False

    def bar_generator(self):
        if self.data_path_day is not None:
            df_day = self._load_data(self.data_path_day)
        else:
            df_day = None
        if self.data_path_min is not None:
            df_min = self._load_data(self.data_path_min)
        else:
            df_min = None
        if df_min is None and df_day is None:
            raise ValueError("No data be loaded")

        # 是否需要降频传入数据
        if self.resample is not None and self.data_path_min is not None:
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'money': 'sum',
                'open_interest': 'last',
                'symbol': 'last'
            }
            df_min = df_min.resample(self.resample).agg(agg_dict)

        # 初始化生成器状态
        min_index = 0
        day_index = 0
        # last_day_data = None

        if df_day is not None and df_min is not None:
            if self.time_mask_1515 is not None:
                self._close_check(df_min.index)

            while min_index < len(df_min) and day_index < len(df_day):
                # 获取当前分钟和日期数据
                min_data = df_min.iloc[min_index]
                day_data = df_day.iloc[day_index]

                # 盘内，传入分钟数据
                if min_data.name.hour < 15:
                    yield min_data, None
                    min_index += 1
                # 如果收盘，传入日级数据
                elif min_data.name.hour == 15 and not self.time_mask_1515:
                    yield min_data, day_data
                    day_index += 1
                    min_index += 1
                # 国债期货
                elif (min_data.name.hour == 15 and
                      min_data.name.minute == 15 and
                      self.time_mask_1515):
                    yield min_data, day_data
                    min_index += 1
                    day_index += 1
                # 如果分钟数据的日期小于日级数据的日期，则不传入日级数据
                # 主要为了夜盘
                elif min_data.name.date() < day_data.name.date():
                    yield min_data, None
                    min_index += 1

        elif df_min is not None:
            if self.time_mask_1515 is not None:
                self._close_check(df_min.index)
            while min_index < len(df_min):
                min_data = df_min.iloc[min_index]
                # 盘内，传入分钟数据
                if min_data.name.hour < 15:
                    yield min_data, False
                    min_index += 1
                # 如果收盘，传入日级数据
                elif min_data.name.hour == 15 and not self.time_mask_1515:
                    yield min_data, True
                    min_index += 1
                elif (min_data.name.hour == 15 and
                      min_data.name.minute == 15 and
                      self.time_mask_1515):
                    yield min_data, True
                    min_index += 1
                else:
                    # 传入夜盘数据
                    yield min_data, False
                    min_index += 1

        elif df_day is not None:
            while day_index < len(df_day):
                day_data = df_day.iloc[day_index]
                yield day_data
                day_index += 1
