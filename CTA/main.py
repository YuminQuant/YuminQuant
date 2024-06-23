import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from CTA import CTA

if __name__ == '__main__':
    from strategy import strategy
    cta = CTA(strategy=strategy(), asset='IF', leverage=10, cost=0.002,
              data_path_min=r"D:\实习\研报复现\dataAll\futuresMinData\IF01.parquet",
              data_path_day=None,
              benchmark_path=r"D:\实习\研报复现\dataAll\futuresDayData\IF01.parquet",
              data_type='min', start='2010', end='2024', resample=None,
              plot='download',
              output_path=r"D:\实习\研报复现\QuantResearch\CTA_strats", verbose=True)
    cta.backtest()
    cta.output_result()
    cta.create_plot()