import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import EventDriven as event

if __name__ == '__main__':

    trade_list = pd.read_csv(...)
    ###
    # Make sure the trade_list has columns as ['date', 'stock_code', 'isEvent']
    #    date          stock_code    is Event
    #    2021-01-01      000001          1
    #    2021-01-05      000002          1
    #    ....
    # also make sure there is no duplicated row in the df!!!
    ###
    event.EventDriven(trade_list.query(""" date<='2024-02-29' """), 'EVENT1', event_period=60, tail=True,
                      holding_period='M', stock_pool='mkt', benchmark='mkt', verbose=True).main()
