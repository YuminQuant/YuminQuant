import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import Alpha as alpha


if __name__ == '__main__':

    fct = alpha.load_fct('J', fct_path=r"D:\实习\研报复现\QuantResearch\pv_alpha")
    alpha.Alpha(fct, 'J', verbose=True, freq='D', layer=10, rolling_ic=True).main()