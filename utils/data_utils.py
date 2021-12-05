import pandas as pd
import numpy as np
import json
import tqdm
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    """
    Class for keeping track of train (and test) samples
    """
    name: str                      # 'train_2_stocks'
    permno_list: list              # e.g. ['14593', '10107'],  # AAPL, MSFT
    return_horizons: list          # [20_day, 60_day]
    start_date: str                # e.g. '19930101'
    end_date: str                  # e.g. '19991231'

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            attributes = json.load(f)
        return cls(**attributes)

    def save(self, filename=None):
        filename = filename or f'samples/{self.name}.json'
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f)

    def describe_permno(self, permno):
        filepath = [f for f in Path('data/raw/').rglob(f'*{permno}.csv')][0]
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True).sort_index()
        df = df.loc[pd.to_datetime(self.start_date) - pd.offsets.BDay(self.return_horizon):pd.to_datetime(self.end_date)]
        df['ret'] = df.Close.shift(-self.return_horizon).div(df.Close).sub(1) # return in the NEXT n days, n=return_horizon
        df = handle_missing_values(df)
        df['ret_sign'] = df.ret.apply(np.sign)
        permno_summary = df[['ret_sign', 'ret']].groupby('ret_sign').count()
        return permno_summary

    def describe(self, save_summary=True):
        summary = pd.DataFrame([0, 0, 0],
                               index=pd.Index([1.0, 0, -1.0], name='ret_sign'),
                               columns=['ret'])
        for permno in tqdm(self.permno_list, desc='Calculating fraction of pos/neg returns in the sample'):
            summary = (summary + self.describe_permno(permno)).fillna(0)
        summary['ret_frac'] = summary.div(summary.sum())
        print(summary)
        if save_summary:
            summary.to_csv(f'samples/{self.name}_summary.csv')


def handle_missing_values(data):
    """
    Fill missing Volume with 0, drop rows with any missing OHLC prices
    """
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close'], how='any')
    data = data.assign(Volume=data.Volume.fillna(0))
    # Old logic:
    # Drop rows where only Close and Volume are available
    # Open: if missing, take previous day Close
    # High: if missing, take max(Open, Close)
    # Low: if missing, take min(Open, Close)
    # Close: if missing, drop the row
    # Volume: if missing, fill with 0
    #data = data.dropna(subset=['Open', 'High', 'Low'], how='all')
    #data = data.dropna(subset=['Close'])
    #data = data.assign(Open=data.Open.fillna(data.Close.shift(1)),
    #                   High=data.High.fillna(data.Close),
    #                   Low=data.Low.fillna(data.Close),
    #                   Volume=data.Volume.fillna(0))
    return data