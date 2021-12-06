import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    """
    Class for keeping track of train (and test) samples
    """
    name: str                      # 'train_2_stocks'
    permno_list: list              # e.g. ['14593', '10107'],  # AAPL, MSFT
    return_horizons: list          # [20, 60]
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
        df = handle_missing_values(df)
        df = df.loc[pd.to_datetime(self.start_date) - pd.offsets.BDay(max(self.return_horizons)):
                    pd.to_datetime(self.end_date)]

        summary = pd.DataFrame(index=[1., 0., -1.])
        for ret_horizon in self.return_horizons:
            # return in the NEXT n days, n=ret_horizon
            df[f'ret_{ret_horizon}'] = df.Close.shift(-ret_horizon).div(df.Close).sub(1).apply(np.sign)
            summary[f'ret_{ret_horizon}'] = df[f'ret_{ret_horizon}'].value_counts()
        return summary

    def describe(self, savepath=None):
        ret_cols = []
        for ret_horizon in self.return_horizons:
            ret_cols.append(f'ret_{ret_horizon}')
        summary = pd.DataFrame(np.zeros((3, len(ret_cols)), float),
                               index=[1., 0., -1.],
                               columns=ret_cols)
        for permno in tqdm(self.permno_list, desc=f'Calculating fraction of pos/neg returns in the {self.name}-sample'):
            summary = (summary + self.describe_permno(permno)).fillna(0)
        summary = summary.div(summary.sum())
        print(summary)
        if savepath is not None:
            summary.to_csv(savepath)


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