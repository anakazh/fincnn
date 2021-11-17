import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm


def download_cboe_tickers():
    # List of CBOE tickers: https://www.cboe.com/us/equities/market_statistics/listed_symbols/
    tickers = pd.read_csv('https://www.cboe.com/us/equities/market_statistics/listed_symbols/csv')
    tickers.to_csv('data/cboe_tickers.csv')


def calculate_adjusted_prices(df, column, adj_column_verify=None):
    """
    Adjust prices with respect to stock splits and dividends
    Source: https://joshschertz.com/2016/08/27/Vectorizing-Adjusted-Close-with-Python/

    :param df: DataFrame with raw prices along with 'Dividends' and 'Stock Splits' values
    :param column: String of which price column should have adjusted prices created for it
    :return: DataFrame with the addition of the adjusted price column
    """
    adj_column = 'adj_' + column
    # Reverse the DataFrame order, sorting by date in descending order
    df.sort_index(ascending=False, inplace=True)

    price_col = df[column].values
    split_col = df['Stock Splits'].values
    dividend_col = df['Dividends'].values
    adj_price_col = np.zeros(len(df.index))
    adj_price_col[0] = price_col[0]
    for i in range(1, len(price_col)):
        adj_price_col[i] = round((adj_price_col[i - 1] + adj_price_col[i - 1] *
                                 (((price_col[i] * split_col[i - 1]) - price_col[i - 1] -
                                   dividend_col[i - 1]) / price_col[i - 1])), 4)
    df[adj_column] = adj_price_col
    # Change the DataFrame order back to dates ascending
    df.sort_index(ascending=True, inplace=True)
    return df


def adjust_prices(df, column):
    """
    Adjust prices with respect to stock splits and dividends
    Logic and sample calculation: https://help.yahoo.com/kb/SLN28256.html

    :param df: DataFrame with raw prices along with 'Dividends' and 'Stock Splits' values
    :param column: Name of column with prices to adjust
    :return: DataFrame with 'adj_' + column added
    """
    adj_column = 'adj_' + column
    df.sort_index(ascending=False, inplace=True)
    df['split_multiplier'] = df['Stock Splits'].replace(0, 1).apply(lambda x: 1/x).cumprod().shift(1).fillna(1)
    df['div_multiplier'] = df[['Dividends', column]].apply(lambda x: 1 - (x[0]/x[1]), axis=1).cumprod().shift(1).fillna(1)
    df[adj_column] = df[column] * df['split_multiplier'] * df['div_multiplier']
    df.sort_index(ascending=True, inplace=True)
    df.drop(['split_multiplier', 'div_multiplier'], axis=1, inplace=True)
    return df


def adjust_open_high_low(df, inplace=True):
    """
    Adjust OHLC prices using the ratio of Close/Adj_Close
    :param df: DataFrame with raw prices and 'Adj Close' column
    :param inplace: If True replaces original price columns with adjusted one
    :return: DataFrame with adjusted prices and 'Adj Close' column
    """
    assert 'Adj Close' in df.columns
    df['adj_multiplier'] = df['Adj Close'] / df['Close']
    if inplace:
        df['Open'] = df['Open'] * df['adj_multiplier']
        df['High'] = df['High'] * df['adj_multiplier']
        df['Low'] = df['High'] * df['adj_multiplier']
    else:
        df['Adj Open'] = df['Open'] * df['adj_multiplier']
        df['Adj High'] = df['High'] * df['adj_multiplier']
        df['Adj Low'] = df['High'] * df['adj_multiplier']
    df.drop('adj_multiplier', axis=1, inplace=True)
    return df


if __name__ == '__main__':

    #download_cboe_tickers()

    tickers_table = pd.read_csv('data/cboe_tickers.csv')

    #for ticker_str in  tickers_table.Name.to_list():
        # adjusted and unadjusted OHLC are the same
        #ticker = yf.Ticker(ticker_str)
        #hist = ticker.history(period="max", back_adjust=True, autoadjust=True).drop(['Dividends', 'Stock Splits'], axis=1)

    tickers = ' '.join(tickers_table.Name.to_list())
    print('Download in progress:')
    all_data = yf.download(tickers, group_by='Ticker',
                           period='max',
                           #start="2000-01-01", end="2020-01-01",
                           actions=False, # do not download dividend + stock splits data
                           #autoadjust=True, doesn't adjust OHLC
                           show_errors=True, # doesn't print errors if True
                           threads=10)

    # adjusted and unadjusted Close should be the same in the last row ??? they are not
    tickers = tickers.split()
    for ticker_str in tqdm(tickers, desc='Adjusting and saving prices to csv: '):
        df = all_data[ticker_str].dropna(how='all')
        if not df.empty:
            df = adjust_open_high_low(df, inplace=True)
            df = df.drop('Close', axis=1).rename({'Adj Close': 'Close'}, axis=1)
            df.to_csv('data/raw/'+ticker_str+'.csv')
