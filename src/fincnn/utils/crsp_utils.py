import pandas as pd
import numpy as np
import sqlite3
import csv
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from fincnn.config.crsp_config import SPX_HISTORY_FOR_CRSP_PATH, CRSP_CSV_PATH, CRSP_DB_PATH
from fincnn.config.paths_config import RAW_DATA_PATH


def str2float(x):
    try:
        return float(x)
    except:
        return np.nan


def csv_to_sqlite():
    csv_path = CRSP_CSV_PATH
    conn = sqlite3.connect(CRSP_DB_PATH)
    cursor = conn.cursor()
    sql_statement = "DROP TABLE IF EXISTS OHLC"
    cursor.execute(sql_statement)
    sql_statement = """CREATE TABLE OHLC (Date text NOT NULL, permno text NOT NULL, ticker text NOT NULL,
                                          sh_class text,
                                          Open float, High float, Low float, Close float,
                                          Volume float, cfac_pr float, cfac_vol float)"""
    #
    cursor.execute(sql_statement)
    conn.commit()
    with open(csv_path, 'r') as infile, conn:
        content = csv.DictReader(infile, delimiter=',')  # csv generator to the file, will be read line by line
        cursor = conn.cursor()
        for line in tqdm(content, desc='Inserting data into OHLC table in progress'):
            # line is a dict, where each column name is the key
            # no need to sanitize the header row, that was done automatically upon reading the file
            cursor.execute(
                """INSERT into OHLC values
                (:Date, :permno, :ticker, :sh_class, :Open, :High, :Low, :Close, :Volume, :cfac_pr, :cfac_vol)
                """,
                {'Date': line['date'],
                 'permno': line['PERMNO'],
                 'ticker': line['TICKER'],
                 'sh_class': line['SHRCLS'],
                 'Open': str2float(line['OPENPRC']),
                 'High': str2float(line['ASKHI']),
                 'Low': str2float(line['BIDLO']),
                 'Close': str2float(line['PRC']),
                 'Volume': str2float(line['VOL']),
                 'cfac_pr': str2float(line['CFACPR']),
                 'cfac_vol': str2float(line['CFACSHR'])
                 })
        # no file closing, that is automatically done by 'with open()'
        # no db close or commit - the DB connection is in a context manger,
        # that'll be done automatically (the commit - if there are no exceptions)


def sqlite_to_spx_csvs():
    """
    Saves data for SPX constituents from sqlite DB to security csv files
    """
    RAW_DATA_PATH.mkdir(exist_ok=False)
    spx_history = pd.read_csv(SPX_HISTORY_FOR_CRSP_PATH, parse_dates=['date_added', 'date_removed'])
    sql_statements = {}
    for i, (ticker, permno, start_date, end_date) in spx_history[['ticker', 'permno', 'date_added', 'date_removed']].iterrows():
        start_dt_str = start_date.strftime('%Y%m%d')
        end_dt_str = end_date.strftime('%Y%m%d')
        sql_statement = f"""
            SELECT Date, ticker, Open, High, Low, Close, Volume, cfac_pr, cfac_vol FROM OHLC
            WHERE permno IN ('{permno}')
            AND Date BETWEEN {start_dt_str} AND {end_dt_str}
            """  # permno IN - for speed

        savepath = RAW_DATA_PATH.joinpath(f'{ticker}_{permno}_{start_dt_str}_{end_dt_str}.csv')
        # if stock was added and removed from index multiple times there will be multiple csv files for the same stock
        sql_statements[savepath] = sql_statement

    def _save_data_to_csv(savepath):
        sql_statement = sql_statements[savepath]
        with sqlite3.connect(CRSP_DB_PATH) as conn:
            df = pd.read_sql_query(sql_statement, conn)
            # Adjustment for price and volume:
            # https://www.crsp.org/products/documentation/crsp-calculations
            df = df.assign(Open=df.Open / df.cfac_pr,
                           High=df.High / df.cfac_pr,
                           Low=df.Low / df.cfac_pr,
                           Close=df.Close / df.cfac_pr,
                           Volume=df.Volume * df.cfac_vol)
            df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('Date').to_csv(savepath)

    thread_map(_save_data_to_csv,
               list(sql_statements.keys()),
               max_workers=10,
               desc='Saving SPX data to separate csv files')


if __name__ == '__main__':
    #csv_to_sqlite()
    sqlite_to_spx_csvs()
