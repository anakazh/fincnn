import pandas as pd
import numpy as np
import sqlite3
import csv
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import os
import shutil


def overwrite_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def str2float(x):
    try:
        return float(x)
    except:
        return np.nan


def csv_to_sqlite(csv_path='data/crsp/full_data.csv',
                  db_path='data/crsp/full_data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql_statement = "DROP TABLE IF EXISTS OHLC"
    cursor.execute(sql_statement)
    sql_statement = """CREATE TABLE OHLC (Date text NOT NULL, permno text NOT NULL, ticker text NOT NULL,
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
                "insert into OHLC values (:Date, :permno, :ticker, :Open, :High, :Low, :Close, :Volume, :cfac_pr, :cfac_vol)",
                {'Date': line['date'],
                 'permno': line['PERMNO'],
                 'ticker': line['TICKER'],
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


def sqlite_to_ticker_csvs(db_path='data/crsp/full_data.db'):
    """
    Saves data from sqlite DB to security csv files
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Due to possible change of ticker use PERMNO to identify stocks
    # https://libguides.stanford.edu/c.php?g=559845&p=6686228
    print("Retrieving the list of permanent numbers (PERMNO)")
    permnos = []
    for row in cursor.execute('SELECT DISTINCT permno FROM OHLC'):
        permnos.append(row[0])
    conn.close()

    overwrite_dir('../../data/raw')

    def _save_data_for_permno(permno):
        conn = sqlite3.connect(db_path)
        query = "SELECT Date, ticker, Open, High, Low, Close, Volume, cfac_pr, cfac_vol " \
                "FROM OHLC WHERE permno IN ({})".format(permno)

        df = pd.read_sql_query(query, conn)
        ticker = df.ticker.iloc[-1]
        savepath = 'data/raw/' + ticker + '_' + permno + '.csv'
        df = df.assign(Open=df.Open * df.cfac_pr,
                       High=df.High * df.cfac_pr,
                       Low=df.Low * df.cfac_pr,
                       Close=df.Close * df.cfac_pr,
                       Volume=df.Volume * df.cfac_vol)
        df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('Date').to_csv(savepath)
        conn.close()

    thread_map(_save_data_for_permno, permnos,
               max_workers=10,
               desc='Saving data to separate csv files')


if __name__ == '__main__':
    csv_to_sqlite()
    sqlite_to_ticker_csvs()
