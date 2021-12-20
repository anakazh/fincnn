import pandas as pd
import numpy as np
from fincnn.config.crsp_config import CRSP_START_DATE, CRSP_END_DATE, CRSP_DB_PATH, SPX_HISTORY_FOR_CRSP_PATH
import sqlite3
from pandas.tseries.holiday import USFederalHolidayCalendar
from tqdm.contrib.concurrent import thread_map
from pathlib import Path


def get_spx_at_end_date(path):
    # downloaded manually from https://www.ishares.com/us/products/239726/#tabsAll
    spx = pd.read_csv(path, na_values=['-'])
    spx = spx[(spx['Asset class'] == 'Equity') & (~spx.Ticker.isna())]
    spx = spx.rename({'Ticker': 'ticker', 'Name': 'name'}, axis=1)
    spx = spx.assign(date=pd.Timestamp(CRSP_END_DATE))
    return spx[['date', 'ticker', 'name']]


def get_share_class(name):
    if name.find('CLASS ') != -1:
        return name[name.find('CLASS')+6]
    else:
        return ''


def save_spx_history_from_wiki(spx_additions_path, spx_removals_path):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data = pd.read_html(url)
    # spx_today = data[0][['Symbol', 'Security', 'Date first added']]
    # spx_today = spx_today.rename({'Date first added': 'date_added',  # it's actually date last added
    #                               'Symbol': 'ticker',
    #                               'Security': 'name'}, axis=1)
    # spx_today = spx_today.assign(date_added=spx_today.date_added.fillna(start_date))
    # spx_today = spx_today.assign(date_added=spx_today.date_added.astype(str).apply(lambda x: re.sub('\s\(.*\)', '', x)))
    # spx_today = spx_today.assign(date_added=pd.to_datetime(spx_today.date_added))
    # date_mask = spx_today.date_added < pd.Timestamp(CRSP_START_DATE)
    # spx_today.loc[date_mask, 'date_added'] = pd.Timestamp(CRSP_START_DATE)
    # spx_today = spx_today.assign(date_removed=pd.Timestamp.today())
    # spx_today.to_csv('data/spx/spx_today.csv')

    spx_updates = data[1]  # from 2000 to 2021
    spx_updates.columns = ['date', 'ticker_added', 'name_added', 'ticker_removed', 'name_removed', 'reason']
    spx_updates = spx_updates.assign(date=pd.to_datetime(spx_updates.date))
    spx_updates = spx_updates[(spx_updates.date < CRSP_END_DATE) & (spx_updates.date >= CRSP_START_DATE)]

    spx_additions = spx_updates[['date', 'ticker_added', 'name_added']]
    spx_additions = spx_additions.rename({'ticker_added': 'ticker',
                                          'name_added': 'name'}, axis=1)
    spx_additions = spx_additions.dropna(subset=['ticker', 'name'], how='all')
    spx_additions.to_csv(spx_additions_path, index=False)

    spx_removals = spx_updates[['date', 'ticker_removed', 'name_removed']]
    spx_removals = spx_removals.rename({'ticker_removed': 'ticker',
                                        'name_removed': 'name'}, axis=1)
    spx_removals = spx_removals.dropna(subset=['ticker', 'name'], how='all')
    spx_removals.to_csv(spx_removals_path, index=False)
    # manually adjusted spx_additions.csv and spx_removals.csv afterwards


SPX_20201231_PATH = Path.cwd().parent.joinpath('spx/spx_20201231.csv')
SPX_ADDITIONS_PATH = Path.cwd().parent.joinpath('spx/spx_additions.csv')
SPX_REMOVALS_PATH = Path.cwd().parent.joinpath('spx/spx_removals.csv')


def get_spx_tables():
    spx_at_end_date = get_spx_at_end_date(SPX_20201231_PATH)
    spx_at_end_date = spx_at_end_date.assign(sh_class=spx_at_end_date.name.apply(lambda x: get_share_class(x)))
    spx_additions = pd.read_csv(SPX_ADDITIONS_PATH, parse_dates=['date'])
    spx_additions = spx_additions.assign(sh_class=spx_additions.name.apply(lambda x: get_share_class(x)))
    spx_removals = pd.read_csv(SPX_REMOVALS_PATH, parse_dates=['date'])
    spx_removals = spx_removals.assign(sh_class=spx_removals.name.apply(lambda x: get_share_class(x)))
    return spx_at_end_date, spx_additions, spx_removals


# define as sql request function for multithreading
def _execute_sql_statement(sql_statement):
    with sqlite3.connect(CRSP_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(sql_statement)
        results = cursor.fetchall()
    if len(results) == 0:
        return np.nan
    elif len(results) == 1:
        return results[0][0]
    else:
        print('The following statement returned multiple permnos')
        print(sql_statement)
        return results


def match_spx_tickers_to_permnos(spx_table, bday_delta1, bday_delta2):
    sql_statements = []
    calendar = USFederalHolidayCalendar()
    bday = pd.offsets.CustomBusinessDay(n=1, calendar=calendar)
    for i, (date, ticker, sh_class) in spx_table[['date', 'ticker', 'sh_class']].iterrows():
        sql_statement = f"""
        SELECT DISTINCT permno FROM OHLC
        WHERE Date BETWEEN {(date-bday_delta1*bday).strftime('%Y%m%d')} AND {(date+bday_delta2*bday).strftime('%Y%m%d')}
        AND ticker = '{ticker}'
        AND sh_class = '{sh_class}'
        """
        sql_statements.append(sql_statement)
    permnos = thread_map(_execute_sql_statement,
                         sql_statements,
                         desc='Executing sql statements',
                         max_workers=50)

    res = dict(zip(spx_table['ticker'], permnos))
    res = {k: v for k, v in res.items() if not isinstance(v, str)}

    def h(ticker):
        date1 = spx_table.loc[spx_table.ticker == ticker, 'date'].iloc[0] - pd.offsets.BDay(5)
        date2 = spx_table.loc[spx_table.ticker == ticker, 'date'].iloc[0] + pd.offsets.BDay(5)
        sql_statement = f"""
        SELECT Date, permno, ticker, sh_class FROM OHLC
        WHERE ticker = '{ticker}'
        AND Date BETWEEN {date1.strftime('%Y%m%d')} AND {date2.strftime('%Y%m%d')}
        """
        with sqlite3.connect(CRSP_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_statement)
            results = cursor.fetchall()
        return results
    for ticker in res.keys():
        print(ticker)
        print(h(ticker))

    spx_table = spx_table.assign(permno=permnos)
    return spx_table


def save_spx_history_with_permnos():
    spx_at_end_date, spx_additions, spx_removals = get_spx_tables()
    print('Matching permnos for spx_at_end_date table')
    spx_at_end_date = match_spx_tickers_to_permnos(spx_at_end_date, bday_delta1=1, bday_delta2=0)
    assert len(set(spx_at_end_date['permno'])) == len(spx_at_end_date['permno'])
    print('Matching permnos for spx_additions table')
    spx_additions = match_spx_tickers_to_permnos(spx_additions, bday_delta1=0, bday_delta2=1)
    print('Matching permnos for spx_removals table')
    spx_removals = match_spx_tickers_to_permnos(spx_removals, bday_delta1=1, bday_delta2=0)

    spx_at_end_date = spx_at_end_date.rename({'date': 'date_removed'}, axis=1)
    spx_additions = spx_additions.rename({'date': 'date_added'}, axis=1).sort_values(by='date_added')
    spx_removals = spx_removals.rename({'date': 'date_removed'}, axis=1)
    spx_removals = spx_at_end_date.append(spx_removals).sort_values(by='date_removed')

    # compication: several companies have been added and removed from the index multiple times
    # (i.e. spx_additions and spx_removals contain multiple rows with the same permno,
    # so cannot simply merge because this will result in missing rows)
    spx_add_nodup = spx_additions[~spx_additions['permno'].duplicated(keep='last')]
    spx_add_dup_last = spx_additions[spx_additions['permno'].duplicated(keep='last')]
    spx_rem_nodup = spx_removals[~spx_removals['permno'].duplicated(keep='last')]
    spx_rem_dup_last = spx_removals[spx_removals['permno'].duplicated(keep='last')]

    spx_history = spx_rem_nodup.merge(spx_add_nodup[['permno', 'date_added']], how='outer')
    spx_history = spx_history.assign(date_added=spx_history.date_added.fillna(pd.Timestamp(CRSP_START_DATE)))
    spx_history_dup = spx_rem_dup_last.merge(spx_add_dup_last[['permno', 'date_added']], how='outer')
    spx_history_dup = spx_history_dup.assign(date_added=spx_history_dup.date_added.fillna(pd.Timestamp(CRSP_START_DATE)))

    spx_history = spx_history.append(spx_history_dup)
    assert len(spx_history) == len(spx_removals)
    spx_history.to_csv(SPX_HISTORY_FOR_CRSP_PATH, index=False)


if __name__ == '__main__':

    save_spx_history_with_permnos()