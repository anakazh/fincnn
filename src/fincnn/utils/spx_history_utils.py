import pandas as pd
from fincnn.config.crsp_config import CRSP_START_DATE, CRSP_END_DATE


def get_spx_at_end_date():
    # downloaded manually from https://www.ishares.com/us/products/239726/#tabsAll
    spx = pd.read_csv('data/spx/spx_20201231.csv', na_values=['-'])
    spx = spx[(spx['Asset class'] == 'Equity') & (~spx.Ticker.isna())]
    spx = spx.rename({'Ticker': 'ticker', 'Name': 'name'}, axis=1)
    spx = spx.assign(date=pd.Timestamp(CRSP_END_DATE))
    return spx[['date', 'ticker', 'name']]


def get_share_class(name):
    if name.find('CLASS ') != -1:
        return name[name.find('CLASS')+6]
    else:
        return ''


def save_spx_history_from_wiki():
    start_date = CRSP_START_DATE
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data = pd.read_html(url)
    # spx_today = data[0][['Symbol', 'Security', 'Date first added']]
    # spx_today = spx_today.rename({'Date first added': 'date_added',  # it's actually date last added
    #                               'Symbol': 'ticker',
    #                               'Security': 'name'}, axis=1)
    # spx_today = spx_today.assign(date_added=spx_today.date_added.fillna(start_date))
    # spx_today = spx_today.assign(date_added=spx_today.date_added.astype(str).apply(lambda x: re.sub('\s\(.*\)', '', x)))
    # spx_today = spx_today.assign(date_added=pd.to_datetime(spx_today.date_added))
    # date_mask = spx_today.date_added < pd.Timestamp(start_date)
    # spx_today.loc[date_mask, 'date_added'] = pd.Timestamp(start_date)
    # spx_today = spx_today.assign(date_removed=pd.Timestamp.today())

    spx_updates = data[1]  # from 2000 to 2021
    spx_updates.columns = ['date', 'ticker_added', 'name_added', 'ticker_removed', 'name_removed', 'reason']
    spx_updates = spx_updates.assign(date=pd.to_datetime(spx_updates.date))
    spx_updates = spx_updates[(spx_updates.date < CRSP_END_DATE) & (spx_updates.date >= CRSP_START_DATE)]

    spx_additions = spx_updates[['date', 'ticker_added', 'name_added']]
    spx_additions = spx_additions.rename({'ticker_added': 'ticker',
                                          'name_added': 'name'}, axis=1)
    spx_additions = spx_additions.dropna(subset=['ticker', 'name'], how='all')

    spx_removals = spx_updates[['date', 'ticker_removed', 'name_removed']]
    spx_removals = spx_removals.rename({'ticker_removed': 'ticker',
                                        'name_removed': 'name'}, axis=1)
    spx_removals = spx_removals.dropna(subset=['ticker', 'name'], how='all')
    # spx_today.to_csv('data/spx/spx_today.csv')
    spx_additions.to_csv('data/spx/spx_additions.csv', index=False)
    spx_removals.to_csv('data/spx/spx_removals.csv', index=False)


def get_spx_tables():
    spx_at_end_date = get_spx_at_end_date()
    spx_at_end_date = spx_at_end_date.assign(sh_class=spx_at_end_date.name.apply(lambda x: get_share_class(x)))
    spx_additions = pd.read_csv('data/spx/spx_additions.csv', parse_dates=['date'])
    spx_additions = spx_additions.assign(sh_class=spx_additions.name.apply(lambda x: get_share_class(x)))
    spx_removals = pd.read_csv('data/spx/spx_removals.csv', parse_dates=['date'])
    spx_removals = spx_removals.assign(sh_class=spx_removals.name.apply(lambda x: get_share_class(x)))
    return spx_at_end_date, spx_additions, spx_removals


# if __name__ == '__main__':
#     save_spx_history_from_wiki()