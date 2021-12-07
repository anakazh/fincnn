import sqlite3
from pathlib import Path
from tech_analysis_cnn import Sample

RAW_DATA_PATH = Path('../data/raw/')
PROCESSED_DATA_PATH = Path(f'../data/processed/')

RETURN_HORIZONS = [20, 60]

TRAIN_SAMPLE_PERMNO_LIST_PATH = Path('../samples/train_permno_list.txt')
TRAIN_SAMPLE_START_DATE = '19930101'
TRAIN_SAMPLE_END_DATE = '19991231'


TEST_SAMPLE_PERMNO_LIST_PATH = Path('../samples/test_permno_list.txt')
TEST_SAMPLE_START_DATE = '20000101'
TEST_SAMPLE_END_DATE = '20200101'

with open(TRAIN_SAMPLE_PERMNO_LIST_PATH, 'r') as f:
    train_permno_list = f.read().split()

TRAIN_SAMPLE = Sample(name='train',
                      start_date=TRAIN_SAMPLE_START_DATE,
                      end_date=TRAIN_SAMPLE_END_DATE,
                      permno_list=train_permno_list,
                      return_horizons=RETURN_HORIZONS)

with open(TRAIN_SAMPLE_PERMNO_LIST_PATH, 'r') as f:
    test_permno_list = f.read().split()

TEST_SAMPLE = Sample(name='test',
                     start_date=TEST_SAMPLE_START_DATE,
                     end_date=TEST_SAMPLE_END_DATE,
                     permno_list=test_permno_list,
                     return_horizons=RETURN_HORIZONS)


def select_2000_lagest_train_sample():
    db_path = '../data/crsp/full_data.db'
    sql_statement = """
    SELECT DISTINCT permno FROM OHLC
    WHERE Date BETWEEN '19930101' AND '19991231'
    GROUP BY permno
    ORDER BY AVG(Close*cfac_pr*Volume*cfac_vol) DESC
    LIMIT 2000
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql_statement)
        permnos = cursor.fetchall()
    permnos = [p[0]+'\n' for p in permnos]
    with open(TRAIN_SAMPLE_PERMNO_LIST_PATH, 'w') as file:
        file.writelines(permnos)


def select_2000_lagest_test_sample():
    db_path = '../data/crsp/full_data.db'
    sql_statement = """
    SELECT DISTINCT permno FROM OHLC
    WHERE Date BETWEEN '19990101' AND '19991231'
    GROUP BY permno
    ORDER BY AVG(Close*cfac_pr*Volume*cfac_vol) DESC
    LIMIT 2000
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql_statement)
        permnos = cursor.fetchall()
    permnos = [p[0]+'\n' for p in permnos]
    with open(TEST_SAMPLE_PERMNO_LIST_PATH, 'w') as file:
        file.writelines(permnos)


if __name__ == '__main__':

    select_2000_lagest_train_sample()
    select_2000_lagest_test_sample()

    with open(TRAIN_SAMPLE_PERMNO_LIST_PATH, 'r') as f:
        train_permno_list = f.read().split()

    train_sample = Sample(name='train',
                          start_date=TRAIN_SAMPLE_START_DATE,
                          end_date=TRAIN_SAMPLE_END_DATE,
                          permno_list=train_permno_list,
                          return_horizons=RETURN_HORIZONS)
    train_sample.describe(savepath='samples/train_summary.csv')

    with open(TEST_SAMPLE_PERMNO_LIST_PATH, 'r') as f:
        test_permno_list = f.read().split()

    test_sample = Sample(name='test',
                         start_date=TEST_SAMPLE_START_DATE,
                         end_date=TEST_SAMPLE_END_DATE,
                         permno_list=test_permno_list,
                         return_horizons=RETURN_HORIZONS)
    test_sample.describe(savepath='samples/test_summary.csv')

