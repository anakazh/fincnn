from pathlib import Path


CRSP_CSV_PATH = Path('data/crsp/full_data.csv')
CRSP_DB_PATH = Path('data/crsp/full_data.db')

CRSP_START_DATE = '2000-01-01'
CRSP_END_DATE = '2020-12-31'

SPX_HISTORY_FOR_CRSP_PATH = Path(__file__).resolve().parent.parent.joinpath('spx/spx_history.csv')