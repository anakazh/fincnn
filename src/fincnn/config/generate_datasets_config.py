from fincnn.config.crsp_config import CRSP_START_DATE, CRSP_END_DATE
from fincnn.utils.data_utils import Sample


RETURN_HORIZONS = [20, 60]

TRAIN_SAMPLE_START_DATE = CRSP_START_DATE
TRAIN_SAMPLE_END_DATE = '2010-12-31'

TEST_SAMPLE_START_DATE = '2010-01-01'
TEST_SAMPLE_END_DATE = CRSP_END_DATE

TRAIN_SAMPLE = Sample(name='train',
                      start_date=TRAIN_SAMPLE_START_DATE,
                      end_date=TRAIN_SAMPLE_END_DATE,
                      return_horizons=RETURN_HORIZONS)

TEST_SAMPLE = Sample(name='test',
                     start_date=TEST_SAMPLE_START_DATE,
                     end_date=TEST_SAMPLE_END_DATE,
                     return_horizons=RETURN_HORIZONS)


def main():
    TRAIN_SAMPLE.describe(savepath='train_sample_summary.csv')
    TEST_SAMPLE.describe(savepath='test_sample_summary.csv')


if __name__ == '__main__':
    main()



