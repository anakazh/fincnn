# Convolutional Neural Networks for stock return prediction

Replication of image classification model from ["**(Re-)Imag(in)ing Price Trends**" by Jiang, Kelly, Xiu (2020)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587)

## Usage:
1. Get raw data:

   a) Download raw data from CRSP database as csv file and save it to `data/crsp/full_data.csv`
   
      Run `fincnn.utils.crsp_utils` to save data for stocks from S&P500 into `data/raw/`
      
   b) Manually download and save csv-files with Date, Open, High, Low, Close, Volume data into `data/raw/`
      
      (each csv-file should correspond to 1 stock)
2. Run `python -m fincnn.generate_datasets.py` to generate stock chart images (they will be saved into `data/processed/`)
3. Run `python -m fincnn.jiang_kelly_xiu_models.py` to train and evaluate the models

   Alternatively train and evaluate model in two steps:
   
   `python -m fincnn.jiang_kelly_xiu_models.py --train`
   
   `python -m fincnn.jiang_kelly_xiu_models.py --evaluate`


Directory structure:
```
fincnn/
├─ data/
│  ├─ crsp/
│  │  ├─ full_data.csv
│  ├─ processed/
│  │  ├─ 5_day/
│  │  │  ├─ test/
│  │  │  ├─ train/
│  ├─ raw/
│  │  ├─ AAPL_14593_20000101_20201231.csv
├─ models/
│  ├─ jkx_CNN_5_20/
│  │  ├─ images/
│  │  │  ├─ test/
│  │  │  │  ├─ neg/
│  │  │  │  ├─ pos/
│  │  │  ├─ train/
│  │  │  │  ├─ neg/
│  │  │  │  ├─ pos/
│  │  ├─ CNN_5_20.h5
│  │  ├─ history.json
│  │  ├─ metrics.json
```
