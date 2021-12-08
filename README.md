# Convolutional Neural Networks for stock return prediction

Replication of image classification model from "**(Re-)Imag(in)ing Price Trends**" by Jiang, Kelly, Xiu (2020):
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587

## Usage:
1. Save csv-files with Date, Open, High, Low, Close, Volume data into `data/raw/` (each csv-file should correspond to 1 stock)
2. Run generate_datasets.py to generate stock chart images (they will be saved into `data/processed/`)
3. Run `jiang_kelly_xiu_models.py` to see model design for each image-horizon
3. Run `jiang_kelly_xiu_train_evaluate.py` to train and evaluate the models


Directory structure:
```
fincnn/
├─ data/
│  ├─ processed/
│  │  ├─ 5_day/
│  │  │  ├─ test/
│  │  │  │  ├─ AAPL_14593_20000107_ret20_0.1464_ret60_0.2795.png
│  │  │  ├─ train/
│  │  │  │  ├─ AAPL_14593_19930108_ret20_-0.0803_ret60_-0.1968.png
│  ├─ raw/
│  │  ├─ AAPL_14593.csv
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
├─ samples/
│  ├─ test_permno_list.txt
│  ├─ tain_permno_list.txt
├─ src/
│  ├─ fincnn/
│  │  ├─ utils/
│  │  │  ├─ crsp_utils.py
│  │  │  ├─ data_utils.py
│  │  │  ├─ image_utils.py
│  │  ├─ base_model.py
│  │  ├─ generate_datasets.py
│  │  ├─ generate_datasets_sample_params.py
│  │  ├─ jiang_kelly_xiu_models.py
│  │  ├─ jiang_kelly_xiu_train_evaluate.py
├─ .gitignore
├─ README.md
├─ setup.py
```
