Replication of image classification model from "**(Re-)Imag(in)ing Price Trends**" by Jiang, Kelly, Xiu (2020):
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587

Usage:
1. Save csv-files with Date, Open, High, Low, Close, Volume data into `data/raw/` (each csv-file should correspond to 1 stock)
2. Run generate_datasets.py to generate stock chart images (they will be saved into `data/processed/`)
3. Run `jiang_kelly_xiu_models.py` to see model design for each image-horizon
3. Run `jiang_kelly_xiu_train_evaluate.py` to train and evaluate the models


Directory structure:
```
tech_analysis_cnn/
├─ data/
│  ├─ processed/
│  │  ├─ 5_day/
│  │  │  ├─ ret_20/
│  │  │  │  ├─ neg/
│  │  │  │  ├─ pos/
│  │  │  ├─ ret_60/
│  │  │  │  ├─ neg/
│  │  │  │  ├─ pos/
│  ├─ raw/
├─ models/
├─ utils/
│  ├─ data_utils.py
│  ├─ image_utils.py
├─ .gitignore
├─ base_model.py
├─ generate_datasets.py
├─ jiang_kelly_xiu_models.py
├─ jiang_kelly_xiu_train_evaluate.py
├─ README.md
├─ requirements.txt
```
