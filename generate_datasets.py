import mplfinance as mpl
from utils.image_utils import style, width_config, convert_rgba_to_bw, img_specs
from utils.data_utils import handle_missing_values, Sample
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import pandas as pd
import shutil
from functools import partial
import multiprocessing as mp


CPU_COUNT = mp.cpu_count()  # used for image generation here

def gen_image(data, img_spec):
    """
    Generate OHLC chart with volume bars given specified width, height, volume height
    :param data: DataFrame with Open, Low, High, Close, Volume colums and Date index
    :param img_spec: Dictionary with keys: img_width, img_height, volume_height
    :return: matplotlib.figure object
    """

    assert len(data) == img_spec.img_width / 3
    dpi = style['rc']['figure.dpi']
    fig, _ = mpl.plot(data,
                      volume=True,
                      style=style,
                      figsize=(img_spec.img_width / dpi, img_spec.img_height / dpi),
                      panel_ratios=((img_spec.img_height - img_spec.volume_height - 1) / img_spec.img_height,
                                     img_spec.volume_height / img_spec.img_height),
                      update_width_config=width_config,
                      xlim=(- 1 /3, img_spec.img_width / 3 - 1 /3),
                      axisoff=True,
                      tight_layout=True,
                      returnfig=True,
                      closefig=True,
                      scale_padding=0)
    return fig


def generate_images_for_permno(permno, img_horizon, img_spec, sample, target_path):
    filepath = [f for f in Path('data/raw/').rglob(f'*{permno}.csv')][0]
    ticker_permno_str = filepath.name[:-4]
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True).sort_index()
    df = handle_missing_values(df)
    ret_columns = []
    for return_horizon in sample.return_horizons:
        df[f'ret_{return_horizon}'] = df.Close.shift(-return_horizon).div(df.Close).sub(1)
        ret_columns.append(f'ret_{return_horizon}')
    start_index = pd.to_datetime(sample.start_date)
    end_index = pd.to_datetime(sample.end_date)
    data = df.loc[start_index:end_index]
    for row_n in range(img_horizon, len(data) - max(sample.return_horizons)):
        dataslice = data.iloc[row_n - img_horizon:row_n]
        price_scaling_factor = dataslice.iloc[0].Close / 100  # set initial price to 100
        dataslice = dataslice.assign(Close=dataslice.Close.div(price_scaling_factor))
        fig = gen_image(dataslice.drop(ret_columns, axis=1), img_spec)

        date_str = dataslice.iloc[-1].name.strftime('%Y%m%d')
        filename = f'{ticker_permno_str}_{date_str}.png'
        ret_column = ret_columns[0]
        if dataslice[ret_column].iloc[-1] > 0:
            savepath = target_path.joinpath(ret_column, sample.name, 'pos', filename)
        elif dataslice[ret_column].iloc[-1] < 0:
            savepath = target_path.joinpath(ret_column, sample.name, 'neg', filename)
        # ignore ret == 0

        fig.savefig(savepath, dpi=style['rc']['figure.dpi'])
        convert_rgba_to_bw(savepath)  # reads, converts and saves the image - do only once

        for ret_column in ret_columns[1:]:
            if dataslice[ret_column].iloc[-1] > 0:
                new_savepath = target_path.joinpath(ret_column, sample.name, 'pos', filename)
            elif dataslice[ret_column].iloc[-1] < 0:
                new_savepath = target_path.joinpath(ret_column, sample.name, 'neg', filename)
            shutil.copy(savepath, new_savepath)


def main():

    train_sample = Sample(name='train',
                          start_date='19930101',
                          end_date='19991231',
                          permno_list=['14593', '10107'],
                          return_horizons=[20, 60])

    test_sample = Sample(name='test',
                         start_date='20000101',
                         end_date='20200101',
                         permno_list=['14593', '10107'],
                         return_horizons=[20, 60])

    for img_horizon, img_spec in img_specs.items():
        target_path = Path(f'data/processed/{img_horizon}_day')
        target_path.mkdir()
        for sample in [train_sample, test_sample]:

            for ret in [f'ret_{x}' for x in sample.return_horizons]:
                for sign in ['pos', 'neg']:
                    target_path.joinpath(ret, sample.name, sign).mkdir(parents=True, exist_ok=True)

            generate_images_partial = partial(generate_images_for_permno,
                                              img_horizon=img_horizon,
                                              img_spec=img_spec,
                                              sample=sample,
                                              target_path=target_path)
            # generate_images_partial has only one argument: permno (others are fixed)
            process_map(generate_images_partial,
                        sample.permno_list,
                        desc=f'{img_horizon}-day image generation for {sample.name}-sample in progress ',
                        max_workers=2)


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # print(plt.get_backend())

    # apparently need to change start method for some matplotlib backends
    # https://matplotlib.org/stable/gallery/misc/multiprocess_sgskip.html
    mp.set_start_method("forkserver")
    main()