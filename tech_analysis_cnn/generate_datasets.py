import mplfinance as mpl
from tech_analysis_cnn.utils.image_utils import style, width_config, convert_rgba_to_bw, img_specs
from tech_analysis_cnn.utils.data_utils import handle_missing_values
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pandas as pd
from functools import partial
import multiprocessing as mp
from generate_datasets_config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TEST_SAMPLE, TRAIN_SAMPLE


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
    filepath = [f for f in RAW_DATA_PATH.rglob(f'*{permno}.csv')][0]
    ticker_permno_str = filepath.name[:-4]
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True).sort_index()
    df = handle_missing_values(df)
    ret_columns = []
    for return_horizon in sample.return_horizons:
        df[f'ret{return_horizon}'] = df.Close.shift(-return_horizon).div(df.Close).sub(1)
        ret_columns.append(f'ret{return_horizon}')
    start_index = pd.to_datetime(sample.start_date)
    end_index = pd.to_datetime(sample.end_date)
    data = df.loc[start_index:end_index]
    for row_n in range(img_horizon, len(data) - max(sample.return_horizons)):
        dataslice = data.iloc[row_n - img_horizon:row_n]
        price_scaling_factor = dataslice.iloc[0].Close / 100  # set initial price to 100
        dataslice = dataslice.assign(Close=dataslice.Close.div(price_scaling_factor))

        date_str = dataslice.iloc[-1].name.strftime('%Y%m%d')

        filename = f'{ticker_permno_str}_{date_str}'
        for ret_column in ret_columns:
            filename = filename + f'_{ret_column}_{round(dataslice[ret_column].iloc[-1], 4)}'
        filename = filename + '.png'
        savepath = target_path.joinpath(filename)

        # generate, convert and saves the image - expensive operations, do only once
        fig = gen_image(dataslice.drop(ret_columns, axis=1), img_spec)
        fig.savefig(savepath, dpi=style['rc']['figure.dpi'])
        convert_rgba_to_bw(savepath)


def generate_datasets():
    for img_horizon, img_spec in img_specs.items():

        if img_horizon == 20 or img_horizon == 60:
            continue  # try generating only 5-day images for now

        for sample in [TRAIN_SAMPLE, TEST_SAMPLE]:
            target_path = PROCESSED_DATA_PATH.join('{img_horizon}_day/{sample.name}')
            target_path.mkdir(parents=True, exist_ok=False)
            for permno in tqdm(sample.permno_list,
                               desc=f'{img_horizon}-day image generation for {sample.name}-sample in progress '):
                generate_images_for_permno(permno, img_horizon=img_horizon, img_spec=img_spec,
                                           sample=sample, target_path=target_path)


def generate_datasets_mp():
    # import matplotlib.pyplot as plt
    # print(plt.get_backend())

    # apparently need to change start method for some matplotlib backends
    # https://matplotlib.org/stable/gallery/misc/multiprocess_sgskip.html
    mp.set_start_method("forkserver")

    for img_horizon, img_spec in img_specs.items():

        if img_horizon == 20 or img_horizon == 60:
            continue  # try generating only 5-day images for now

        for sample in [TRAIN_SAMPLE, TEST_SAMPLE]:
            target_path = PROCESSED_DATA_PATH.join('{img_horizon}_day/{sample.name}')
            target_path.mkdir(parents=True, exist_ok=False)

            generate_images_partial = partial(generate_images_for_permno,
                                              img_horizon=img_horizon,
                                              img_spec=img_spec,
                                              sample=sample,
                                              target_path=target_path)
            # generate_images_partial has only one argument: permno (others are fixed)
            process_map(generate_images_partial,
                        sample.permno_list,
                        desc=f'{img_horizon}-day image generation for {sample.name}-sample in progress ',
                        max_workers=CPU_COUNT)

def profile_generate_datasets():
    import cProfile
    import pstats
    from tech_analysis_cnn import Sample

    profiler = cProfile.Profile()

    profiler.enable()

    sample = Sample(name='train',
                    start_date='19930101',
                    end_date='19940101',
                    permno_list=['14593', '10107'],
                    return_horizons=[20, 60])
    img_horizon = 5
    img_spec = img_specs[img_horizon]
    target_path = PROCESSED_DATA_PATH.join('profiling/')
    target_path.mkdir()
    for permno in sample.permno_list:
        for ret in [f'ret_{x}' for x in sample.return_horizons]:
            for sign in ['pos', 'neg']:
                target_path.joinpath(ret, sample.name, sign).mkdir(parents=True, exist_ok=True)
        generate_images_for_permno(permno, img_horizon=img_horizon, img_spec=img_spec,
                                       sample=sample, target_path = target_path)

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='../generate_datasets.prof')


if __name__ == '__main__':

    generate_datasets_mp()

