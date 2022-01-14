import mplfinance as mpl
from fincnn.config.paths_config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from fincnn.config.generate_datasets_config import TEST_SAMPLE, TRAIN_SAMPLE
from fincnn.utils.image_utils import style, width_config, convert_rgba_to_bw, IMG_SPECS
from fincnn.utils.data_utils import handle_missing_values
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pandas as pd
from functools import partial
import multiprocessing as mp


CPU_COUNT = mp.cpu_count()  # used for image generation in generate_datasets_mp
CHUNK_SIZE = 1  # larger chunksize does not improve performance


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


def generate_images_for_one_stock(filepath, img_horizon, img_spec, sample, target_path):
    ticker_permno_str = filepath.name[:-22] # at the end of the filename: '_date1_date2.csv'
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
        # set initial Close price to 100, adjust OHL by the same factor
        price_scaling_factor = dataslice.iloc[0].Close / 100
        dataslice = dataslice.assign(Close=dataslice.Close.div(price_scaling_factor),
                                     Open=dataslice.Open.div(price_scaling_factor),
                                     High=dataslice.High.div(price_scaling_factor),
                                     Low=dataslice.Low.div(price_scaling_factor))

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


def generate_datasets(img_specs, samples, verbose=False):
    if verbose:
        print('No MP')

    for img_horizon, img_spec in img_specs.items():

        for sample in samples:
            target_path = PROCESSED_DATA_PATH.joinpath(f'{img_horizon}_day/{sample.name}')
            target_path.mkdir(parents=True, exist_ok=False)
            filepaths = [f for f in RAW_DATA_PATH.rglob(f'*.csv')]
            for filepath in tqdm(filepaths,
                                 desc=f'{img_horizon}-day image generation for {sample.name}-sample '):
                generate_images_for_one_stock(filepath, img_horizon=img_horizon, img_spec=img_spec,
                                              sample=sample, target_path=target_path)


def generate_datasets_mp(img_specs, samples, verbose=False):
    if verbose:
        print(f'MP, cpu_count={CPU_COUNT}')

    for img_horizon, img_spec in img_specs.items():

        for sample in samples:

            target_path = PROCESSED_DATA_PATH.joinpath(f'{img_horizon}_day/{sample.name}')
            target_path.mkdir(parents=True, exist_ok=False)

            generate_images_partial = partial(generate_images_for_one_stock,
                                              img_horizon=img_horizon,
                                              img_spec=img_spec,
                                              sample=sample,
                                              target_path=target_path)
            # generate_images_partial has only one argument: filepath (others are fixed)
            filepaths = [f for f in RAW_DATA_PATH.rglob(f'*.csv')]

            process_map(generate_images_partial,
                        filepaths,
                        desc=f'{img_horizon}-day image generation for {sample.name}-sample',
                        max_workers=CPU_COUNT,
                        chunksize=CHUNK_SIZE)


def profile_generate_datasets():
    import cProfile
    import pstats
    from fincnn.utils.data_utils import Sample

    profiler = cProfile.Profile()

    profiler.enable()

    sample = Sample(name='train',
                    start_date='20010101',
                    end_date='20010701',
                    return_horizons=[20, 60])
    img_horizon = 5
    img_spec = IMG_SPECS[img_horizon]
    target_path = PROCESSED_DATA_PATH.joinpath('profiling/')
    target_path.mkdir()
    filepaths = [f for f in RAW_DATA_PATH.rglob(f'*.csv')]
    for filepath in filepaths:
        generate_images_for_one_stock(filepath, img_horizon=img_horizon, img_spec=img_spec,
                                      sample=sample, target_path=target_path)

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename=f'generate_datasets.prof')

    
def main():

    import argparse
    parser = argparse.ArgumentParser(description='Generate images for train and test samples.')
    parser.add_argument('-img-horizon', type=int, nargs='?',
                        help='generate images with given horizon (choose from 5, 20, 60), \
                        if omitted generate images for all horizons (5, 20, 60)')
    parser.add_argument('-sample', type=str, nargs='?',
                        help='specify test or train, if omitted generate images for both')
    parser.add_argument('--verbose', action='store_true',
                        help='verbosity flag, if not specified show only progress bar')

    args = parser.parse_args()

    if args.img_horizon is None:
        img_specs = IMG_SPECS
    else:
        img_specs = dict()
        img_specs[args.img_horizon] = IMG_SPECS[args.img_horizon]

    if args.sample == 'test':
        samples = [TEST_SAMPLE]
    elif args.sample == 'train':
        samples = [TRAIN_SAMPLE]
    else:
        samples = [TRAIN_SAMPLE, TEST_SAMPLE]

    # import warnings
    # warnings.filterwarnings("error")

    if args.verbose:
        import matplotlib.pyplot as plt
        print(f'Matplotlib backend: {plt.get_backend()}')

    if "forkserver" in mp.get_all_start_methods():
        # forkserver start method needed for some backends
        # https://matplotlib.org/stable/gallery/misc/multiprocess_sgskip.html
        mp.set_start_method("forkserver")
        if args.verbose:
            print('Setting multiprocessing start_method to forkserver')

    generate_datasets_mp(img_specs, samples, args.verbose)
    #generate_datasets(img_specs, samples, args.verbose)


if __name__ == '__main__':

    main()


    

