import os
import shutil
import pandas as pd
from PIL import Image, ImageOps

style = {'base_mpl_style': 'ggplot',
         'marketcolors'  : {'candle': {'up': '#000000', 'down': '#000000'},
                            'edge': {'up': '#000000', 'down': '#000000'},
                            'wick': {'up': '#000000', 'down': '#000000'},
                            'ohlc': {'up': '#000000', 'down': '#000000'},
                            'volume': {'up': '#000000', 'down': '#000000'},
                            'vcedge': {'up': '#000000', 'down': '#000000'},
                            'vcdopcod': False,  #Volume Color Depends on Price Change on Day
                            'alpha': 1,  # color transparency
                            },
         'mavcolors'     : None,
         'facecolor'     : 'w',
         'gridcolor'     : 'w',
         'gridstyle'     : '-',
         'y_on_right'    : True,
         'rc'            : {'axes.grid'     : False,
                            'axes.edgecolor': '#000000',
                            'axes.labelcolor': 'k',
                            'ytick.color': 'k',
                            'xtick.color': 'k',
                            'lines.markeredgecolor': '#000000',
                            'patch.force_edgecolor': False,
                            'figure.titlesize': 'x-large',
                            'figure.titleweight': 'semibold',
                            'figure.dpi': 100,
                            },
         'base_mpf_style': 'checkers'}


def convert_gray_to_black(image):
    # Source: https://www.codementor.io/@isaib.cicourel/image-manipulation-in-python-du1089j1u
    width, height = image.size

    # Create new Image and a Pixel Map
    new = Image.new("1", image.size, "white")
    pixels = new.load()
    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i, j)) # Get Pixel from original image
            if pixel < 255: # grey pixels
                pixels[i, j] = 0  # reset to black
            else:  # white pixels
                pixels[i, j] = 1  # leave them white
    return new


def convert_rgba_to_bw(savepath):
    image = Image.open(savepath).convert('L')
    image = convert_gray_to_black(image)
    image = image.convert('L')  # convert it into "L" for the next line to run
    image = ImageOps.invert(image)
    image = image.convert('1')  # convert back to 1-bit pixels, black and white
    image.save(savepath)
    image.close()


def overwrite_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def get_raw_data():
    # source: https://www.nasdaq.com/market-activity/stocks/aapl/historical
    data = pd.read_csv('data/raw/AAPL.csv',
                       index_col='Date',
                       converters={'Close/Last': lambda s: float(s.replace('$', '')),
                                   'Open': lambda s: float(s.replace('$', '')),
                                   'High': lambda s: float(s.replace('$', '')),
                                   'Low': lambda s: float(s.replace('$', ''))},
                       parse_dates=True,
                       ).rename({'Close/Last': 'Close'}, axis=1)
    return data.sort_index()