from PIL import Image, ImageOps
from dataclasses import dataclass


@dataclass(frozen=True)
class ImgSpec:
    img_height: int
    img_width: int
    volume_height: int


IMG_SPECS = {
    5: ImgSpec(img_height=32, img_width=15, volume_height=12),
    20: ImgSpec(img_height=64, img_width=60, volume_height=12),
    60: ImgSpec(img_height=96, img_width=180, volume_height=19),
             }

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


# width-related code in mplfinance:
# https://github.com/matplotlib/mplfinance/blob/d777f433e92588a09803cefb56053a49e3618d39/src/mplfinance/_widths.py#L86
# width is given in Points, 1 Pixel = 72 Points, source:
# https://stackoverflow.com/questions/57657419/how-to-draw-a-figure-in-specific-pixel-size-with-matplotlib
width_config = {'volume_width': 1/3,  # width is given as a fraction [0, 1] - NOT POINTS
                'volume_linewidth': 0,  # Width of the bar edge(s). If 0, don't draw edges (grey area around volume bars)
                'line_width': 0,  # apparently this is the width of mav, vlines and hlines
                'ohlc_linewidth': 1 * 72 / style['rc']['figure.dpi'] / 2,
                'ohlc_ticksize': 1/3,  # width is given as a fraction [0, 1] - NOT POINTS
                }


def convert_gray_to_black(image):
    # Source: https://www.codementor.io/@isaib.cicourel/image-manipulation-in-python-du1089j1u
    width, height = image.size

    # Create new Image and a Pixel Map
    new = Image.new("1", image.size, "white")
    pixels = new.load()
    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i, j)) # Get Pixel from original image
            if pixel < 255:  # grey pixels
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