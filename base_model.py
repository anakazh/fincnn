import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import json
import mplfinance as mpl
import os
import shutil
from tqdm import tqdm
import pandas as pd

style = {'base_mpl_style': 'ggplot',
         'marketcolors'  : {'candle': {'up': '#000000', 'down': '#000000'},
                            'edge'  : {'up': '#000000', 'down': '#000000'},
                            'wick'  : {'up': '#000000', 'down': '#000000'},
                            'ohlc'  : {'up': '#000000', 'down': '#000000'},
                            'volume': {'up': '#000000', 'down': '#000000'},
                            'vcedge': {'up': '#000000', 'down': '#000000'},
                            'vcdopcod'      : False, #Volume Color Depends on Price Change on Day
                            'alpha'         : 0,# 0.9,
                            },
         'mavcolors'     : None,
         'facecolor'     : 'w',
         'gridcolor'     : 'w',
         'gridstyle'     : '-',
         'y_on_right'    : True,
         'rc'            : {'axes.grid'     : False,
                            'axes.edgecolor': '#000000',
                            'axes.labelcolor': 'k',
                            'ytick.color'   : 'k',
                            'xtick.color'   : 'k',
                            'lines.markeredgecolor': 'k',
                            'patch.force_edgecolor': False,
                            'figure.titlesize'     : 'x-large',
                            'figure.titleweight'   : 'semibold',
                            },
         'base_mpf_style': 'checkers'}


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
                       ).rename({'Close/Last':'Close'}, axis=1)
    return data.sort_index()


class BaseCNN:

    def __init__(self, name='model1', image_horizon=60, return_horizon=20,
                 img_height=96, img_width=180):
        self.name = name
        self.image_horizon = image_horizon
        self.return_horizon = return_horizon
        self.img_height = img_height
        self.img_width = img_width
        self.train_dataset_path = 'data/processed/' + self.name + '/train'
        if not os.path.exists(self.train_dataset_path):
            os.makedirs(self.train_dataset_path)
        self.val_dataset_path = 'data/processed/' + self.name + '/val'
        if not os.path.exists(self.val_dataset_path):
            os.makedirs(self.val_dataset_path)
        if not os.path.exists('models/' + self.name):
            os.makedirs('models/' + self.name)
        self.model_path = 'models/' + self.name + '/' + self.name + '.h5'
        self.history_path = 'models/' + self.name + '/history.json'
        try:
            self.model = models.load_model(self.model_path)
            self.history = json.load(open(self.history_path))
        except:
            self.model = None
            self.history = None

    def gen_image(self, data, savepath):
        """
        Generate model-specific technical graph
        """
        # TODO: adjust volume bar width, they are unequal for different days due to:
        # https: // github.com / matplotlib / mplfinance / blob / d777f433e92588a09803cefb56053a49e3618d39 / src / mplfinance / _widths.py  # L86

        fig, _ = mpl.plot(data,
                          volume=True,
                          style=style,
                          figsize=(self.img_width/10, self.img_height/10),
                          #width_adjuster_version='v1', #same issue
                          axisoff=True,
                          tight_layout=True,
                          returnfig=True,
                          closefig=True,
                          scale_padding=0)
        fig.savefig(savepath, dpi=100)

    def gen_train_val_images(self):
        data = get_raw_data()
        train_val_cutoff = round((len(data) - self.image_horizon - self.return_horizon) * 0.7)
        last_train_index = self.image_horizon + self.return_horizon + train_val_cutoff

        sample_start = data.iloc[0].name.strftime('%Y-%m-%d')
        last_train_date_str = data.iloc[last_train_index].name.strftime('%Y-%m-%d')
        first_val_index = self.image_horizon + self.return_horizon + train_val_cutoff + 1
        first_val_date = data.iloc[first_val_index].name.strftime('%Y-%m-%d')
        sample_end = data.iloc[-1].name.strftime('%Y-%m-%d')
        print('Train sample: ' + sample_start + ' - ' + last_train_date_str)
        print('Validation sample: ' + first_val_date + ' - ' + sample_end)

        overwrite_dir(os.path.join(self.train_dataset_path, 'pos'))
        overwrite_dir(os.path.join(self.train_dataset_path, 'neg'))
        overwrite_dir(os.path.join(self.val_dataset_path, 'pos'))
        overwrite_dir(os.path.join(self.val_dataset_path, 'neg'))

        for row_n in tqdm(range(self.image_horizon, len(data) - self.return_horizon),
                          desc='Image generation in progress '):
            dataslice = data.iloc[row_n - self.image_horizon:row_n]
            price_scaling_factor = dataslice.iloc[0].Close / 100  # set initial price to 100
            dataslice = dataslice.assign(Close=dataslice.Close.div(price_scaling_factor))

            ret = data.iloc[row_n + self.return_horizon].Close / data.iloc[row_n].Close - 1
            filename = dataslice.iloc[-1].name.strftime('%Y%m%d') + '.png'
            if ret > 0 and row_n <= last_train_index:
                savepath = os.path.join(self.train_dataset_path, 'pos', filename)
            elif ret < 0 and row_n <= last_train_index:
                savepath = os.path.join(self.train_dataset_path, 'neg', filename)
            elif ret > 0 and row_n > last_train_index:
                savepath = os.path.join(self.val_dataset_path, 'pos' , filename)
            else:
                savepath = os.path.join(self.val_dataset_path, 'neg', filename)

            self.gen_image(dataslice, savepath)

        train_pos_ret = len(os.listdir(os.path.join(self.train_dataset_path, 'pos')))
        train_sample_size = train_pos_ret + len(os.listdir(os.path.join(self.train_dataset_path, 'neg')))
        val_pos_ret = len(os.listdir(os.path.join(self.val_dataset_path, 'pos')))
        val_sample_size = val_pos_ret + len(os.listdir(os.path.join(self.val_dataset_path, 'neg')))
        print('Positive returns in the train sample: {} ({}%)'.format(train_pos_ret,
                                                                      round(train_pos_ret / train_sample_size,
                                                                            4) * 100))
        print('Positive returns in the validation sample: {} ({}%)'.format(val_pos_ret,
                                                                           round(val_pos_ret / val_sample_size,
                                                                                 4) * 100))

    def fit(self):
        # For a grayscale image pixel values range from 0 to 255 (0 represents black and 255 represents white)
        batch_size = 1
        full_dataset = image_dataset_from_directory(self.train_dataset_path,
                                                    batch_size=batch_size,  # default: 32
                                                    color_mode='grayscale',
                                                    image_size=(self.img_height, self.img_width))

        # train-test split
        dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
        train_size = int(0.7 * dataset_size)
        train_dataset = full_dataset.take(train_size)
        test_dataset = full_dataset.skip(train_size)

        # Model design
        self.model = models.Sequential()
        # input_shape = (height, width, channels)
        # channels = 1 for grayscale, channels = 3 for RGB
        self.model.add(layers.InputLayer(input_shape=(self.img_height, self.img_width, 1),
                                         batch_size=batch_size))
        # Normalize pixel values to be between 0 and 1
        self.model.add(layers.Rescaling(1./255))
        self.model.add(layers.Conv2D(64, (5, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 1), padding='same'))
        self.model.add(layers.Conv2D(128, (5, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D(2, 1, padding='same'))
        self.model.add(layers.Conv2D(256, (5, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 1), padding='same'))
        self.model.add(layers.Conv2D(512, (5, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 1), padding='same'))
        self.model.add(layers.Flatten())   # the output of the last CNN block is flattened to a 1D vector
        self.model.add(layers.Dense(512))  # Dense layer = fully connected (FC) layer
        self.model.add(layers.Dense(1, activation='softmax'))  # yields the probabilities of up and down
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  # default False
                           metrics=['accuracy'])

        self.history = self.model.fit(x=train_dataset, y=None, batch_size=batch_size,
                                      epochs=2,
                                      verbose=2,
                                      validation_data=test_dataset,
                                      max_queue_size=5,  # default is 10
                                      workers=3,
                                      use_multiprocessing=True)  # default is False
        json.dump(self.history.history, open(self.history_path, 'w'))
        self.model.save(self.model_path)

    def predict(self):
        if self.model is None:
            raise Exception('No ' + self.name + ' found, train the model first!')
        raise NotImplementedError

    def evaluate(self):
        if self.model is None:
            raise Exception('No ' + self.name + ' found, train the model first!')
        raise NotImplementedError


if __name__ == '__main__':
    cnn = BaseCNN()
    cnn.gen_train_val_images()
    cnn.fit()

    #data = get_raw_data()
    #dataslice = data.iloc[0:60]
    #cnn.gen_image(dataslice, 'testsave_new.png')
