import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import json
import mplfinance as mpl
import os
from tqdm import tqdm
import pandas as pd
from utils import style, width_config, overwrite_dir, get_raw_data, convert_rgba_to_bw
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class BaseCNN:

    def __init__(self, name, image_horizon, return_horizon,
                 img_height, img_width, volume_height, batch_size=1):
        self.name = name
        self.image_horizon = image_horizon
        self.return_horizon = return_horizon
        self.img_height = img_height
        self.img_width = img_width
        self.volume_height = volume_height
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
        self.val_metrics_path = 'models/' + self.name + '/val_metrics.json'
        try:
            self.model = models.load_model(self.model_path)
            self.history = json.load(open(self.history_path))
        except:
            self.model = None
            self.history = None
        self.batch_size = batch_size  # should be the save for self.fit(), self.predict() and self.evaluate()

    def gen_image(self, data, savepath):
        """
        Generate model-specific technical graph
        """
        assert len(data) == self.img_width / 3
        dpi = style['rc']['figure.dpi']
        fig, _ = mpl.plot(data,
                          volume=True,
                          style=style,
                          figsize=(self.img_width / dpi, self.img_height / dpi),
                          panel_ratios=((self.img_height - self.volume_height - 1) / self.img_height,
                                        self.volume_height / self.img_height),
                          update_width_config=width_config,
                          xlim=(-1/3, self.img_width/3-1/3),
                          axisoff=True,
                          tight_layout=True,
                          returnfig=True,
                          closefig=True,
                          scale_padding=0)
        fig.savefig(savepath, dpi=dpi)
        convert_rgba_to_bw(savepath)

    def generate_images(self):
        #TODO: validation cutoff date
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
                savepath = os.path.join(self.val_dataset_path, 'pos', filename)
            else:
                savepath = os.path.join(self.val_dataset_path, 'neg', filename)

            self.gen_image(dataslice, savepath)
        self.describe_dataset()

    def describe_dataset(self):
        train_pos_count = len(os.listdir(os.path.join(self.train_dataset_path, 'pos')))
        train_sample_size = train_pos_count + len(os.listdir(os.path.join(self.train_dataset_path, 'neg')))
        train_pos_frac = train_pos_count / train_sample_size
        val_pos_count = len(os.listdir(os.path.join(self.val_dataset_path, 'pos')))
        val_sample_size = val_pos_count + len(os.listdir(os.path.join(self.val_dataset_path, 'neg')))
        val_pos_frac = val_pos_count / val_sample_size
        print('Train sample size: {}'.format(train_sample_size))
        print('Positive returns in the train sample: {} ({:.2%})'.format(train_pos_count, train_pos_frac))
        print('Validation sample size: {}'.format(val_sample_size))
        print('Positive returns in the validation sample: {} ({:.2%})'.format(val_pos_count, val_pos_frac))

    def load_dataset(self):
        # For a grayscale image pixel values range from 0 to 255 (0 represents black and 255 represents white)
        full_dataset = image_dataset_from_directory(self.train_dataset_path,
                                                    batch_size=self.batch_size,  # default: 32
                                                    color_mode='grayscale',
                                                    image_size=(self.img_height, self.img_width))
        self.describe_dataset()

        # train-test split
        dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
        train_size = int(0.7 * dataset_size)
        train_dataset = full_dataset.take(train_size)
        test_dataset = full_dataset.skip(train_size)
        return train_dataset, test_dataset

    def compile(self):
        raise Exception('Model design missing')

    def fit(self, epochs=2):
        train_dataset, test_dataset = self.load_dataset()
        self.compile()
        self.history = self.model.fit(x=train_dataset, y=None, batch_size=self.batch_size,
                                      epochs=epochs,
                                      verbose=2,
                                      validation_data=test_dataset,
                                      max_queue_size=5,  # default is 10
                                      workers=3,
                                      use_multiprocessing=True)  # default is False
        self.history = self.history.history
        json.dump(self.history, open(self.history_path, 'w'))
        self.model.save(self.model_path)

    def plot_history(self):
        pd.DataFrame(self.history).plot()
        plt.savefig(self.history_path[:-4]+'png')

    def predict(self):
        if self.model is None:
            raise Exception('No ' + self.name + ' found, train the model first!')
        raise NotImplementedError

    def evaluate(self):
        if self.model is None:
            raise Exception('No ' + self.name + ' found, train the model first!')
        val_dataset = image_dataset_from_directory(self.val_dataset_path,
                                                   batch_size=self.batch_size,  # default: 32
                                                   color_mode='grayscale',
                                                   image_size=(self.img_height, self.img_width))
        self.describe_dataset()

        # self.model.evaluate throws an error when batch_size is > 1
        #val_metrics = self.model.evaluate(x=val_dataset, y=None, batch_size=self.batch_size, return_dict=True)
        #val_metrics['f1score'] = 2 / (1/val_metrics['precision']+1/val_metrics['recall'])
        #json.dump(val_metrics, open(self.val_metrics_path, 'w'))

        input = val_dataset.map(lambda x, y: x)
        target = val_dataset.map(lambda x, y: y)
        target_true = np.array(0)
        for batch in target.as_numpy_iterator():
            target_true = np.append(target_true, batch)
        target_true = np.delete(target_true, 0)
        target_pred = self.model.predict(input).reshape(target_true.shape)
        accuracy = (target_pred == target_true).sum() / len(target_true)
        precision, recall, f1score, _ = precision_recall_fscore_support(target_true, target_pred,
                                                                              average='binary')
        val_metrics = {'accuracy': accuracy, 'precision': precision,
                       'recall': recall, 'f1score': f1score}
        json.dump(val_metrics, open(self.val_metrics_path, 'w'))


class CNN_5_20(BaseCNN):

    def __init__(self, batch_size=1):
        super().__init__(name='CNN_5_20', image_horizon=5, return_horizon=20,
                         img_height=32, img_width=15, volume_height=12, batch_size=batch_size)

    def compile(self):
        # Model design
        self.model = models.Sequential(name=self.name)
        # input_shape = (height, width, channels)
        # channels = 1 for grayscale, channels = 3 for RGB
        self.model.add(layers.InputLayer(input_shape=(self.img_height, self.img_width, 1),
                                         batch_size=self.batch_size))
        # Normalize pixel values to be between 0 and 1
        # TODO: examples from the paper (page 14) - no rescaling
        self.model.add(layers.Rescaling(1. / 255))
        self.model.add(layers.Conv2D(64, (5, 3), activation=layers.LeakyReLU(alpha=0.01)))
        self.model.add(layers.MaxPooling2D((2, 1), padding='same'))
        self.model.add(layers.Conv2D(128, (5, 3), activation=layers.LeakyReLU(alpha=0.01)))
        self.model.add(layers.MaxPooling2D(2, 1, padding='same'))
        self.model.add(layers.Flatten())  # the output of the last CNN block is flattened to a 1D vector
        self.model.add(layers.Dense(1))  # Dense layer = fully connected (FC) layer
        self.model.add(layers.Softmax())  # boolean mask
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  # default False
                           metrics=['accuracy', 'Precision', 'Recall'])


class CNN_20_20(BaseCNN):

    def __init__(self, batch_size=1):
        super().__init__(name='CNN_20_20', image_horizon=20, return_horizon=20,
                         img_height=64, img_width=60, volume_height=12, batch_size=batch_size)

    def compile(self):
        # Model design
        self.model = models.Sequential(name=self.name)
        # input_shape = (height, width, channels)
        # channels = 1 for grayscale, channels = 3 for RGB
        self.model.add(layers.InputLayer(input_shape=(self.img_height, self.img_width, 1),
                                         batch_size=self.batch_size))
        # Normalize pixel values to be between 0 and 1
        self.model.add(layers.Rescaling(1. / 255))
        self.model.add(layers.Conv2D(64, (5, 3), activation=layers.LeakyReLU(alpha=0.01)))
        self.model.add(layers.MaxPooling2D((2, 1), padding='same'))
        self.model.add(layers.Conv2D(128, (5, 3), activation=layers.LeakyReLU(alpha=0.01)))
        self.model.add(layers.MaxPooling2D(2, 1, padding='same'))
        self.model.add(layers.Conv2D(256, (5, 3), activation=layers.LeakyReLU(alpha=0.01)))
        self.model.add(layers.MaxPooling2D((2, 1), padding='same'))
        self.model.add(layers.Flatten())  # the output of the last CNN block is flattened to a 1D vector
        self.model.add(layers.Dense(1))  # Dense layer = fully connected (FC) layer
        self.model.add(layers.Softmax())  # boolean mask
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  # default False
                           metrics=['accuracy', 'Precision', 'Recall'])


class CNN_60_20(BaseCNN):
    def __init__(self, batch_size=1):
        super().__init__(name='CNN_60_20', image_horizon=60, return_horizon=20,
                         img_height=96, img_width=180, volume_height=19, batch_size=batch_size)

    def compile(self):
        # Model design
        self.model = models.Sequential(name=self.name)
        # input_shape = (height, width, channels)
        # channels = 1 for grayscale, channels = 3 for RGB
        self.model.add(layers.InputLayer(input_shape=(self.img_height, self.img_width, 1),
                                         batch_size=self.batch_size))
        # Normalize pixel values to be between 0 and 1
        self.model.add(layers.Rescaling(1. / 255))
        self.model.add(layers.Conv2D(64, (5, 3), activation=layers.LeakyReLU(alpha=0.01)))
        self.model.add(layers.MaxPooling2D((2, 1), padding='same'))
        self.model.add(layers.Conv2D(128, (5, 3), activation=layers.LeakyReLU(alpha=0.01)))
        self.model.add(layers.MaxPooling2D(2, 1, padding='same'))
        self.model.add(layers.Conv2D(256, (5, 3), activation=layers.LeakyReLU(alpha=0.01)))
        self.model.add(layers.MaxPooling2D((2, 1), padding='same'))
        self.model.add(layers.Conv2D(512, (5, 3), activation=layers.LeakyReLU(alpha=0.01)))
        self.model.add(layers.MaxPooling2D((2, 1), padding='same'))
        self.model.add(layers.Flatten())  # the output of the last CNN block is flattened to a 1D vector
        self.model.add(layers.Dense(1))  # Dense layer = fully connected (FC) layer
        self.model.add(layers.Softmax())  # boolean mask
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  # default False
                           metrics=['accuracy', 'Precision', 'Recall'])


if __name__ == '__main__':

    cnn = CNN_5_20()
    cnn.compile()

    cnn = CNN_20_20()
    cnn.compile()

    cnn = CNN_60_20()
    cnn.compile()