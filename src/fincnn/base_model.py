from tensorflow.keras import models, callbacks
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data.experimental import cardinality
from sklearn import metrics
import pandas as pd
import numpy as np
import json
from pathlib import Path
from multiprocessing import cpu_count
from fincnn.utils.image_utils import IMG_SPECS as img_specs
from fincnn.utils.data_utils import get_return_from_filename
import shutil
from fincnn.config.paths_config import PROCESSED_DATA_PATH
from tqdm import tqdm


CPU_COUNT = cpu_count()  # used for model training and image generation


class BaseCNN:
    """
    Basic functions for CNN classification model: image generation, training, evaluation of model accuracy
    Add model design to BaseCNN.compile() method
    """
    def __init__(self, name, image_horizon, return_horizon,
                 batch_size=1, crossentropy='categorical', train_noise_threshold=0):
        self.name = name
        self.image_horizon = image_horizon
        self.return_horizon = return_horizon
        self.train_noise_threshold = train_noise_threshold
        self.img_height = img_specs[image_horizon].img_height
        self.img_width = img_specs[image_horizon].img_width
        self.volume_height = img_specs[image_horizon].volume_height

        self.model_dir_path = Path(f'models/{self.name}/')
        self.train_dataset_path = Path(f'models/{self.name}/images/train/')
        self.train_dataset_path.mkdir(parents=True, exist_ok=True)
        self.test_dataset_path = Path(f'models/{self.name}/images/test/')
        self.test_dataset_path.mkdir(parents=True, exist_ok=True)

        self.model_path = self.model_dir_path.joinpath(f'{self.name}.h5')
        self.history_path = self.model_dir_path.joinpath(f'history.json')
        self.metrics_path = self.model_dir_path.joinpath(f'metrics.json')

        self.batch_size = batch_size  # should be the same for self.fit(), self.predict() and self.evaluate()
        self.crossentropy = crossentropy

        if self.model_path.exists():
            self.model = models.load_model(self.model_path)
            self.history = json.load(open(self.history_path))
        else:
            self.compile()  # assigns model to self.model attribute
            self.history = None

    def label_images(self, source_path, target_path, ret_threshold):
        for sign in ['pos', 'neg']:
            target_path.joinpath(sign).mkdir(parents=True, exist_ok=True)

        for filepath in tqdm(source_path.iterdir(), desc='Labelling images in progress '):
            filename = filepath.name
            ret = get_return_from_filename(filename, self.return_horizon)
            if ret > abs(ret_threshold):
                savepath = target_path.joinpath('pos', filename)
            elif ret < -abs(ret_threshold):
                savepath = target_path.joinpath('neg', filename)
            # ignore ret == 0
            shutil.copy(filepath, savepath)

    def compile(self):
        raise Exception('Model design missing')

    def __repr__(self):
        self.model.summary()

    def describe_train_dataset(self):
        train_pos_count = len([f for f in self.train_dataset_path.joinpath('pos').rglob('*')])
        train_sample_size = train_pos_count + len([f for f in self.train_dataset_path.joinpath('neg').rglob('*')])
        train_pos_frac = train_pos_count / train_sample_size
        print(f'Train sample size: {train_sample_size}')
        print(f'Positive returns in the train sample: {round(train_pos_frac*100, 2)}%')

    def describe_test_dataset(self):
        test_pos_count = len([f for f in self.test_dataset_path.joinpath('pos').rglob('*')])
        test_sample_size = test_pos_count + len([f for f in self.test_dataset_path.joinpath('neg').rglob('*')])
        test_pos_frac = test_pos_count / test_sample_size
        print(f'Test sample size: {test_sample_size}')
        print(f'Positive returns in the test sample: {round(test_pos_frac*100, 2)}%')

    def load_dataset(self, path):
        dataset = image_dataset_from_directory(path,
                                               label_mode=self.crossentropy,
                                               # 'categorical' for categorical crossentropy,
                                               # 'binary' for binary crossentropy
                                               batch_size=self.batch_size,  # default: 32
                                               color_mode='grayscale', # pixel values range from 0 to 255
                                                                       # (0 represents black and 255 represents white)
                                               image_size=(self.img_height, self.img_width))
        return dataset

    def fit(self, train_validation_split=0.7):
        if not any(PROCESSED_DATA_PATH.joinpath(f'{self.image_horizon}_day/train/').iterdir()):
            raise Exception('No train dataset found, generate images first!')

        self.label_images(source_path=PROCESSED_DATA_PATH.joinpath(f'{self.image_horizon}_day/train/'),
                          target_path=self.train_dataset_path,
                          ret_threshold=self.train_noise_threshold)

        self.describe_train_dataset()

        dataset = self.load_dataset(path=self.train_dataset_path)
        dataset_size = cardinality(dataset).numpy()
        train_size = int(train_validation_split * dataset_size)
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)

        self.model.summary()
        
        max_epochs = 10
        stopping_rule = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1, patience=2)
        # Jiang, Kelly, Xiu (2020): We use early stopping to halt training once the validation
        # sample loss function fails to improve for two consecutive epochs
        self.history = self.model.fit(x=train_dataset, #y=None,
                                      batch_size=self.batch_size,
                                      epochs=max_epochs,
                                      callbacks=[stopping_rule],
                                      verbose=2,
                                      validation_data=val_dataset,
                                      max_queue_size=5,  # default is 10
                                      workers=CPU_COUNT,
                                      use_multiprocessing=True)  # default is False
        self.history = self.history.history
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f)
        self.model.save(self.model_path)
        # remove directory with images sorted into pos and neg, originals stay at PROCESSED_DATA_PATH
        shutil.rmtree(self.train_dataset_path)

    def predict(self):
        if self.model is None:
            raise Exception(f'No {self.name} found, train the model first!')
        raise NotImplementedError

    def evaluate(self):
        if self.history is None:
            raise Exception(f'No {self.name} found, train the model first!')

        if not any(PROCESSED_DATA_PATH.joinpath(f'{self.image_horizon}_day/test/').iterdir()):
            raise Exception('No test dataset found, generate images first!')

        self.label_images(source_path=PROCESSED_DATA_PATH.joinpath(f'{self.image_horizon}_day/test/'),
                          target_path=self.test_dataset_path,
                          ret_threshold=0)

        self.describe_test_dataset()
        print('Loading dataset in progress')
        test_dataset = self.load_dataset(self.test_dataset_path)

        filenames = [x.split('/')[-1] for x in test_dataset.file_paths]
        tickers_permnos = [x.split('_')[0]+'_'+x.split('_')[1] for x in filenames]
        ret_dates = [x.split('_')[2] for x in filenames]
        rets = [get_return_from_filename(x, self.return_horizon) for x in filenames]

        print('Predicting returns in progress')
        y_pred = self.model.predict(test_dataset)
        print('Accuracy computation in progress')
        y_true = np.array([a for a in test_dataset.map(lambda x, y: y).unbatch().as_numpy_iterator()])

        if 'pos' == test_dataset.class_names[0]:
            pred_prob_pos = [x[0] for x in y_pred]
            pred_prob_neg = [x[1] for x in y_pred]
        else:
            pred_prob_neg = [x[0] for x in y_pred]
            pred_prob_pos = [x[1] for x in y_pred]

        df = pd.DataFrame(list(zip(tickers_permnos, ret_dates, rets, pred_prob_pos, pred_prob_neg)),
                          columns=['Ticker_permno', 'ret_date', 'ret', 'pred_prob_pos', 'pred_prob_neg'])
        # save predictions for test sample as dataframe with tickers and timestamps
        df.to_csv(self.model_dir_path.joinpath('test_sample_pred_prob.csv'), index=False)

        if self.crossentropy == 'categorical':
            y_pred = y_pred.round()  # y_pred is probability, transform to 0 and 1
            accuracy = metrics.accuracy_score(y_true, y_pred)
            precision = metrics.precision_score(y_true, y_pred, average=None)
            precision = dict(zip(test_dataset.class_names, precision))
            recall = metrics.recall_score(y_true, y_pred, average=None)
            recall = dict(zip(test_dataset.class_names, recall))
            f1score = metrics.f1_score(y_true, y_pred, average=None)
            f1score = dict(zip(test_dataset.class_names, f1score))

        elif self.crossentropy == 'binary':
            # TODO: test on a model with binary crossentropy
            accuracy = (y_pred == y_true).sum() / len(y_true)
            precision, recall, f1score, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

        model_metrics = {'accuracy': accuracy, 'precision': precision,
                         'recall': recall, 'f1score': f1score}

        if self.crossentropy == 'categorical':
            print(metrics.classification_report(y_true, y_pred))
        elif self.crossentropy == 'binary':
            print(pd.DataFrame(model_metrics))

        json.dump(model_metrics, open(self.metrics_path, 'w'))

        # remove directory with images sorted into pos and neg, originals stay at PROCESSED_DATA_PATH
        shutil.rmtree(self.test_dataset_path)

    def trading_signal_df(self, longonly=True):
        """
        Trading signal based on predictions of the model; strategy holding period equals self.return_horizon
        longonly - True for strategy with no short positions
        :return: dataframe with daily trading signal (columns correspond to securities), equally-weighted
        """
        df = pd.read_csv(self.model_dir_path.joinpath('test_sample_pred_prob.csv'), parse_dates=True)
        df['ret_date'] = df.ret_date.apply(str).apply(pd.Timestamp)
        df['pred_prob_pos_signal'] = df.pred_prob_pos.round()
        df['pred_prob_neg_signal'] = df.pred_prob_neg.round()
        df['signal'] = df.pred_prob_pos_signal - df.pred_prob_neg_signal
        returns = df.pivot(columns=['Ticker_permno'], index='ret_date', values='ret')
        signal = df.pivot(columns=['Ticker_permno'], index='ret_date', values='signal')
        signal_resampled = signal.resample(pd.offsets.BDay(self.return_horizon)).first()
        signal = signal_resampled.fillna(0).reindex(returns.index).ffill()
        if longonly:
            signal = signal.replace(-1, 0)
            signal = signal.div(signal.sum(axis=1), axis=0)
        else:
            # long-short, total exposure maintained at 1
            # (i.e. when short side has 30% weight, the long side is set to 130%)
            signal_count = signal[signal != 0].count(axis=1)
            signal = signal.div(signal_count, axis=0)
            adj_factor = signal[signal < 0].sum(axis=1).mul(-1)
            signal = signal + signal[signal > 0].mul(adj_factor, axis=0).fillna(0)
        return signal

