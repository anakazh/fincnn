import tensorflow as tf
from tensorflow.keras import models, callbacks
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data.experimental import cardinality
from sklearn import metrics
import pandas as pd
import numpy as np
import json
from pathlib import Path
from multiprocessing import cpu_count
from utils.image_utils import img_specs


CPU_COUNT = cpu_count()  # used for model training and image generation
                         # TODO: check if cpu_count() works in colab


class BaseCNN:
    """
    Basic functions for CNN classification model: image generation, training, evaluation of model accuracy
    Add model design to BaseCNN.compile() method
    """
    def __init__(self, name, image_horizon, return_horizon, batch_size=1, crossentropy='categorical'):
        self.name = name
        self.image_horizon = image_horizon
        self.return_horizon = return_horizon
        self.img_height = img_specs[image_horizon].img_height
        self.img_width = img_specs[image_horizon].img_width
        self.volume_height = img_specs[image_horizon].volume_height

        self.train_dataset_path = Path(f'data/processed/{self.image_horizon}_day/ret_{self.return_horizon}train/')
        self.test_dataset_path = Path(f'data/processed/{self.image_horizon}_day/ret_{self.return_horizon}/test/')

        self.model_dir_path = Path(f'models/{self.name}/')
        self.model_dir_path.mkdir(parents=True, exist_ok=True)
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
        # TODO: add noise filtering

        if not (any(self.train_dataset_path.joinpath('pos').iterdir())
                and any(self.train_dataset_path.joinpath('neg').iterdir())):
            raise Exception('No train dataset found, generate images first!')
        self.describe_train_dataset()

        dataset = self.load_dataset(path=self.train_dataset_path)
        dataset_size = cardinality(dataset).numpy()
        train_size = int(train_validation_split * dataset_size)
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)

        max_epochs = 10
        stopping_rule = callbacks.EarlyStopping(patience=2)
        # Jiang, Kelly, Xiu (2020): We use early stopping to halt training once the validation
        # sample loss function fails to improve for two consecutive epochs
        self.history = self.model.fit(x=train_dataset, y=None, batch_size=self.batch_size,
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

    def predict(self):
        if self.model is None:
            raise Exception(f'No {self.name} found, train the model first!')
        raise NotImplementedError

    def evaluate(self):
        if self.history is None:
            raise Exception(f'No {self.name} found, train the model first!')

        if not(any(self.test_dataset_path.joinpath('pos').iterdir())
               and any(self.test_dataset_path.joinpath('neg').iterdir())):
            raise Exception('No test dataset found, generate images first!')

        test_dataset = self.load_dataset(self.test_dataset_path)
        self.describe_test_dataset()

        y_pred = self.model.predict(test_dataset)
        y_true = np.array([a for a in test_dataset.map(lambda x, y: y).unbatch().as_numpy_iterator()])

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

        if self.crossentropy =='categorical':
            print(metrics.classification_report(y_true, y_pred))
        elif self.crossentropy == 'binary':
            print(pd.DataFrame(model_metrics))

        json.dump(model_metrics, open(self.metrics_path, 'w'))