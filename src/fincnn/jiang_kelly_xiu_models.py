from fincnn.base_model import BaseCNN
from tensorflow.keras import layers, models, losses


BATCH_SIZE = 128  # used by Jiang, Kelly, Xiu (2020), adjust to a lower value when running low on free system memory

# Weight initialization:
# Jiang, Kelly, Xiu(2020): We apply the Xavier initializer for weights in each layer.
# This promotes faster convergence by generating starting values for weights to
# ensure that prediction variance begins on a comparable scale to that of the labels.

w_init = 'glorot_uniform'  # The Glorot uniform initializer, also called Xavier uniform initializer.
# w_init = 'glorot_normal'  # The Glorot normal initializer, also called Xavier normal initializer.


class CNN_5(BaseCNN):

    def __init__(self, return_horizon=20, name='CNN_5_20', batch_size=BATCH_SIZE):
        super().__init__(name=name, image_horizon=5, return_horizon=return_horizon,
                         crossentropy='categorical', batch_size=batch_size)

    def compile(self):
        # Model design
        self.model = models.Sequential(name=self.name)
        # input_shape = (height, width, channels)
        # channels = 1 for grayscale, channels = 3 for RGB
        self.model.add(layers.InputLayer(input_shape=(self.img_height, self.img_width, 1),
                                         batch_size=self.batch_size))
        # Normalize pixel values to be between 0 and 1
        self.model.add(layers.Rescaling(1. / 255))

        # Padding (when stride is 1):
        # padding="SAME":  Output size is the same as input size (add zeros around edges)
        #                  This requires the filter window to slip outside input map, hence the need to pad.
        # padding="VALID": Filter window stays at valid position inside input map,
        #                  so output size shrinks by filter_size - 1. No padding occurs.
        # Jiang, Kelly, Xiu(2020): For elements at the image’s border,
        # we fill in the absent neighbor elements with zeros in order to compute
        # the convolution. This is a common CNN practice known as “padding,”
        # and ensures that the convolution output has the same dimension as the image itself

        # Batch normalization:
        # Jiang, Kelly, Xiu(2020): We use a batch normalization layer between the convolution and
        # non-linear activation within each building block to reduce covariate shift.

        # Dropout to avoid over-fitting:
        # Jiang, Kelly, Xiu(2020): We apply 50% dropout to the fully connected layer
        # (the relatively low parameterization in convolutional blocks avoids the need for dropout there).

        # BUILDING BLOCK 1:
        self.model.add(layers.Conv2D(64, (5, 3), kernel_initializer=w_init, padding='same'))  # no activation is applied
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 1)))

        # BUILDING BLOCK 2:
        self.model.add(layers.Conv2D(128, (5, 3), kernel_initializer=w_init, padding='same'))  # no activation is applied
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 1)))

        # FINAL BLOCK:
        self.model.add(layers.Flatten())  # the output of the last CNN block is flattened to a 1D vector
        self.model.add(layers.Dropout(rate=0.5))
        self.model.add(layers.Dense(2, kernel_initializer=w_init))  # Dense layer = fully connected layer
        # Jiang, Kelly, Xiu (2020): the number of parameters in a fully connected layer is calculated as L × 2,
        # where L is the length of the input vector and 2 corresponds to the two classification labels.
        #self.model.add(layers.Dense(1, kernel_initializer=w_init))  # needed when using BinaryCrossentropy
        self.model.add(layers.Softmax())  # Softmax layer
        # Wiki: The softmax function takes as input a vector z of K real numbers,
        # and normalizes it into a probability distribution consisting of K probabilities proportional
        # to the exponentials of the input numbers. That is, prior to applying softmax,
        # some vector components could be negative, or greater than one; and might not sum to 1;
        # but after applying softmax, each component will be in the interval [0,1],
        # and the components will add up to 1, so that they can be interpreted as probabilities.
        self.model.compile(optimizer='adam',
                           loss=losses.CategoricalCrossentropy(from_logits=False),
                           # to use BinaryCrossentropy add layer.Dense(1) above
                           #loss=losses.BinaryCrossentropy(from_logits=False),
                           # set from_logits=False for inputs to be interpreted as probabilities
                           metrics=['accuracy'])
        # Jiang, Kelly, Xiu (2020):
        # the fully connected layer has 15,360 neurons for 5_day-day model (same as mine)
        # the total number of parameters are 155,138 for the 5_day-day model (mine has 155,138 trainable parameters)


class CNN_20(BaseCNN):

    def __init__(self, return_horizon=20, name='CNN_20_20', batch_size=BATCH_SIZE):
        super().__init__(name=name, image_horizon=20, return_horizon=return_horizon,
                         crossentropy='categorical', batch_size=batch_size)

    def compile(self):
        # Model design
        self.model = models.Sequential(name=self.name)
        self.model.add(layers.InputLayer(input_shape=(self.img_height, self.img_width, 1),
                                         batch_size=self.batch_size))
        self.model.add(layers.Rescaling(1. / 255))

        # BUILDING BLOCK 1:
        self.model.add(layers.Conv2D(64, (5, 3),
                                     strides=(3, 1),  # the strides of the convolution along the height and width
                                     # dilation_rate=(2, 1),
                                     # Jiang, Kelly, Xiu (2020): We use horizontal and vertical strides of 1 and 2
                                     # and vertical dilation rate of 2 for 20_day-day images,
                                     # only on the first layer because raw images are sparse.
                                     # Problem:  In TensorFlow 2.7 specifying any `dilation_rate` value != 1 is
                                     # incompatible with specifying any stride value != 1.
                                     # This, however, should not result in loss of accuracy
                                     # (according to robustness checks, Appendix B, Table 18)
                                     kernel_initializer=w_init, padding='same'))  # no activation is applied
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 1)))

        # BUILDING BLOCK 2:
        self.model.add(
            layers.Conv2D(128, (5, 3), kernel_initializer=w_init, padding='same'))  # no activation is applied
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 1)))

        # BUILDING BLOCK 3:
        self.model.add(
            layers.Conv2D(256, (5, 3), kernel_initializer=w_init, padding='same'))  # no activation is applied
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 1)))

        # FINAL BLOCK:
        self.model.add(layers.Flatten())  # the output of the last CNN block is flattened to a 1D vector
        self.model.add(layers.Dropout(rate=0.5))
        self.model.add(layers.Dense(2, kernel_initializer=w_init))  # Dense layer = fully connected layer
        # self.model.add(layers.Dense(1, kernel_initializer=w_init))  # needed when using BinaryCrossentropy
        self.model.add(layers.Softmax())  # Softmax layer
        self.model.compile(optimizer='adam',
                           loss=losses.CategoricalCrossentropy(from_logits=False),
                           # to use BinaryCrossentropy add layer.Dense(1) above
                           # loss=losses.BinaryCrossentropy(from_logits=False),
                           # set from_logits=False for inputs to be interpreted as probabilities
                           metrics=['accuracy'])
        # Jiang, Kelly, Xiu (2020):
        # the fully connected layer has 46,080, neurons for 20_day-day model (mine has 30720)
        # the total number of parameters are 708,866 for the 20_day-day model (mine has 678,146 trainable parameters)


class CNN_60(BaseCNN):
    def __init__(self, return_horizon=20, name='CNN_60_20', batch_size=BATCH_SIZE):
        super().__init__(name=name, image_horizon=60, return_horizon=return_horizon,
                         crossentropy='categorical', batch_size=batch_size)

    def compile(self):
        # Model design
        self.model = models.Sequential(name=self.name)
        self.model.add(layers.InputLayer(input_shape=(self.img_height, self.img_width, 1),
                                         batch_size=self.batch_size))
        self.model.add(layers.Rescaling(1. / 255))

        # BUILDING BLOCK 1:
        self.model.add(layers.Conv2D(64, (5, 3),
                                     strides=(3, 1),  # the strides of the convolution along the height and width
                                     # dilation_rate=(3, 1),
                                     # Jiang, Kelly, Xiu (2020): We use horizontal and vertical strides of 1 and 3
                                     # and vertical dilation rate of 3 for 60_day-day images,
                                     # only on the first layer because raw images are sparse.
                                     # Problem:  In TensorFlow 2.7 specifying any `dilation_rate` value != 1 is
                                     # incompatible with specifying any stride value != 1.
                                     # This, however, should not result in loss of accuracy
                                     # (according to robustness checks, Appendix B, Table 18)
                                     kernel_initializer=w_init, padding='same'))  # no activation is applied
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 1)))

        # BUILDING BLOCK 2:
        self.model.add(layers.Conv2D(128, (5, 3), kernel_initializer=w_init, padding='same'))  # no activation is applied
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 1)))

        # BUILDING BLOCK 3:
        self.model.add(
            layers.Conv2D(256, (5, 3), kernel_initializer=w_init, padding='same'))  # no activation is applied
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 1)))

        # BUILDING BLOCK 4:
        self.model.add(
            layers.Conv2D(512, (5, 3), kernel_initializer=w_init, padding='same'))  # no activation is applied
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.01))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 1)))

        # FINAL BLOCK:
        self.model.add(layers.Flatten())  # the output of the last CNN block is flattened to a 1D vector
        self.model.add(layers.Dropout(rate=0.5))
        self.model.add(layers.Dense(2, kernel_initializer=w_init))  # Dense layer = fully connected layer
        # self.model.add(layers.Dense(1, kernel_initializer=w_init))  # needed when using BinaryCrossentropy
        self.model.add(layers.Softmax())  # Softmax layer
        self.model.compile(optimizer='adam',
                           loss=losses.CategoricalCrossentropy(from_logits=False),
                           # to use BinaryCrossentropy add layer.Dense(1) above
                           #loss=losses.BinaryCrossentropy(from_logits=False),
                           # set from_logits=False for inputs to be interpreted as probabilities
                           metrics=['accuracy'])
        # Jiang, Kelly, Xiu (2020):
        # the fully connected layer has 184,320 neurons for 60_day-day model (same as mine)
        # the total number of parameters are 2,952,962 for the 60_day-day model (mine has 2,952,962 trainable parameters)


if __name__ == '__main__':

    from fincnn.config.generate_datasets_config import RETURN_HORIZONS
    import argparse
    parser = argparse.ArgumentParser(description='Train and evaluate models.')
    parser.add_argument('-img-horizon', type=int, nargs='?',
                        help='fit/valuate models with the given image horizon (choose from 5, 20, 60), \
                        if omitted fit/valuate all models')
    parser.add_argument('--train', action='store_true',
                        help='train models only, do not run evaluation')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate models only')

    args = parser.parse_args()

    model_dict = zip(['jkx_CNN_5', 'jkx_CNN_20', 'jkx_CNN_60'], [CNN_5, CNN_20, CNN_60])
    if args.img_horizon is not None:
        model_dict = model_dict[f'jkx_CNN_{args.img_horizon}']

    for model_name, CNN in model_dict:
        for return_horizon in RETURN_HORIZONS:
            name = f'{model_name}_{return_horizon}'
            cnn = CNN(return_horizon=return_horizon, name=name)
            if args.train:
                cnn.fit()
            elif args.evaluate:
                cnn.evaluate()
            else:
                cnn.fit()
                cnn.evaluate()