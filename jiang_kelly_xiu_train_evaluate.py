from jiang_kelly_xiu_models import CNN_5, CNN_20, CNN_60
from generate_datasets_sample_params import RETURN_HORIZONS

if __name__ == '__main__':

    for return_horizon in RETURN_HORIZONS:
        for CNN, model_name in zip([CNN_5, CNN_20, CNN_60], ['jkx_CNN_5', 'jkx_CNN_20', 'jkx_CNN_60']):
            name = f'{model_name}_{return_horizon}'
            cnn = CNN(return_horizon=return_horizon)
            cnn.fit()
            cnn.evaluate()