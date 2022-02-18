from fincnn.jiang_kelly_xiu_models import CNN_5, CNN_20, CNN_60
from fincnn.config.generate_datasets_config import RETURN_HORIZONS

if __name__ == '__main__':

    for CNN, model_name in zip([CNN_5, CNN_20, CNN_60], ['jkx_CNN_5', 'jkx_CNN_20', 'jkx_CNN_60']):
        if model_name in ['jkx_CNN_20', 'jkx_CNN_60']:
            continue
        for return_horizon in RETURN_HORIZONS:
            name = f'{model_name}_{return_horizon}'
            cnn = CNN(return_horizon=return_horizon, name=name)
            cnn.evaluate()