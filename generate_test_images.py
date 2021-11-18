from models import CNN_5_20, CNN_20_20, CNN_60_20
from utils import get_raw_data

data = get_raw_data()

cnn = CNN_5_20()
dataslice = data.iloc[0:5]
cnn.gen_image(dataslice, 'testsave_05.png')

cnn = CNN_20_20()
dataslice = data.iloc[0:20]
cnn.gen_image(dataslice, 'testsave_20.png')

cnn = CNN_60_20()
dataslice = data.iloc[0:60]
cnn.gen_image(dataslice, 'testsave_60.png')