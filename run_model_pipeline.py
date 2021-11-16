from models import CNN_5_20

if __name__ == '__main__':

    cnn = CNN_5_20(batch_size=1)
    cnn.generate_images()  # ~10 mins
    cnn.fit()
    cnn.plot_history()
    cnn.evaluate()

