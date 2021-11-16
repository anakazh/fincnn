from models import CNN_5_20

if __name__ == '__main__':

    cnn = CNN_5_20()
    cnn.generate_images()  # ~10 mins
    cnn.fit(batch_size=5)
    cnn.plot_history()
    cnn.evaluate()

