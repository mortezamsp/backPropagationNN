def load_mnist():
    import os
    import gzip
    import numpy as np

    with gzip.open("train-labels-idx1-ubyte.gz", 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open("train-images-idx3-ubyte.gz", 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


