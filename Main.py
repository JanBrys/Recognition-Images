import numpy as np
import pickle as pkl
import matplotlib as plt
from predict import predict
from matplotlib import pyplot as plt
from KMeans import kmeans


if __name__ == "__main__":
    with open('train.pkl', 'rb') as file:
        images, labels = pkl.load(file)

    kmeans()


    test_images = images[10000:20000]
    test_labels = labels[10000:20000]

    result = predict(test_images)

    summary = 0
    for i in range (len(result)):
        if(test_labels[i]==result[i]):
            summary+=1

    print(summary/10000)