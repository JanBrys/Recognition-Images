
import pickle as pkl
import numpy as np

def predict(x):



    with open('kmeans_images.pkl', 'rb') as file:
        images = pkl.load(file)

    with open('kmeans_labels.pkl', 'rb') as file2:
        labels = pkl.load(file2)

    images = images.astype(np.float64) / 255
    x = centerIimages(x)
    x = x.reshape(x.shape[0], 36, 36)
    x = x[0:len(x), 4:32, 4:32]
    x = x.reshape(x.shape[0], -1)
    distances_matrix = distances(images, x)
    result = np.zeros([len(x), 1], dtype=np.dtype(np.int64))
    result = labels[np.argmin(distances_matrix, axis=1)]
    result = result.reshape(len(result), -1)
    return result


    pass


def centerIimages(x):
    x = x.reshape((x.shape[0], 36, 36))
    for i in range(len(x)):
        x_indexes, y_indexes = np.where(x[i] >= 0.55)
        centered_image = np.roll(x[i], int(17.5 - (y_indexes.sum() / len(y_indexes))), axis=1)
        centered_image = np.roll(centered_image, int(17.5 - (x_indexes.sum() / len(x_indexes))), axis=0)
        x[i] = centered_image
    return x.reshape(len(x), -1)
    pass

def distances(my_images, test_images):
    dists = -2 * np.dot(test_images, my_images.T) + np.sum(my_images ** 2, axis=1) + np.sum(test_images ** 2, axis=1)[:,np.newaxis]
    return dists


"""
SOURCES:

https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
https://docs.scipy.org/doc/numpy/reference/generated/numpy.roll.html
https://medium.com/dataholiks-distillery/l2-distance-matrix-vectorization-trick-26aa3247ac6c
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
https://scikit-learn.org/stable/modules/clustering.html
https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68

"""