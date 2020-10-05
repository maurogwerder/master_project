from sklearn import cluster
import numpy as np
import cv2
import os


def main():
    """
    Preprocesses all the images of a folder with the kmeans-clustering algorithm.
    :return: None
    """
    MYPATH = os.getcwd() + '/img_norm/'
    file_names = os.listdir(MYPATH)
    print(f'All files to be processed: {file_names}\n')

    for file in file_names:
        print(f'{file} is being processed. \n')
        # Read the data as greyscale
        img = cv2.imread(MYPATH + file, 0)
        # Group similar grey levels using 8 clusters
        values, labels = km_clust(img, n_clusters=5)
        # Create the segmented array from labels and values
        img_segm = np.choose(labels, values)
        # Reshape the array as the original image
        img_segm.shape = img.shape

        cv2.imwrite(os.getcwd() + '/img_kmeans/kmeans_' + file, img_segm)


# function from https://www.idtools.com.au/segmentation-k-means-python/
def km_clust(array, n_clusters):
    """
    Clusters an image by the kmeans-clustering algorithm.
    :param array: a 2D-array
    :param n_clusters: The amount of clusters in which the values should be binned
    :return: values: 1D-array of the binning values
             labels: a 2D-array of the input-array-size with the cluster positions in the image
    """
    # Create a line array
    X = array.reshape((-1, 1))
    # Define the k-means clustering problem
    k_m = cluster.KMeans(n_clusters=n_clusters, n_init=4)
    # Solve the k-means clustering problem
    k_m.fit(X)
    # Get the coordinates of the clusters centres as a 1D array
    values = k_m.cluster_centers_.squeeze()
    # Get the label of each point
    labels = k_m.labels_
    return values, labels


main()

