from logging.handlers import WatchedFileHandler
import numpy as np
import imageio
from matplotlib import pyplot as plt
import sys 
import os

from scipy import spatial

import warnings
warnings.filterwarnings("error")

import time

def mykmeans(pixels, K):
    """
    Your goal of this assignment is implementing your own K-means.

    Input:
        pixels: data set. Each row contains one data point. For image
        dataset, it contains 3 columns, each column corresponding to Red,
        Green, and Blue component.

        K: the number of desired clusters. Too high value of K may result in
        empty cluster error. Then, you need to reduce it.

    Output:
        class: the class assignment of each data point in pixels. The
        assignment should be 0, 1, 2, etc. For K = 5, for example, each cell
        of class should be either 0, 1, 2, 3, or 4. The output should be a
        column vector with size(pixels, 1) elements.

        centroid: the location of K centroids in your result. With images,
        each centroid corresponds to the representative color of each
        cluster. The output should be a matrix with K rows and
        3 columns. The range of values should be [0, 255].
    """
   
    np.random.seed(0)  #fixing seed for reproducing random initialisation
    points = np.reshape(pixels, (pixels.shape[0]*pixels.shape[1], 3))  #flattening image
    classes = np.zeros((points.shape[0],1))
    centers = np.zeros((K, 3))

    #initializing centers by picking random points from image 
    for q in range(K):
        centers[q,:] = points[np.random.randint(0,points.shape[0]),:]

    iter = 0
    itermax = 500
    prev_centers = np.zeros(centers.shape)

    #algorithm stops when the centers stop moving
    #or when number of iterations exceeds maxiter
    while iter < itermax and np.linalg.norm(centers - prev_centers)!=0:
        prev_centers = centers.copy() 

        #assigning cluster to each data point based on euclidean distance
        for i in range(points.shape[0]):
            dist = ((points[i,:]-centers)**2).sum(axis=1)
            classes[i] = np.argmin(dist, axis = 0)

        #finding new centroid for each cluster by recalculating cluster mean
        for j in range(centers.shape[0]):
            cluster_points = points[np.repeat(classes == j, repeats = 3, axis = 1)].reshape(-1,3) #finding points in current cluster

            #if cluster is empty, we call the function with the next lower K value iteratively 
            #till we find a K value which does not result in an empty cluster
            try: 
                centers[j] = cluster_points.mean(axis = 0)
            except RuntimeWarning: 
                return mykmeans(pixels, K-1)
        iter = iter + 1
    classes = classes.astype(int)
    return classes, centers
    raise NotImplementedError

def mykmedoids(pixels, K):
    """
    Your goal of this assignment is implementing your own K-medoids.
    Please refer to the instructions carefully, and we encourage you to
    consult with other resources about this algorithm on the web.

    Input:
        pixels: data set. Each row contains one data point. For image
        dataset, it contains 3 columns, each column corresponding to Red,
        Green, and Blue component.

        K: the number of desired clusters. Too high value of K may result in
        empty cluster error. Then, you need to reduce it.

    Output:
        class: the class assignment of each data point in pixels. The
        assignment should be 0, 1, 2, etc. For K = 5, for example, each cell
        of class should be either 0, 1, 2, 3, or 4. The output should be a
        column vector with size(pixels, 1) elements.

        centroid: the location of K centroids in your result. With images,
        each centroid corresponds to the representative color of each
        cluster. The output should be a matrix with K rows and
        3 columns. The range of values should be [0, 255].
    """
    np.random.seed(0)  #fixing seed for reproducing random initialisation
    points = np.reshape(pixels, (pixels.shape[0]*pixels.shape[1], 3)) #flattening image
    classes = np.zeros((points.shape[0],1))
    centers = np.zeros((K, 3))

    #initializing centers by picking random points from image 
    for q in range(K):
        centers[q,:] = points[np.random.randint(0, points.shape[0]),:]

    iter = 0
    itermax = 500
    prev_centers = np.zeros(centers.shape)
    dist = np.zeros((1,K))

    J = 10000 #number of points closest to center to pick next center from

    #algorithm stops when the centers stop moving 
    #or when number of iterations exceeds maxiter
    while iter < itermax and np.linalg.norm(prev_centers - centers) != 0:
        prev_centers = centers.copy() 

        #assigning cluster to each data point based on euclidean distance
        for i in range(points.shape[0]):
            dist = spatial.distance.cdist(points[i][np.newaxis,:], centers, metric = "euclidean")
            classes[i] = np.argmin(dist)
        
        #recalculating centroid as point in cluster closest to mean
        for k in range(centers.shape[0]):
            cluster_points = points[np.repeat(classes == k, repeats = 3, axis = 1)].reshape(-1,3) 
            #calculating distance of all points in the cluster from the current center
            pairwise_dist_curcent = np.sum(spatial.distance.cdist(cluster_points, centers[k][np.newaxis,:], metric='euclidean'), axis=1)
            #finding J potential centers as the J points closest to current center
            pot_cent = cluster_points[np.argsort(pairwise_dist_curcent)][0:J]
            prev_dist = np.sum(pairwise_dist_curcent)
            #finding new center by minimising cost (sum of distances from all other points in the cluster)
            J = min(J, cluster_points.shape[0])
            for j in range(J):
                #if cluster is empty, we call the function with the next lower K value iteratively 
                #till we find a K value which does not result in an empty cluster
                try:
                    pairwise_dist_n1 = np.sum(np.sum(spatial.distance.cdist(cluster_points, pot_cent[j][np.newaxis,:], metric='euclidean'), axis=1))
                except: 
                    return mykmedoids(pixels, K-1)
                if(pairwise_dist_n1 < prev_dist):
                    centers[k] = pot_cent[j]
                    prev_dist = pairwise_dist_n1

        iter = iter + 1
    classes = classes.astype(int)
    return classes, centers
    raise NotImplementedError

def main():
    if(len(sys.argv) < 2):
        print("Please supply an image file")
        return

    image_file_name = sys.argv[1]
    K = 5 if len(sys.argv) == 2 else int(sys.argv[2])
    print(image_file_name, K)
    im = np.asarray(imageio.imread(image_file_name))

    fig, axs = plt.subplots(1, 2)
    
    classes, centers = mykmedoids(im, K)
    print(classes, centers)
    new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
    imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
    axs[0].imshow(new_im)
    axs[0].set_title('K-medoids')
    
    classes, centers = mykmeans(im, K)
    print(classes, centers)
    new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
    imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
    axs[1].imshow(new_im)
    axs[1].set_title('K-means')

    plt.show()

if __name__ == '__main__':
    main()
