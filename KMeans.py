
import pickle as pkl
import numpy as np
from sklearn.cluster import KMeans
from predict import centerIimages

def kmeans():




    with open('train.pkl', 'rb') as file:
        images, labels = pkl.load(file)


    images = centerIimages(images).reshape(images.shape[0],36,36)
    images = images[0:len(images), 4:32, 4:32]

    images = images[0:60000]
    labels = labels[0:60000]


    all = images,labels


    label_0_images = images[0:6000].copy()
    label_1_images = images[0:6000].copy()
    label_2_images = images[0:6000].copy()
    label_3_images = images[0:6000].copy()
    label_4_images = images[0:6000].copy()
    label_5_images = images[0:6000].copy()
    label_6_images = images[0:6000].copy()
    label_7_images = images[0:6000].copy()
    label_8_images = images[0:6000].copy()
    label_9_images = images[0:6000].copy()

    k0=0
    k1=0
    k2=0
    k3=0
    k4=0
    k5=0
    k6=0
    k7=0
    k8=0
    k9=0

    for i in range (60000):
        if all[1][i] ==0:
            label_0_images[k0] = all[0][i]
            k0+=1
        if all[1][i] ==1:
            label_1_images[k1] = all[0][i]
            k1+=1
        if all[1][i] == 2:
            label_2_images[k2] = all[0][i]
            k2 += 1
        if all[1][i] ==3:
            label_3_images[k3] = all[0][i]
            k3+=1
        if all[1][i] ==4:
            label_4_images[k4] = all[0][i]
            k4+=1
        if all[1][i] ==5:
            label_5_images[k5] = all[0][i]
            k5+=1
        if all[1][i] ==6:
            label_6_images[k6] = all[0][i]
            k6+=1
        if all[1][i] ==7:
            label_7_images[k7] = all[0][i]
            k7+=1
        if all[1][i] ==8:
            label_8_images[k8] = all[0][i]
            k8+=1
        if all[1][i] ==9:
            label_9_images[k9] = all[0][i]
            k9+=1


    label_0_images = label_0_images.reshape(label_0_images.shape[0],-1)
    label_1_images = label_1_images.reshape(label_1_images.shape[0], -1)
    label_2_images = label_2_images.reshape(label_2_images.shape[0], -1)
    label_3_images = label_3_images.reshape(label_3_images.shape[0], -1)
    label_4_images = label_4_images.reshape(label_4_images.shape[0], -1)
    label_5_images = label_5_images.reshape(label_5_images.shape[0], -1)
    label_6_images = label_6_images.reshape(label_6_images.shape[0], -1)
    label_7_images = label_7_images.reshape(label_7_images.shape[0], -1)
    label_8_images = label_8_images.reshape(label_8_images.shape[0], -1)
    label_9_images = label_9_images.reshape(label_9_images.shape[0], -1)

    size = 750


    print("training nr. 1")
    kmeans0 = KMeans(n_clusters=size, init='k-means++',precompute_distances=True,n_init=20).fit(label_0_images)
    print("training nr. 2")
    kmeans1 = KMeans(n_clusters=size, init='k-means++',precompute_distances=True,n_init=20).fit(label_1_images)
    print("training nr. 3")
    kmeans2 = KMeans(n_clusters=size, init='k-means++',precompute_distances=True,n_init=20).fit(label_2_images)
    print("training nr. 4")
    kmeans3 = KMeans(n_clusters=size, init='k-means++',precompute_distances=True,n_init=20).fit(label_3_images)
    print("training nr. 5")
    kmeans4 = KMeans(n_clusters=size, init='k-means++',precompute_distances=True,n_init=20).fit(label_4_images)
    print("training nr. 6")
    kmeans5 = KMeans(n_clusters=size, init='k-means++',precompute_distances=True,n_init=20).fit(label_5_images)
    print("training nr. 7")
    kmeans6 = KMeans(n_clusters=size, init='k-means++',precompute_distances=True,n_init=20).fit(label_6_images)
    print("training nr. 8")
    kmeans7 = KMeans(n_clusters=size, init='k-means++',precompute_distances=True,n_init=20).fit(label_7_images)
    print("training nr. 9")
    kmeans8 = KMeans(n_clusters=size, init='k-means++',precompute_distances=True,n_init=20).fit(label_8_images)
    print("training nr. 10")
    kmeans9 = KMeans(n_clusters=size, init='k-means++',precompute_distances=True,n_init=20).fit(label_9_images)

    kmeans_clusters_for_label_0 = kmeans0.cluster_centers_
    kmeans_clusters_for_label_1 = kmeans1.cluster_centers_
    kmeans_clusters_for_label_2 = kmeans2.cluster_centers_
    kmeans_clusters_for_label_3 = kmeans3.cluster_centers_
    kmeans_clusters_for_label_4 = kmeans4.cluster_centers_
    kmeans_clusters_for_label_5 = kmeans5.cluster_centers_
    kmeans_clusters_for_label_6 = kmeans6.cluster_centers_
    kmeans_clusters_for_label_7 = kmeans7.cluster_centers_
    kmeans_clusters_for_label_8 = kmeans8.cluster_centers_
    kmeans_clusters_for_label_9 = kmeans9.cluster_centers_

    images = images.reshape(images.shape[0], -1)
    kmeans_all_images = images[0:(size*10)].copy()
    kmeans_all_labels = labels[0:(size*10)].copy()

    kmeans_all_images[0:size] = kmeans_clusters_for_label_0
    kmeans_all_images[size:(2*size)] = kmeans_clusters_for_label_1
    kmeans_all_images[(2*size):(3*size)] = kmeans_clusters_for_label_2
    kmeans_all_images[(3*size):(4*size)] = kmeans_clusters_for_label_3
    kmeans_all_images[(4*size):(5*size)] = kmeans_clusters_for_label_4
    kmeans_all_images[(5*size):(6*size)] = kmeans_clusters_for_label_5
    kmeans_all_images[(6*size):(7*size)] = kmeans_clusters_for_label_6
    kmeans_all_images[(7*size):(8*size)] = kmeans_clusters_for_label_7
    kmeans_all_images[(8*size):(9*size)] = kmeans_clusters_for_label_8
    kmeans_all_images[(9*size):(10*size)] = kmeans_clusters_for_label_9




    kmeans_all_labels[0:size] = 0
    kmeans_all_labels[size:(2*size)] = 1
    kmeans_all_labels[(2*size):(3*size)] = 2
    kmeans_all_labels[(3*size):(4*size)] = 3
    kmeans_all_labels[(4*size):(5*size)] = 4
    kmeans_all_labels[(5*size):(6*size)] = 5
    kmeans_all_labels[(6*size):(7*size)] = 6
    kmeans_all_labels[(7*size):(8*size)] = 7
    kmeans_all_labels[(8*size):(9*size)] = 8
    kmeans_all_labels[(9*size):(10*size)] = 9

    kmeans_all_images = kmeans_all_images.reshape(kmeans_all_images.shape[0],-1)


    kmeans_all_images = (kmeans_all_images*255).astype(np.uint8)
    kmeans_all_labels.astype(np.uint8)



    output = open('kmeans_labels.pkl', 'wb')
    pkl.dump(kmeans_all_labels, output)
    output.close()

    output2 = open('kmeans_images.pkl', 'wb')
    pkl.dump(kmeans_all_images, output2)
    output2.close()

