import os
import math
from scipy.spatial import distance
import argparse
import numpy as np
import scipy
from sklearn.cluster import KMeans
from readAndWriteData import readCSV, write_csv_from_dict, write_json, read_json, read_features_csv
from operator import itemgetter
from kMeansClustering import run as kMeansAlgo

def spec_clustering(gesture_names, sim_matrix,k):
    deg_list = []
    N = sim_matrix.shape[0]
    for i in range(N):
        temp = 0
        for j in range(sim_matrix.shape[1]):
            temp+=sim_matrix[i][j]
        deg_list.append(temp)
    
    deg_matrix = np.diag(np.array(deg_list))
    laplacian_matrix = deg_matrix-sim_matrix
    eigen_values,eigen_vectors = scipy.linalg.eigh(laplacian_matrix)
    V = eigen_vectors[0:N,0:k]
#     kmeans_algo = KMeans(n_clusters=k)
    center_group = kMeansAlgo(1, k, V, gesture_names)
    for k, v in sorted(center_group.items()):
             print(k+1, v)
#     print(center_group)
#     cluster_index = kmeans_algo.fit_predict(V)
#     for i in range(k):
#         members = np.where(cluster_index==i)[0]
#         gesture_name = ""
#         for j in members:
#             gesture_name+='{}.csv  '.format(gesture_names[j])
#         print("Cluster {} members: {}".format(i+1, gesture_name))
 
        # print(gesture_name)
    # print(cluster_index)

def process_input(userOption):
    if userOption == 2:
        similarity_dict = read_json("Outputs/similaritymatrix_pca.json")
    elif userOption == 3:
        similarity_dict = read_json("Outputs/similaritymatrix_svd.json")
        # write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_svd.json")
    elif userOption == 4:
        similarity_dict = read_json("Outputs/similaritymatrix_nmf.json")
        # write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_nmf.json")
    elif userOption == 5:
        similarity_dict = read_json("Outputs/similaritymatrix_lda.json")
        # write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_lda.json")
    elif userOption == 1:
        similarity_dict = read_json("Outputs/similaritymatrix_dp.json")
        # write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_dp.json")
    elif userOption == 6:
        similarity_dict = read_json("Outputs/similaritymatrix_ed.json")
        # write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_ed.json")
    else: #Default for dtw
        similarity_dict = read_json("Outputs/similaritymatrix_dtw.json")
        # write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_dtw.json")
    return similarity_dict

def process_clustering(clustering_technique, gesture_names, similarity_matrix, p):
    if clustering_technique == "1":
        center_group = kMeansAlgo(1, p, similarity_matrix, gesture_names)
        for k, v in sorted(center_group.items()):
            print(k+1, v)
    elif clustering_technique == "2":
        spec_clustering(gesture_names, similarity_matrix,p)


def main():
    p = int(input('Enter number of groups p: '))
    print("Choose one of the option to compare : \n1. Dot product \n2. PCA \n3. SVD \n4. NMF \n5. LDA \n6. Edit Distance\n7. DTW")
    user_option = int(input('Enter your choice: '))
    similarity_dict = process_input(user_option)
    gesture_names = list(map(itemgetter(0), similarity_dict.items()))
    similarity_matrix = np.zeros((len(gesture_names),len(gesture_names)))
    for g1,v1 in similarity_dict.items():
        for g2,v2 in v1.items():
            similarity_matrix[gesture_names.index(g1)][gesture_names.index(g2)] = v2
    
    print("Choose one of clustering technique: \n1. KMeans\n2. Spectral Clustering ")
    clustering_technique = input("Enter your choice: ")
    process_clustering(clustering_technique, gesture_names, similarity_matrix, p)

main()
