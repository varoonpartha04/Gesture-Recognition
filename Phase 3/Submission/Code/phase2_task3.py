from collections import defaultdict, Counter
from readAndWriteData import readCSV, write_csv_from_dict, write_json, read_json, read_features_csv
import re
from os import listdir
from os.path import isfile, join
import os
import math
import numpy as np
import heapq
from operator import itemgetter
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from phase2_similarity import similarityOption2
from phase2_dwt import similarityMatrixDTW
from phase2_editDistance import similarityMatrixForED
from phase2_dotProduct import similarityMatrixForDP
from phase2_vectorModels import svd, nmf

def calculateDotProduct(filename):
    vector = "Outputs/vector.txt"
    data = read_json(vector)
    value = data["tf_value"][filename[:-4]]
    tf = data["tf_value"]
    dotProduct = {}
    sum = 0
    for key in value:
        sum = sum + value[key] * value[key]
    sum = math.sqrt(sum)

    for file in tf:
        dotproductsum = 0
        fileSum = 0
        for key in tf[file]:
            if key in value:
                dotproductsum = dotproductsum + value[key] * tf[file][key]
            fileSum = fileSum + tf[file][key] * tf[file][key]
        fileSum = math.sqrt(fileSum)
        dotProduct[file] = dotproductsum / (sum * fileSum)

    return dotProduct

def create_output(vector_model, k, input_data, semantic_features, indicator):
    if vector_model == "1":
        result = svd(k, input_data, semantic_features)
        write_to_file(result[0], result[1], output_directory+"/svd_sorted_similarity.json", output_directory+ "/svd_transformed_similarity.json")
    elif vector_model == "2":
        result = nmf(k, input_data, semantic_features)
        write_to_file(result[0], result[1], output_directory+"/nmf_sorted_similarity.json", output_directory+"/nmf_transformed_similarity.json")
    else:
        print("Please provide the correct input")

def write_to_file(membership, transformed_feature_set, file_name_1, file_name_2):
    write_json(membership,file_name_1)
    write_json(transformed_feature_set, file_name_2)

def process_input(user_option, vector_model, p, semantic_features):
    if user_option == 1:
        similarity_matrix = similarityMatrixForDP()
        create_output(vector_model, p, similarity_matrix, semantic_features, "dp")
        write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_dp.json")
    elif user_option == 2:
        similarity_matrix = similarityOption2('Outputs/pca_transformed.json')
        create_output(vector_model, p, similarity_matrix, semantic_features, "pca")
        write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_pca.json")
    elif user_option == 3:
        similarity_matrix = similarityOption2('Outputs/svd_transformed.json')
        create_output(vector_model, p, similarity_matrix, semantic_features, "svd")
        write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_svd.json")
    elif user_option == 4:
        similarity_matrix = similarityOption2('Outputs/nmf_transformed.json')
        create_output(vector_model, p, similarity_matrix, semantic_features, "nmf")
        write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_nmf.json")
    elif user_option == 5:
        similarity_matrix = similarityOption2('Outputs/lda_transformed.json')
        create_output(vector_model, p, similarity_matrix, semantic_features, "lda")
        write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_lda.json")
    elif user_option == 6:
        similarity_matrix = similarityMatrixForED('Outputs')
        create_output(vector_model, p, similarity_matrix, semantic_features, "ed")
        write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_ed.json")
    elif user_option == 7:
        similarity_matrix = similarityMatrixDTW('Outputs')
        create_output(vector_model, p, similarity_matrix, semantic_features, "dtw")
        write_json(similarity_matrix.to_dict(), "Outputs/similaritymatrix_dtw.json")
    else:
        print("Invalid option.")
output_directory = "Outputs"
def main():
    print("Choose one of the vector model from the following: \n1. SVD \n2. NMF")
    vector_model = input("Choose 1 - 2: ")
    p = int(input("Enter the number of components: "))
    semantic_features = ['LT-'+str(1+i) for i in range(p)]
    print("Choose one of the option to generate similarity matrix : \n1. dot product \n2. PCA \n3. SVD \n4. NMF \n5. LDA \n6. Edit Distance\n7. DTW")
    user_option = int(input('Enter your choice: '))
    process_input(user_option, vector_model, p, semantic_features)

if __name__ == "__main__":
    main()