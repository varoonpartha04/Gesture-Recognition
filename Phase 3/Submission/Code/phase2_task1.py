from readAndWriteData import read_features_csv, write_json
from phase2_vectorModels import pca, svd, nmf, lda
from sklearn.preprocessing import StandardScaler
from readAndWriteData import read_json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from readAndWriteData import write_json

def create_output(vector_model, k, input_data, semantic_features, output_directory):
    if vector_model == "1":
        result = pca(k, input_data, semantic_features)
        write_to_file(result[0], result[1], output_directory+"/pca_sorted.json", output_directory+"/pca_transformed.json")
    elif vector_model == "2":
        result = svd(k, input_data, semantic_features)
        write_to_file(result[0], result[1], output_directory+"/svd_sorted.json", output_directory+"/svd_transformed.json")
    elif vector_model == "3":
        result = nmf(k, input_data, semantic_features)
        write_to_file(result[0], result[1], output_directory+"/nmf_sorted.json", output_directory+"/nmf_transformed.json")
    elif vector_model == "4":
        result = lda(k, input_data, semantic_features)
        write_to_file(result[0], result[1], output_directory+"/lda_sorted.json", output_directory+"/lda_transformed.json")
    else:
        print("Please provide the correct input")

def write_to_file(membership, transformed_feature_set, file_name_1, file_name_2):
    write_json(membership,file_name_1)
    write_json(transformed_feature_set, file_name_2)

# converts dictionary to a dataframe
def covert_dict_to_dataframe(user_dict):
    transformed_dict = {(i): user_dict[i] 
                               for i in user_dict.keys()}
    matrix = pd.DataFrame.from_dict(transformed_dict, orient='index')
    matrix.fillna(0, inplace=True)
    input_scaled = pd.DataFrame(preprocessing.scale(matrix, with_mean=False),columns = matrix.columns, index = matrix.index)
    return input_scaled

def process_vector_representation(vector_representation):
    if vector_representation == "1":
        return "tf_value"
    elif vector_representation == "2":
        return "tf_idf_value"
    else:
        print("Wrong Input")

def main():
    directory_name = input("Enter the path of directory: ")
    output_directory = "Outputs"
    output_file_name = "/vector.txt"
    input_data = read_json(output_directory + output_file_name)
    print("Choose one of the vector representation from the following: \n1. TF \n2. TF-IDF")
    vector_representation = input("Choose 1 - 2: ")
    measure = process_vector_representation(vector_representation)
    print("Choose one of the model from the following: \n1. PCA \n2. SVD \n3. NMF \n4. LDA")
    vector_model = input("Choose 1 - 4: ")
    k = int(input("Enter the number of components(k): "))
    semantic_features = ['LT-'+str(1+i) for i in range(k)]
    user_dict = covert_dict_to_dataframe(input_data[measure])
    create_output(vector_model, k, user_dict, semantic_features, output_directory)
    print("Latent Semantic and new feature set created.")

main()
