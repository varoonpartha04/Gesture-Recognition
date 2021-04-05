from collections import defaultdict, Counter
from readAndWriteData import readCSV, write_csv_from_dict, write_json, read_features_csv
# from editDistance import calculateEditDistance
import dotProduct
from editDistance import displayEditDistance
from dwt import displayDTWDistance
import re
from os import listdir
from os.path import isfile, join
import os
import math
import numpy as np
import json
import pandas as pd

# PCA SVD LSD --------------------------------------
def option2(dir, fileName):
    f = open(dir)
    dataJson = json.load(f)
    ecuDist = {}
    # ecuList = []
    list1 = dataJson[fileName]
    for key in dataJson:
        list2 = dataJson[key]
        m = len(list1)
        total = 0.0
        for i in list2:
            total = total + pow(list1[i] - list2[i],2)
        dist = math.sqrt(total)
        # ecuList.append(dist)
        ecuDist[key] = dist

    # ecuDist = sorted(ecuDist.items(), key=lambda x: x[1])
    # print(ecuDist[0:10])
    # return(ecuDist[0:10])

    return(ecuDist)

def similarityOption2(dir):
    f = open(dir)
    dataJson = json.load(f)
    matrix_values = np.zeros((len(dataJson.keys()), len(dataJson.keys())))
    df = pd.DataFrame(matrix_values, columns = dataJson.keys(), index = dataJson.keys())
    for file in dataJson:
        sim = option2(dir, str(file))
        for key in sim:
            df[file][key] = 1/(1 + sim[key])
    return df

def displayOption2(dir, fileName):
    ecuDist = option2(dir, fileName)
    ecuDist = sorted(ecuDist.items(), key=lambda x: x[1])
    print("\nThe top 10 gestures similar to ", fileName, " are: " + str([key[0] for key in ecuDist[:10]]))

def process_input(user_option, input_file, vector_model):
    if user_option == 1:
        dotProduct.find_similar_files(input_file)
    elif user_option == 2:
        displayOption2('./Outputs/pca_transformed.json', input_file)
    elif user_option == 3:
        displayOption2('./Outputs/svd_transformed.json', input_file)
    elif user_option == 4:
        displayOption2('./Outputs/nmf_transformed.json', input_file)
    elif user_option == 5:
        displayOption2('./Outputs/lda_transformed.json', input_file)
    elif user_option == 6:
        displayEditDistance('./Outputs', input_file, '.wrd')
    elif user_option == 7:
        displayDTWDistance('./Outputs', input_file,  '.awrd')

def main():
    input_file = input('Enter file name to compare without extention: ')
    print("Choose one of the option to compare : \n1. Dot product \n2. PCA \n3. SVD \n4. NMF \n5. LDA \n6. Edit Distance\n7. DTW")
    user_option = int(input('Enter your choice: '))
    print("Choose one of the vector model from the following: \n1. TF \n2. TF-IDF")
    vector_model = input("Choose 1 - 2: ")
    process_input(user_option, input_file, vector_model)

main()
