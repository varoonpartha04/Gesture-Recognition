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


def calculateDotProduct(filename):
    vector = "Outputs/vector.txt"
    data = read_json(vector)
    value = data["tf_value"][filename]
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

def find_similar_files(filename, n = 10):
    dot_product = calculateDotProduct(filename)
    topitems = heapq.nlargest(n, dot_product.items(), key=itemgetter(1))
    topitemsasdict = dict(topitems)
    print("\nThe top 10 gestures similar to ", filename, " are: " + str(list(topitemsasdict.keys())))

def similarityMatrixForDP():
    vector = "Outputs/vector.txt"
    data = read_json(vector)
    data = data["tf_value"]

    matrix_values = np.zeros((len(data.keys()), len(data.keys())))
    df = pd.DataFrame(matrix_values, columns = data.keys(), index = data.keys())
    # print(df)
    for file in data:
        dotProduct = calculateDotProduct(str(file))
        for key in dotProduct:
            df[file][key] = dotProduct[key]
    return df

# displayDotProduct("8.wrd")
# print(similarityMatrixForDP())
