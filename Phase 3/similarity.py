from collections import defaultdict, Counter
from readAndWriteData import readCSV, write_csv_from_dict, write_json, read_features_csv
# from editDistance import calculateEditDistance
from dotProduct import find_similar_files
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
    for key in ecuDist[0:10]:
        print(key[0])
