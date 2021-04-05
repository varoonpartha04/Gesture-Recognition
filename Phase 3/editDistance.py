from collections import defaultdict, Counter
from readAndWriteData import readCSV, write_csv_from_dict, write_json, read_features_csv
import re
from os import listdir
from os.path import isfile, join
import os
import math
import numpy as np
import heapq
from operator import itemgetter
import pandas as pd


def read_all_files(dir, output_directory):
    files = [f for f in listdir(output_directory) if isfile(join(output_directory, f)) and re.search('\.wrd', f)]
    data = {}
    for file in files:
        file_path =  output_directory + "/" + file
        data[file] = read_features_csv(file_path)
    return data


def calculation(list1, list2):
    total_distance = 0
    if len(list1) != len(list2):
        print("Error")
    for i in range(80):
        n = len(list1[i])
        m = len(list2[i])
        dist = np.zeros((n,m))
        for j in range(n):
            dist[j][0] = j
        for j in range(m):
            dist[0][j] = j
        dist[0][0] = 0
        for j in range(1,n):
            for k in range(1,m):
                if list1[i][j] == list2[i][k]:
                    dist[j][k] = dist[j-1][k-1]
                else:
                    dist[j][k] = 1 + min(dist[j-1][k], dist[j][k-1], dist[j-1][k-1])
        total_distance = total_distance + dist[n-1][m-1]
    return total_distance


def calculateEditDistance(dir, filename):
    data = read_all_files(dir,dir)
    comparison = data[filename]
    list1 = []
    list2 = []
    distance = {}
    count = 1
    prev = re.split(",|'|\(|\)", comparison[0][0])
    smalllist=[]
    for rows in comparison:
        vals = re.split(",|'|\(|\)", rows[0])
        if(prev[7] == vals[7]):
            smalllist.append(rows[1])
        else:
            prev = vals
            list1.append(smalllist)
            smalllist = []
            smalllist.append(rows[1])
    list1.append(smalllist)

    for key in data:
        distance[key] = 0
        list2 = []
        prev = re.split(",|'|\(|\)", data[key][0][0])
        smalllist = []
        for rows in data[key]:
            vals = re.split(",|'|\(|\)", rows[0])
            if(prev[7] == vals[7]):
                smalllist.append(rows[1])
            else:
                prev = vals
                list2.append(smalllist)
                smalllist = []
                smalllist.append(rows[1])
        list2.append(smalllist)
        distance[key] = calculation(list1,list2)

    return distance


def similarityMatrixForED(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f)) and re.search('\.wrd', f)]
    # filesnew = []
    # for key in files:
    #     filesnew.append(key[:-4])
    # files = filesnew
    matrix_values = np.zeros((len(files),len(files)))
    df = pd.DataFrame(matrix_values, columns = files, index = files)
    data = read_all_files(dir,dir)
    for file1 in files:
        list1 = []
        list2 = []
        comparison1 = data[file1]
        prev = re.split(",|'|\(|\)", comparison1[0][0])
        smalllist=[]
        for rows in comparison1:
            vals = re.split(",|'|\(|\)", rows[0])
            if(prev[7] == vals[7]):
                smalllist.append(rows[1])
            else:
                prev = vals
                list1.append(smalllist)
                smalllist = []
                smalllist.append(rows[1])
        list1.append(smalllist)

        for file2 in files:
            if df[file1][file2] == 0:
                comparison2 = data[file2]
                list2 = []
                prev = re.split(",|'|\(|\)", comparison2[0][0])
                smalllist = []
                for rows in comparison2:
                    vals = re.split(",|'|\(|\)", rows[0])
                    if(prev[7] == vals[7]):
                        smalllist.append(rows[1])
                    else:
                        prev = vals
                        list2.append(smalllist)
                        smalllist = []
                        smalllist.append(rows[1])
                list2.append(smalllist)

                temp = calculation(list1,list2)
                temp = temp / 100
                df[file1][file2] = 1 / (1 + temp)
                df[file2][file1] = 1 / (1 + temp)
        # distances = calculateEditDistance(dir, file)
        # for key in distances:
        #     df[file][key] = 1 / (1 + distances[key])
    return df


def displayEditDistance(dir, filename):
    distance = calculateEditDistance(dir, filename)
    n = 10
    topitems = heapq.nsmallest(n, distance.items(), key=itemgetter(1))  # Use .iteritems() on Py2
    topitemsasdict = dict(topitems)
    print("\nThe top 10 gestures similar to ", filename, " are:")
    for key in topitemsasdict:
        print(key)


# displayEditDistance("/Users/varoon/MWDB/515ProjectFall2020/Phase 2/Code/Outputs", "4.wrd")
# print(similarityMatrixForED("/Users/varoon/MWDB/515ProjectFall2020/Phase 2/Code/Outputs"))
