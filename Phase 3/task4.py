from readAndWriteData import read_features_csv, write_json, read_json
import numpy as np;
import pandas as pd;
from math import sqrt;
from random import randrange
from task1 import ppr
from readAndWriteData import readCSV
import heapq
from operator import itemgetter
from os import listdir
from os.path import isfile, join
import re
import os
from task3 import LSH

pvalue_initial = {}
qvalue_initial = {}
pvalue_revised = {}
qvalue_revised = {}

def get_neighbors(train, test_row, num_neighbors):
    distances = {}
    for row in train:
        dist = euclidean_distance(train[row], test_row)
        distances[row] = dist

    topitems = heapq.nsmallest(num_neighbors + 1, distances.items(), key=itemgetter(1))
    topitemsasdict = dict(topitems)
    return topitemsasdict


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in row1:
        if i in row2:
            distance += (row1[i] - row2[i])**2
        else:
            distance += row1[i]**2
    for i in row2:
        if i not in row1:
            distance += row2[i]**2

    return sqrt(distance)


def read_all_files(dir, output_directory):
    files = [f for f in listdir(output_directory) if isfile(join(output_directory, f)) and re.search('\.wrd', f)]
    data = {}
    for file in files:
        file_path =  output_directory + "/" + file
        data[file] = read_features_csv(file_path)
    return data


def order_results(features, valid, filename, N):
    full_filename = filename + '.wrd'
    for word in features[full_filename]:
        pvalue_initial[word[1]] = 0.5
        qvalue_initial[word[1]] = (N + 0.5) / (N + 1.0)

    wordsinfiles = {}
    for file in features:
        wordsinfiles[file[:-4]] = {}
        for word in features[file]:
            wordsinfiles[file[:-4]][word[1]] = 1

    R = 0
    for word in pvalue_initial:
        count_invalid = 0
        for file in valid:
            if word in wordsinfiles[file]:
                count_invalid = count_invalid + 1
        qvalue_initial[word] = (count_invalid + 0.5) / (N - R + 1.0)

    probval = {}
    for files in valid:
        temp = 1
        for words in wordsinfiles[files]:
            if words in pvalue_initial:
                comp = (1 - qvalue_initial[words]) / qvalue_initial[words]
                temp = temp * comp
        probval[files] = temp

    topitems = heapq.nlargest(N, probval.items(), key=itemgetter(1))
    topitemsasdict = dict(topitems)
    return topitemsasdict


def reorder_results(features, valid, filename, N):
    full_filename = filename + '.wrd'
    for word in features[full_filename]:
        pvalue_revised[word[1]] = 1
        qvalue_revised[word[1]] = 1

    wordsinfiles = {}
    for file in features:
        wordsinfiles[file[:-4]] = {}
        for word in features[file]:
            wordsinfiles[file[:-4]][word[1]] = 1

    # print(wordsinfiles)
    R = 0
    I = 0
    for file in valid:
        if valid[file] == 1:
            R = R + 1
        elif valid[file] == -1:
            I = I + 1

    # print(R)
    for word in pvalue_revised:
        count_valid = 0
        count_invalid = 0
        for file in valid:
                if word in wordsinfiles[file]:
                    if valid[file] == 1:
                        count_valid = count_valid + 1
                    elif valid[file] == -1:
                        count_invalid = count_invalid + 1
        pvalue_revised[word] = (count_valid + 0.5) / (R + 1.0)
        qvalue_revised[word] = (count_invalid + 0.5) / (I + 1.0)

    # print(pvalue_revised)
    # print(qvalue_revised)
    probval = {}
    for files in valid:
        temp = 1
        for words in wordsinfiles[files]:
            if words in pvalue_revised:
                comp = (pvalue_revised[words] * (1 - qvalue_revised[words])) / (qvalue_revised[words] * (1 - pvalue_revised[words]))
                temp = temp * comp
        probval[files] = temp

    topitems = heapq.nlargest(N, probval.items(), key=itemgetter(1))
    topitemsasdict = dict(topitems)
    return topitemsasdict


def findOrder(L, k, query, no_neigh):
    valid = {}
    answers = {}
    output_directory = "Outputs"
    features = read_all_files(".", output_directory)

    neighborlist,c,a = LSH(L, k, query, no_neigh)
    print(neighborlist)

    for file in features:
        if file[:-4] == query:
            continue
        valid[file[:-4]] = -1
    # print(valid)
    result = order_results(features, valid, query, len(valid))
    loop = 0
    for i in result:
        if i == query:
            continue
        elif i in neighborlist:
            if loop == no_neigh:
                break
            answers[i] = result[i]
            loop = loop+1

    write_val = {
    "p_values" : pvalue_initial,
    "q_values" : qvalue_initial }
    write_json(write_val, "Outputs/probabilistic_feedback_initial_values.json")

    return answers.keys()


def findReordering(query, no_neigh, valid):
    answers = {}
    output_directory = "Outputs"
    features = read_all_files(".", output_directory)



    result = reorder_results(features, valid, query, no_neigh)
    loop = 0
    for i in result:
        if i == query:
            continue
        elif i in valid:
            if loop == no_neigh:
                break
            answers[i] = result[i]
            loop = loop+1

    write_val = {
    "p_values" : pvalue_revised,
    "q_values" : qvalue_revised }
    write_json(write_val, "Outputs/probabilistic_feedback_revised_values.json")

    return answers.keys()


def main():
    answer = findOrder(15, 5, "1", 10)
    print(answer)
    valid = {}
    # valid = {'17': 1, '250': 0, '569': -1, '279': 0, '566': -1, '267': 0, '564': -1, '2': 1, '577': -1, '19': 1}

    loop = 0
    for file in answer:
        if len(file) <= 2:
            valid[file] = 1
        else:
            valid[file] = -1
        loop = loop + 1
    print(valid)
    answer = findReordering("1", 10, valid)
    print(answer)
    # for i in answer:
        # print(i)


# main()
