from readAndWriteData import read_json;
import numpy as np;
import pandas as pd;
from math import sqrt;
from random import randrange
from task1 import ppr
from task3 import LSH
from readAndWriteData import readCSV
import csv
from pandas import DataFrame
from scipy.stats import norm
import random

import math
import heapq


def LSHFeedback(similarity_dict, L, k, query_gesture_id, t):
	# L = input("Enter Number of layers L")	
	# k = input("Enter number of hashes per layer k")
	# print(gestures_list)
	# query_gesture_id = input("Enter query gesture id")
	# t = input("Enter the value of t")
	# L = int(L)
	# k = int(k)
	# t = int(t)
    gestures_list =  similarity_dict.keys()
    gestures_list = list(gestures_list)
    similarity_matrix = np.array([[similarity_dict[row][column] for column in (similarity_dict[row])] for row in (similarity_dict)])

    query_gesture = similarity_matrix[gestures_list.index(query_gesture_id)]

    gesture_vectors = np.delete(similarity_matrix,gestures_list.index(query_gesture_id),0)
    gestures_list = np.delete(gestures_list,gestures_list.index(query_gesture_id),0)
    w = 3
    b = np.zeros((L,k))
    for i in range(L):
        for j in range(k):
            b[i][j] = random.uniform(0,w)
            n,d = gesture_vectors.shape

    gaussian = norm()
    LSH_family = np.zeros((L,k,d))
    for i in range(L):
        for j in range(k):
            LSH_family[i][j] = gaussian.rvs(size=d)
    hash_tables = [{}]*L
    for i in range(L):
        for j in range(n):
            temp = []
            x = gesture_vectors[j]
            for m in range(k):
                temp.append(math.floor(np.dot(LSH_family[i][m],x)+b[i][m]/w))
            if str(temp) not in hash_tables[i]:
                hash_tables[i][str(temp)] = []
            hash_tables[i][str(temp)].append(gestures_list[j])
    possible_points = []
    for i in range(L):
        temp = []
        for j in range(k):
            temp.append(math.floor(np.dot(LSH_family[i][j],query_gesture)+b[i][j]/w))
        if str(temp) not in hash_tables[i]:
            possible_points.append('None')
        else:
            possible_points.append(hash_tables[i][str(temp)])
    distances = {}
    c = 0
    alpha = 0 
    for i in possible_points:
        if not i=='None':
            c+=1
            for j in i:
                alpha+=1
                distances[str(j)] = np.linalg.norm(gesture_vectors[np.where(gestures_list==j)[0][0]]-query_gesture)
    result = heapq.nsmallest(t,distances)
    return result,c,alpha

def feedback(L, k, query_gesture, t, feedbackDict): 
    irrelevent = []
    for key, value in feedbackDict.items():
        if value == -1:
            irrelevent.append(key)

    print("Irrelavant data: ", irrelevent)
    similarity_dict = read_json("Outputs/similaritymatrix_pca.json")
    gestures_list =  similarity_dict.keys()
    gestures_list = list(gestures_list)
    for num in irrelevent:
        print(num)
        # print(similarity_dict[query_gesture][num])
        similarity_dict[query_gesture][num] = 0
        similarity_dict[num][query_gesture] = 0
    # similarity_matrix = np.array([[similarity_dict[row][column] for column in (similarity_dict[row])] for row in (similarity_dict)])

    r,c,a = LSHFeedback(similarity_dict, L, k, query_gesture, t)
    return r
    
def initialComputation(L, k, query_gesture_id, t):
    # similarity_dict = read_json("Outputs/similaritymatrix_pca.json")
    # gestures_list =  similarity_dict.keys()
    # gest_df = pd.DataFrame(data={"gestures": gestures_list})
    # gest_df.to_csv("./gestures_list.csv", sep=',',index=False)
    # gestures_list = list(gestures_list)
	# # print(gestures_list)
    # similarity_matrix = np.array([[similarity_dict[row][column] for column in (similarity_dict[row])] for row in (similarity_dict)])

    r,c,a = LSH(L, k, query_gesture_id, t)
    return r



def main():
    
    L = 4
    k = 6
    query_gesture = '14'
    t = 4
    irrelevent = ["252", "255"]
    # initialComputation(L, k, query_gesture, t)
    print('-------')
    feedback(L, k, query_gesture, t, irrelevent)
    


# if __name__ == "__main__":
#     main()