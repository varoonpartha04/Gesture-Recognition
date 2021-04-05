from pandas import DataFrame
import numpy as np
from pahse2_task3 import process_input
from os import listdir
from os.path import isfile, join
from readAndWriteData import read_json, writeCSV
import os
import pandas as pd
import matplotlib.pyplot as plt

# https://github.com/danieljunhee/Tutorial-on-Personalized-PageRank/blob/master/Personalized_PageRank_Tutorial.ipynb

def sim_graph_from_sim_max(similarity_matrix, gestures_list, num_comp):
    gestures_pd = {"gestureId": gestures_list}
    index = 0
    for row in similarity_matrix:
        gestures_pd[gestures_list[index]] = row
        index += 1
    gestures_dataframe = DataFrame(gestures_pd)
    gestures_dataframe = gestures_dataframe.set_index("gestureId")
    sim_graph = np.empty((0, len(similarity_matrix)))
    # sim_matrix = np.empty((0, len(eucl_dist)))
    for row in similarity_matrix:
        k_largest = np.argsort(-np.array(row))[1:num_comp + 1]
        sim_graph_row = [d if i in k_largest else 0 for i, d in enumerate(row)]
        sim_graph = np.append(sim_graph, np.array([sim_graph_row]), axis=0)
    # print(sim_graph)
    row_sums = sim_graph.sum(axis=1)
    sim_graph = sim_graph / row_sums[:, np.newaxis]
    return sim_graph

def ppr(similarity_graph, gestures_list, query_gestures, max_iterations=500, alpha=0.85):
    similarity_graph = similarity_graph.T
    matr_teleport = np.array([0 if gesture not in query_gestures else 1 for gesture in gestures_list]).reshape(len(gestures_list), 1)
    matr_teleport = matr_teleport / len(query_gestures)
    new_vals = matr_teleport
    old_vals = np.array((len(gestures_list), 1))
    # print(old_vals)
    iteration = 0
    while iteration < max_iterations and not np.array_equal(new_vals, old_vals):
        old_vals = new_vals.copy()
        new_vals = alpha * np.matmul(similarity_graph, old_vals) + (1 - alpha) * matr_teleport
        iteration += 1
    # print("Iterations: {}".format(iteration))
    new_vals = new_vals.ravel()
    a = (-new_vals).argsort()
    r = 1 #r = rank
    result = []
    for i in a:
        res = {"gestureId": gestures_list[i], "score": new_vals[i], "rank": r}
        result.append(res)
        r += 1
    return result

def plot_gesture(gesture, a,b, axs):
    for i in range(0,20):
        # plt.plot(gesture.iloc[i])
        axs[a,b].plot(gesture.iloc[i])
        if(a==0 and b==0): axs[a, b].set_title("gesture w")
        if(a==0 and b==1): axs[a, b].set_title("gesture x")
        if(a==1 and b==0): axs[a, b].set_title("gesture y")
        if(a==1 and b==1): axs[a, b].set_title("gesture z")
        # axs.axis([0,40,gesture.min().min(), gesture.max().max()])

def main():
    k = int(input("enter the no of components ")) # no of components
    n = int(input("no of input gestures= ")) # different user specified inputs
    m = int(input("no of dominant gestures= ")) # m -> no of most dominant gestures
    query_gestures = []
    for i in range(n):
        query_gestures.append(input("enter next gesture "))
    # query_gestures = ['1', '13', '14']
    print(query_gestures)
    
    # n=3
    # m=4
    # print(k,n,m)
    method=2 # 2 -> pca
    vector_model = "1" # 1-> svd, 2-> nmf
    semantic_features = ['LT-'+str(1+i) for i in range(k)]
    # print(semantic_features)
    process_input(method, vector_model, k, semantic_features)
    similarity_dict = read_json("Outputs/similaritymatrix_pca.json")
    gestures_list =  similarity_dict.keys()
    gest_df = pd.DataFrame(data={"gestures": gestures_list})
    gest_df.to_csv("./gestures_list.csv", sep=',',index=False)
    gestures_list = list(gestures_list)
    # print(gestures_list)
    similarity_matrix = np.array([[similarity_dict[row][column] for column in (similarity_dict[row])] for row in (similarity_dict)])

    similarity_graph = sim_graph_from_sim_max(similarity_matrix, gestures_list, k)

    # print(similarity_graph)
    open("similarity_graph.csv", "w").close()
    writeCSV(similarity_graph, "similarity_graph")
    # writeCSV(gestures_list, "gestures_list")
    # print(similarity_graph.shape)
    
    # print(similarity_graph)
    ppr_values = ppr(similarity_graph, gestures_list, query_gestures, max_iterations=500, alpha=0.85)
    # print(ppr_values)
    results = ppr_values[:m]
    print(results)
    print("Top {} images from Personalized page Rank are:".format(m))
    for gesture in results:
        print(gesture['gestureId'])
        gesturew = pd.read_csv("3_class_gesture_data/W/"+gesture['gestureId']+".csv", header=None)
        gesturex = pd.read_csv("3_class_gesture_data/X/"+gesture['gestureId']+".csv", header=None)
        gesturey = pd.read_csv("3_class_gesture_data/Y/"+gesture['gestureId']+".csv", header=None)
        gesturez = pd.read_csv("3_class_gesture_data/Z/"+gesture['gestureId']+".csv", header=None)
        fig, axs = plt.subplots(2, 2)
        plot_gesture(gesturew,0,0,axs)
        plot_gesture(gesturex,0,1,axs)
        plot_gesture(gesturey,1,0,axs)
        plot_gesture(gesturez,1,1,axs)
        plt.legend(['Series'+str(i) for i in range(1,21)],bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
        plt.show()

if __name__ == "__main__":
    main()

